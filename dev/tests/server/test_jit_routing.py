import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from dev.tests.server_fixtures import (
    FakeRuntime,
    FakeTokenizer,
    chat_body,
    make_frontend,
)
from server import backend as backend_api
from server import protocol as native_wire
from server import server as api
from server.frontend import ResponseStore
from server.jit import ModelEntry, ModelRouter, ModelSlot
from server.latency import LatencyMetrics
from server.residency import Residency

ALPHA = "alpha/model"
BETA = "beta/model:Q4"


class JitHarness:
    """A server of two models in JIT mode, whose engines are fake runtimes:
    alpha with a 128-token context and vision, beta with 256 tokens and
    language only."""

    def __init__(self, ttl=300.0, timeout=2):
        self.tokenizer = FakeTokenizer()
        self.loads = []
        self.unloads = []
        readiness = {
            ALPHA: native_wire.ReadyEvent(4, 128, True),
            BETA: native_wire.ReadyEvent(4, 256, False),
        }
        entries = [
            ModelEntry(model, Path("/nonexistent") / model, readiness=ready)
            for model, ready in readiness.items()
        ]

        def load(model):
            runtime = FakeRuntime()
            runtime.readiness = readiness[model]
            self.loads.append((model, runtime))
            return backend_api.NativeBackend(
                runtime, self.tokenizer, request_logger=lambda _record: None
            )

        def unload(model, backend):
            self.unloads.append(model)
            backend.close()

        self.residency = Residency(load, unload, ttl, poll=0.01)
        latencies = LatencyMetrics()
        responses = ResponseStore()
        for entry in entries:
            entry.frontend = make_frontend(
                self.tokenizer,
                ModelSlot(self.residency, entry.model),
                entry.model,
                entry.readiness.max_context_tokens,
                timeout,
                2,
                vision=entry.readiness.vision,
                latencies=latencies,
                response_store=responses,
            )
        self.router = ModelRouter(
            entries,
            self.residency,
            latencies=latencies,
            response_store=responses,
            request_timeout=timeout,
        )
        self.server = api.FrontendServer(
            ("127.0.0.1", 0), self.router, request_capacity=4
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(
            method, path, None if body is None else json.dumps(body), headers
        )
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        if "json" not in (response.getheader("Content-Type") or ""):
            return response.status, payload
        return response.status, json.loads(payload)

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.residency.close()


class JitRoutingTests(unittest.TestCase):
    def harness(self, **options):
        harness = JitHarness(**options)
        self.addCleanup(harness.close)
        return harness

    def test_models_lists_every_model_with_its_own_context(self):
        harness = self.harness()
        status, listing = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(
            [
                (item["id"], item["context_length"], item["vision"], item["loaded"])
                for item in listing["data"]
            ],
            [(ALPHA, 128, True, False), (BETA, 256, False, False)],
        )
        self.assertEqual(
            [item["input_modalities"] for item in listing["data"]],
            [["text", "image", "pdf"], ["text"]],
        )
        self.assertEqual(
            [(item["name"], item["description"]) for item in listing["models"]],
            [(ALPHA, "128 context · vision"), (BETA, "256 context · language only")],
        )
        self.assertNotIn("description", listing["data"][0])
        status, model = harness.request("GET", f"/v1/models/{BETA}")
        self.assertEqual((status, model["max_model_len"]), (200, 256))
        status, _ = harness.request("GET", "/v1/models/gamma/model")
        self.assertEqual(status, 404)

    def test_a_request_loads_the_model_it_names_and_switches_models(self):
        harness = self.harness()
        status, response = harness.request(
            "POST", "/v1/chat/completions", chat_body(model=ALPHA)
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(response["model"], ALPHA)
        self.assertEqual([model for model, _ in harness.loads], [ALPHA])
        self.assertEqual(len(harness.loads[0][1].requests), 1)
        _, listing = harness.request("GET", "/v1/models")
        self.assertEqual([item["loaded"] for item in listing["data"]], [True, False])
        # A request for the other model unloads the first.
        status, response = harness.request(
            "POST", "/v1/chat/completions", chat_body(model=BETA)
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(response["model"], BETA)
        self.assertEqual([model for model, _ in harness.loads], [ALPHA, BETA])
        self.assertEqual(harness.unloads, [ALPHA])
        self.assertEqual(len(harness.loads[1][1].requests), 1)
        _, status_report = harness.request("GET", "/status")
        self.assertEqual(status_report["jit"]["resident"], BETA)
        self.assertEqual(status_report["instance"]["model"], ALPHA)

    def test_an_unknown_model_is_refused_without_a_load(self):
        harness = self.harness()
        status, response = harness.request(
            "POST", "/v1/chat/completions", chat_body(model="gamma/model")
        )
        self.assertEqual(status, 404)
        self.assertEqual(response["error"]["code"], "model_not_found")
        self.assertEqual(harness.loads, [])

    def test_a_request_without_a_model_goes_to_the_resident_or_default(self):
        harness = self.harness()
        self.assertEqual(harness.router.select(None).model, ALPHA)
        self.assertEqual(harness.router.select("").model, ALPHA)
        harness.residency.lease(BETA).release()
        self.assertEqual(harness.router.select(None).model, BETA)
        self.assertIsNone(harness.router.select("gamma/model"))

    def test_prompt_only_routes_load_no_model(self):
        harness = self.harness()
        status, response = harness.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": BETA, "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(status, 200, response)
        self.assertGreater(response["input_tokens"], 0)
        self.assertEqual(harness.loads, [])

    def test_the_server_is_ready_with_no_model_loaded(self):
        harness = self.harness()
        self.assertEqual(harness.request("GET", "/ready"), (200, {"status": "ready"}))
        status, report = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertIsNone(report["jit"]["resident"])
        self.assertEqual(report["jit"]["state"], "idle")
        self.assertEqual(
            [model["id"] for model in report["jit"]["models"]], [ALPHA, BETA]
        )
        status, released = harness.request("POST", "/v1/memory/release")
        self.assertEqual((status, released["released"]), (200, True))
        self.assertEqual(harness.loads, [])
        status, metrics = harness.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(b"splash_info", metrics)

    def test_an_idle_model_unloads_after_the_ttl(self):
        harness = self.harness(ttl=0.05)
        status, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(model=ALPHA)
        )
        self.assertEqual(status, 200)
        deadline = time.monotonic() + 2
        while harness.residency.resident is not None:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual(harness.unloads, [ALPHA])
        self.assertTrue(harness.loads[0][1].closed)


class ModelEntryTests(unittest.TestCase):
    def test_description_names_family_format_context_and_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.json").write_text(
                json.dumps({"family": "Qwen3.8-27B", "target_format": "gguf"})
            )
            entry = ModelEntry(
                "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
                root,
                readiness=native_wire.ReadyEvent(4, 262144, True),
            )
            self.assertEqual(
                entry.description,
                "Qwen3.8-27B · GGUF UD-Q4_K_M · 256K context · vision",
            )
        # Without a readable record or a probe, a generic description.
        self.assertEqual(
            ModelEntry("a/b", Path("/nonexistent")).description, "Splash model"
        )


if __name__ == "__main__":
    unittest.main()
