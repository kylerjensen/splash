import io
import math
import signal
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests.server_fixtures import main_args
from server import backend as backend_api
from server import diagnostics
from server import protocol as native_wire
from server import server as api
from server.origins import ANY_ORIGIN


class ServerMainTests(unittest.TestCase):
    def test_sigterm_uses_the_normal_main_cleanup_path(self):
        args = main_args()
        runtime = mock.Mock()
        runtime.readiness = native_wire.ReadyEvent(
            max_concurrent_requests=4, max_context_tokens=262144, vision=False
        )
        backend = mock.Mock()
        server = mock.Mock(server_port=8000)
        tokenizer = object()
        handlers = {}
        order = []
        closing_handlers = []

        def install(signum, handler):
            if handler is api._interrupt:
                handlers[signum] = handler
                return signal.SIG_DFL
            handlers[signum] = handler
            self.assertIn(signum, (signal.SIGTERM, signal.SIGINT))
            if handler is not signal.SIG_IGN:
                self.assertEqual(signum, signal.SIGINT)

        backend.close.side_effect = lambda: closing_handlers.append(dict(handlers))

        def serve():
            self.assertEqual(set(handlers), {signal.SIGTERM, signal.SIGINT})
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        def bind(*_, **__):
            return server

        server.server_bind.side_effect = lambda: order.append("bind")

        runtime.wait_ready.side_effect = lambda: order.append("runtime") or True
        server.serve_forever.side_effect = serve
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ) as runtime_type,
            mock.patch.object(
                api, "NativeBackend", return_value=backend
            ) as backend_type,
            mock.patch.object(api, "ConstraintFactory", return_value=object()),
            mock.patch.object(api, "Frontend", return_value=mock.Mock()) as app_type,
            mock.patch.object(api, "FrontendServer", side_effect=bind),
            mock.patch.object(api.signal, "signal", side_effect=install),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            api.main()
        self.assertEqual(
            handlers, {signal.SIGTERM: signal.SIG_IGN, signal.SIGINT: signal.SIG_IGN}
        )
        # While the engine exits gracefully, a second Ctrl+C stops it now.
        (closing,) = closing_handlers
        self.assertIs(closing[signal.SIGTERM], signal.SIG_IGN)
        runtime.kill.assert_not_called()
        closing[signal.SIGINT](signal.SIGINT, None)
        runtime.kill.assert_called_once_with()
        command = ["splash", "serve-native", "model", "auto", "auto"]
        runtime_type.assert_called_once_with(
            command,
            relaunch_command=mock.ANY,
            startup_timeout=api.NATIVE_START_TIMEOUT,
            pending_limit=1,
        )
        # Every relaunch keeps the context the first engine served.
        relaunch_command = runtime_type.call_args.kwargs["relaunch_command"]
        self.assertEqual(
            relaunch_command(native_wire.ReadyEvent(4, 61433, False)),
            [*command, "--relaunch-context", "61433"],
        )
        backend_type.assert_called_once_with(
            runtime,
            tokenizer,
            request_logger=diagnostics.print_request,
            thinking_end=backend_api.DEFAULT_THINKING_END,
        )
        self.assertEqual(app_type.call_args.args[3], 262144)
        # No --request-timeout, no deadline.
        self.assertEqual(app_type.call_args.args[4], math.inf)
        self.assertEqual(app_type.call_args.args[5], 4)
        runtime.wait_ready.assert_called_once_with()
        server.serve_forever.assert_called_once()
        server.server_close.assert_called_once()
        backend.close.assert_called_once()
        server.server_bind.assert_called_once_with()
        server.server_activate.assert_called_once_with()
        self.assertEqual(order, ["bind", "runtime"])

    def test_main_takes_vision_from_the_ready_event(self):
        args = main_args()
        for vision in (True, False):
            with self.subTest(vision=vision):
                runtime = mock.Mock()
                runtime.readiness = native_wire.ReadyEvent(4, 131072, vision)
                with (
                    mock.patch.object(api, "parse_args", return_value=args),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "ChatTemplates") as templates_type,
                    mock.patch.object(
                        api.engine_runtime,
                        "MultiplexedRuntime",
                        return_value=runtime,
                    ),
                    mock.patch.object(api, "NativeBackend"),
                    mock.patch.object(api, "ConstraintFactory"),
                    mock.patch.object(api, "Frontend") as app_type,
                    mock.patch.object(
                        api,
                        "FrontendServer",
                        return_value=mock.Mock(server_port=8000),
                    ),
                    mock.patch.object(api.signal, "signal"),
                    mock.patch.object(api, "print_status") as status,
                ):
                    api.main()
                self.assertIs(app_type.call_args.kwargs["vision"], vision)
                self.assertIs(
                    app_type.call_args.kwargs["chat_templates"],
                    templates_type.return_value,
                )
                describe = templates_type.return_value.describe
                describe.assert_called_once_with()
                self.assertIn(
                    mock.call(f"Chat template · {describe.return_value}"),
                    status.call_args_list,
                )
                self.assertIn(
                    mock.call(
                        "Ready · test-model · context 128K"
                        + ("" if vision else " · language only")
                        + " · http://127.0.0.1:8000"
                    ),
                    status.call_args_list,
                )

    def test_main_wires_allowed_origins_and_warns_without_a_key(self):
        warning = mock.call(
            "Warning · --allowed-origin '*' without --api-key lets every web "
            "page open in a browser that reaches this server use it",
            error=True,
        )
        for origins, key, warned in (
            ([("tauri", "localhost", None)], None, False),
            ([ANY_ORIGIN], None, True),
            ([ANY_ORIGIN], "key", False),
        ):
            with self.subTest(origins=origins, key=key):
                runtime = mock.Mock()
                runtime.readiness = native_wire.ReadyEvent(
                    max_concurrent_requests=4, max_context_tokens=131072, vision=False
                )
                with (
                    mock.patch.object(
                        api,
                        "parse_args",
                        return_value=main_args(allowed_origin=origins, api_key=key),
                    ),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "ChatTemplates"),
                    mock.patch.object(
                        api.engine_runtime,
                        "MultiplexedRuntime",
                        return_value=runtime,
                    ),
                    mock.patch.object(api, "NativeBackend"),
                    mock.patch.object(api, "ConstraintFactory"),
                    mock.patch.object(api, "Frontend"),
                    mock.patch.object(
                        api,
                        "FrontendServer",
                        return_value=mock.Mock(server_port=8000),
                    ) as server_type,
                    mock.patch.object(api.signal, "signal"),
                    mock.patch.object(api, "print_status") as status,
                ):
                    api.main()
                self.assertEqual(
                    server_type.call_args.kwargs["allowed_origins"], origins
                )
                self.assertIs(warning in status.call_args_list, warned)

    def test_main_cleans_up_when_native_startup_fails_after_reserved_bind(self):
        args = main_args(max_context=128, max_memory=32 * 1024**3)
        runtime = mock.Mock()
        runtime.wait_ready.side_effect = api.engine_runtime.EngineUnhealthy("late")
        backend = mock.Mock()
        server = mock.Mock()
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ),
            mock.patch.object(api, "NativeBackend", return_value=backend),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        server.server_bind.assert_called_once_with()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        backend.close.assert_called_once()
        self.assertIn("Error · late", stderr.getvalue())

    def test_main_rejects_an_unservable_chat_template_before_starting_native(self):
        server = mock.Mock()
        with (
            mock.patch.object(api, "parse_args", return_value=main_args()),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer,
                "from_pretrained",
                return_value=SimpleNamespace(chat_template=None),
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        runtime.assert_not_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        self.assertIn(
            "Error · the tokenizer defines no chat template", stderr.getvalue()
        )

    def test_main_rejects_port_conflict_before_loading_or_starting_native(self):
        args = main_args(port=8000)
        server = mock.Mock()
        server.server_bind.side_effect = OSError(48, "Address already in use")
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.AutoTokenizer, "from_pretrained") as tokenizer,
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        tokenizer.assert_not_called()
        runtime.assert_not_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        self.assertIn("Address already in use", stderr.getvalue())


class JitMainTests(unittest.TestCase):
    """main() serving several models (JIT mode)."""

    MODELS = [("alpha/model", "/models/alpha"), ("beta/model", "/models/beta")]

    def run_main(self, ready, serve=None, args=None):
        """Run main() with fake engines announcing `ready`, by model root;
        the runtimes, backends, frontends and server it made."""
        args = args or main_args(
            model_root=None, tokenizer=None, model=None, jit_model=self.MODELS
        )
        runtimes, backends = [], []

        def runtime(command, **_options):
            made = mock.Mock(readiness=ready[command[2]])
            made.command = command
            made.wait_ready.return_value = True
            runtimes.append(made)
            return made

        def backend(made, *_args, **_options):
            created = mock.Mock(runtime=made)
            backends.append(created)
            return created

        server = mock.Mock(server_port=8000)
        server.serve_forever.side_effect = (
            KeyboardInterrupt if serve is None else lambda: serve(server)
        )
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", side_effect=lambda path, **_: path
            ) as tokenizers,
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates"),
            mock.patch.object(api, "ConstraintFactory"),
            mock.patch.object(api, "thinking_end_tokens"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", side_effect=runtime
            ),
            mock.patch.object(api, "NativeBackend", side_effect=backend),
            mock.patch.object(api, "Frontend") as frontends,
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            try:
                api.main()
            except SystemExit as exit_:
                self.exit_code = exit_.code
        self.tokenizer_paths = [call.args[0] for call in tokenizers.call_args_list]
        self.stdout = stdout.getvalue()
        return runtimes, backends, frontends, server

    def test_startup_probes_every_model_and_keeps_the_first_loaded(self):
        ready = {
            "/models/alpha": native_wire.ReadyEvent(4, 131072, True),
            "/models/beta": native_wire.ReadyEvent(4, 65536, False),
        }
        seen = {}

        def serve(server):
            seen["app"] = server.app
            raise KeyboardInterrupt

        runtimes, backends, frontends, server = self.run_main(ready, serve)
        self.assertEqual(
            self.tokenizer_paths, ["/models/alpha/tokenizer", "/models/beta/tokenizer"]
        )
        # The second model is probed first and stopped; the first, the
        # default, is probed last and stays loaded until the server stops.
        self.assertEqual(
            [made.command[2] for made in runtimes], ["/models/beta", "/models/alpha"]
        )
        for made in runtimes:
            self.assertEqual(made.command[3], "auto")
            self.assertNotIn("--relaunch-context", made.command)
        beta, alpha = backends
        beta.close.assert_called_once_with()
        alpha.close.assert_called_once_with()
        router = seen["app"]
        self.assertIsInstance(router, api.ModelRouter)
        self.assertEqual(router.model_names, ("alpha/model", "beta/model"))
        # Each frontend has its model's context, concurrency and vision.
        self.assertEqual(
            [
                (call.args[2], call.args[3], call.args[5], call.kwargs["vision"])
                for call in frontends.call_args_list
            ],
            [("alpha/model", 131072, 4, True), ("beta/model", 65536, 4, False)],
        )
        self.assertEqual(
            [type(call.args[1]) for call in frontends.call_args_list],
            [api.ModelSlot, api.ModelSlot],
        )
        server.server_activate.assert_called_once_with()
        self.assertIn("Model · beta/model · context 64K · language only", self.stdout)
        self.assertIn("Ready · 2 models · alpha/model loaded · TTL 5m", self.stdout)

    def test_a_later_load_serves_the_context_its_probe_did(self):
        ready = {
            "/models/alpha": native_wire.ReadyEvent(4, 131072, True),
            "/models/beta": native_wire.ReadyEvent(4, 65536, False),
        }
        loaded = {}

        def serve(server):
            lease = server.app.residency.lease("beta/model")
            loaded["engine"] = lease.engine
            lease.release()
            raise KeyboardInterrupt

        runtimes, backends, _, server = self.run_main(ready, serve)
        later = runtimes[-1]
        self.assertEqual(later.command[2], "/models/beta")
        self.assertEqual(later.command[-2:], ["--relaunch-context", "65536"])
        self.assertIs(loaded["engine"], backends[-1])
        # The default model unloaded to make room.
        backends[1].close.assert_called_once_with()

    def test_a_model_that_fails_its_probe_stops_startup(self):
        ready = {
            "/models/alpha": native_wire.ReadyEvent(4, 131072, True),
            "/models/beta": None,
        }
        runtimes, backends, _, server = self.run_main(ready)
        self.assertEqual(self.exit_code, 1)
        self.assertEqual(len(runtimes), 1)
        backends[0].close.assert_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()

    def test_parse_args_takes_jit_models_and_ignores_one_models_options(self):
        args = [
            "--jit-model",
            "alpha/model",
            "/models/alpha",
            "--jit-model",
            "beta/model:Q4",
            "/models/beta",
            "--max-context",
            "64K",
            "--model-ttl",
            "off",
        ]
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            parsed = api.parse_args(args)
        self.assertEqual(
            parsed.jit_model,
            [("alpha/model", "/models/alpha"), ("beta/model:Q4", "/models/beta")],
        )
        self.assertIsNone(parsed.max_context)
        self.assertEqual(parsed.model_ttl, math.inf)
        self.assertIn("--max-context applies to one model", stderr.getvalue())
        for refused in (
            [*args[:3], "--jit-model", "alpha/model", "/models/other"],
            [*args[:3], "--jit-model", "not a model", "/models/other"],
            [*args[:3], "model", "--tokenizer", "tokenizer", "--model", "a/b"],
            [],
        ):
            with (
                self.subTest(arguments=refused),
                mock.patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit),
            ):
                api.parse_args(refused)


if __name__ == "__main__":
    unittest.main()
