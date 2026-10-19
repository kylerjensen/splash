"""Serving several models from one process (JIT mode): the frontend of each
model, and the router the HTTP handler asks for the one a request names.

Each model has its own tokenizer, chat templates and frontend, built at
startup, and an engine that runs only while the model is resident
(residency.Residency). Its frontend's backend is a ModelSlot, which reaches
that engine while a request holds the model's lease.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .errors import APIError
from .residency import Residency

# The weights state /v1/memory/release reports with no model loaded: nothing
# holds weights.
UNLOADED_WEIGHTS = {"released": True, "restoring": False}


def context_text(tokens):
    """A context limit as the startup summary and /v1/models show it."""
    return f"{tokens // 1024}K" if tokens % 1024 == 0 else f"{tokens:,}"


@dataclass
class ModelEntry:
    """One model JIT mode serves, and what startup learned of it."""

    model: str
    root: Path
    # Read once at startup; what loads serve.
    tokenizer: object = None
    chat_templates: object = None
    constraint_factory: object = None
    thinking_end: object = None
    # The ReadyEvent of the model's startup probe, which every later load
    # must announce again.
    readiness: object = None
    frontend: object = None

    @cached_property
    def description(self):
        """What /v1/models describes the model as, from its installation's
        model.json and its probe: its family, format, context and input.
        Read once the probe is done."""
        try:
            record = json.loads((self.root / "model.json").read_text())
        except (OSError, UnicodeDecodeError, ValueError):
            record = {}
        if not isinstance(record, dict):
            record = {}
        parts = []
        family = record.get("family")
        if isinstance(family, str) and family:
            parts.append(family)
        target_format = record.get("target_format")
        if isinstance(target_format, str) and target_format:
            variant = self.model.partition(":")[2]
            parts.append(" ".join(filter(None, (target_format.upper(), variant))))
        if self.readiness is not None:
            parts.append(f"{context_text(self.readiness.max_context_tokens)} context")
            parts.append("vision" if self.readiness.vision else "language only")
        return " · ".join(parts) or "Splash model"


class ModelSlot:
    """A model's backend as its frontend and the handler use it: the
    engine of the model while it is resident."""

    def __init__(self, residency: Residency, model: str):
        self.residency = residency
        self.model = model

    def _engine(self):
        engine = self.residency.engine(self.model)
        if engine is None:
            raise APIError(
                503, f"model {self.model} is not loaded; retry", "model_unavailable"
            )
        return engine

    def refusal(self):
        engine = self.residency.engine(self.model)
        return None if engine is None else engine.refusal()

    def submit(self, job):
        self._engine().submit(job)

    def cancel(self, job):
        engine = self.residency.engine(self.model)
        if engine is not None:
            engine.cancel(job)

    def is_ready(self):
        engine = self.residency.engine(self.model)
        return True if engine is None else engine.is_ready()

    def status(self):
        engine = self.residency.engine(self.model)
        if engine is None:
            return {"ready": True, "engine": "unloaded"}
        return engine.status()

    def release_weights(self):
        engine = self.residency.engine(self.model)
        if engine is None:
            return {"weights": dict(UNLOADED_WEIGHTS)}
        return engine.release_weights()


class RouterBackend:
    """What the handler asks of the backend before it knows a request's
    model, and of the server as a whole."""

    def __init__(self, router):
        self.router = router
        self.closing = False

    def refusal(self):
        if self.closing:
            return APIError(503, "server is shutting down", "server_shutdown")
        return None

    def is_ready(self):
        # A server with no model loaded is ready: a request loads its model.
        return not self.closing

    def release_weights(self):
        resident = self.router.residency.resident
        if resident is None:
            return {"weights": dict(UNLOADED_WEIGHTS)}
        return self.router.entries[resident].frontend.backend.release_weights()

    def status(self):
        return self.router.current().backend.status()


class ModelRouter:
    """The server's app in JIT mode. It answers for the server what one
    model's frontend answers for it, and select() gives the frontend of the
    model a request names."""

    def __init__(
        self, entries, residency, *, latencies, response_store, request_timeout
    ):
        # In configured order: the first is the default model.
        self.entries = {entry.model: entry for entry in entries}
        self.residency = residency
        self.latencies = latencies
        self.response_store = response_store
        self.request_timeout = request_timeout
        self.backend = RouterBackend(self)
        self.default = next(iter(self.entries.values()))
        self.model = self.default.model
        self.model_names = tuple(self.entries)

    def select(self, name):
        """The entry of the model a request names; for none, the resident
        model, else the default. None for a name no model has."""
        if name in (None, ""):
            resident = self.residency.resident
            return self.entries.get(resident, self.default)
        if isinstance(name, str):
            return self.entries.get(name)
        return None

    def current(self):
        """The frontend that answers for the server: the resident model's,
        else the default's."""
        return self.entries.get(self.residency.resident, self.default).frontend

    def request_deadline(self, body, started_at):
        return self.current().request_deadline(body, started_at)

    @property
    def thinking_codec(self):
        return self.default.frontend.thinking_codec

    def models(self):
        """/v1/models' entries, in configured order."""
        resident = self.residency.resident
        return [
            {
                "id": entry.model,
                "object": "model",
                "created": 0,
                "owned_by": "splash",
                "max_model_len": entry.readiness.max_context_tokens,
                "context_length": entry.readiness.max_context_tokens,
                "vision": entry.readiness.vision,
                "input_modalities": entry.frontend.input_modalities,
                "loaded": entry.model == resident,
                "description": entry.description,
            }
            for entry in self.entries.values()
        ]

    def status(self):
        status = self.current().status()
        status["jit"] = {
            **self.residency.status(),
            "models": [
                {
                    "id": entry.model,
                    "maximum_context_tokens": entry.readiness.max_context_tokens,
                    "vision": entry.readiness.vision,
                }
                for entry in self.entries.values()
            ],
        }
        return status
