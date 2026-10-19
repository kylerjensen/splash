"""Which model JIT mode keeps loaded, when it loads and unloads one.

One model is resident at a time. A request leases the model it names: the
lease waits until that model is resident, loading it if none is, and holds
it until the request ends. A request for another model drains the resident
one: it takes no new leases, its leased requests finish, then it unloads and
the requested model loads. Models take their turns in the order their first
waiting request arrived, so neither of two busy models starves the other. A
resident model no request leases unloads once it has been idle for the TTL.

Standard library only: what loads and unloads a model is the caller's.
"""

import math
import threading
import time
from collections.abc import Callable

# How often a waiting lease checks its deadline and its client.
DEFAULT_POLL_SECONDS = 0.25


class ResidencyClosed(RuntimeError):
    """The server is shutting down; no model loads any more."""


class LeaseTimeout(TimeoutError):
    """The request's deadline passed while its model was not yet resident."""


class ModelLoadFailed(RuntimeError):
    """The model a request waited for failed to load; the next lease of it
    tries again."""

    def __init__(self, model, error):
        super().__init__(f"model {model} failed to load: {error}")
        self.model = model
        self.error = error


class _Waiter:
    __slots__ = ("model", "error")

    def __init__(self, model):
        self.model = model
        # A failed load of the model raises to every request waiting for it.
        self.error = None


class Lease:
    """One request's hold on its resident model, which release() ends."""

    def __init__(self, residency, model, engine):
        self._residency = residency
        self._released = False
        self.model = model
        self.engine = engine

    def release(self):
        if not self._released:
            self._released = True
            self._residency._release(self.model)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()


class Residency:
    def __init__(
        self,
        load: Callable[[str], object],
        unload: Callable[[str, object], None],
        ttl: float,
        *,
        poll: float = DEFAULT_POLL_SECONDS,
        timer_factory: Callable[..., threading.Timer] = threading.Timer,
        clock: Callable[[], float] = time.monotonic,
    ):
        """`load(model)` starts a model's engine and returns it once it can
        serve; `unload(model, engine)` stops it and returns once its memory
        is free. Both block, and run without the lock on a leasing thread or
        the TTL's timer. A `ttl` of math.inf keeps a model resident until
        another is requested."""
        if not ttl > 0:
            raise ValueError("ttl must be positive")
        self._load = load
        self._unload = unload
        self._ttl = ttl
        self._poll = poll
        self._timer_factory = timer_factory
        self._clock = clock
        self._condition = threading.Condition()
        self._resident = None
        self._engine = None
        self._leases = 0
        # "loading" or "unloading" while a thread changes the resident model.
        self._transition = None
        self._loading = None
        # The models requests wait for, in the order each model's first
        # waiting request arrived, and their waiting requests.
        self._demand = []
        self._waiters = {}
        self._timer = None
        # Counts residencies, so a timer of an earlier one does nothing.
        self._generation = 0
        self._closed = False

    # -- leases -----------------------------------------------------------

    def lease(
        self,
        model: str,
        deadline: float = math.inf,
        disconnected: Callable[[], bool] = lambda: False,
    ) -> Lease:
        """Wait until `model` is resident and hold it. Raises LeaseTimeout
        once `deadline` (on the clock) passes, ConnectionResetError once
        `disconnected()`, ModelLoadFailed and ResidencyClosed."""
        waiter = _Waiter(model)
        with self._condition:
            self._add_waiter(waiter)
            try:
                while True:
                    if self._closed:
                        raise ResidencyClosed("server is shutting down")
                    if waiter.error is not None:
                        raise waiter.error
                    if self._admits(model):
                        self._leases += 1
                        self._cancel_timer()
                        return Lease(self, model, self._engine)
                    if self._transition is None and self._demand[0] == model:
                        if self._resident is None:
                            self._load_locked(model)
                            continue
                        if self._leases == 0:
                            self._unload_locked()
                            continue
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise LeaseTimeout(f"model {model} did not load in time")
                    self._condition.wait(min(remaining, self._poll))
                    if disconnected():
                        raise ConnectionResetError(
                            "client disconnected while its model loaded"
                        )
            finally:
                self._remove_waiter(waiter)

    def _admits(self, model):
        """Whether a request for `model` may lease it now: it is resident and
        no request waits for another model ahead of it."""
        return (
            self._resident == model
            and self._transition is None
            and self._demand[0] == model
        )

    def _add_waiter(self, waiter):
        waiting = self._waiters.setdefault(waiter.model, [])
        if not waiting:
            self._demand.append(waiter.model)
        waiting.append(waiter)

    def _remove_waiter(self, waiter):
        waiting = self._waiters[waiter.model]
        waiting.remove(waiter)
        if not waiting:
            del self._waiters[waiter.model]
            self._demand.remove(waiter.model)
            # The next model in turn may now drain the resident one, or the
            # resident one may now idle.
            if not self._demand and self._leases == 0:
                self._arm_timer()
        self._condition.notify_all()

    def _release(self, model):
        with self._condition:
            self._leases -= 1
            if self._leases == 0:
                if self._demand:
                    # A model waits for this one's last request to end.
                    self._condition.notify_all()
                else:
                    self._arm_timer()

    # -- transitions ------------------------------------------------------

    def _load_locked(self, model):
        """Load `model`, with the lock held on entry and exit but not
        during the load. A failure raises to every request then waiting."""
        self._transition = "loading"
        self._loading = model
        self._condition.release()
        engine = error = None
        try:
            engine = self._load(model)
        except BaseException as caught:
            # Every request waiting for the model learns of the failure.
            error = caught
        finally:
            self._condition.acquire()
            self._transition = None
            self._loading = None
        if error is not None:
            failure = ModelLoadFailed(model, error)
            for waiter in self._waiters.get(model, ()):
                waiter.error = failure
            self._condition.notify_all()
            if not isinstance(error, Exception):
                raise error
            return
        self._resident = model
        self._engine = engine
        self._generation += 1
        if self._closed:
            self._unload_locked()
        self._condition.notify_all()

    def _unload_locked(self):
        """Unload the resident model, with the lock held on entry and exit
        but not while it stops."""
        model, engine = self._resident, self._engine
        self._cancel_timer()
        self._transition = "unloading"
        self._condition.release()
        try:
            self._unload(model, engine)
        finally:
            self._condition.acquire()
            self._transition = None
            self._resident = None
            self._engine = None
            self._generation += 1
            self._condition.notify_all()

    def adopt(self, model, engine):
        """Make `model`, whose `engine` the caller loaded, the resident model
        without a load, as startup leaves its last probed model."""
        with self._condition:
            if self._resident is not None or self._transition is not None:
                raise RuntimeError("a model is already resident")
            self._resident = model
            self._engine = engine
            self._generation += 1
            if not self._demand and self._leases == 0:
                self._arm_timer()
            self._condition.notify_all()

    # -- the TTL ----------------------------------------------------------

    def _arm_timer(self):
        self._cancel_timer()
        if (
            self._closed
            or self._resident is None
            or self._transition is not None
            or math.isinf(self._ttl)
        ):
            return
        timer = self._timer_factory(self._ttl, self._expire, (self._generation,))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _expire(self, generation):
        with self._condition:
            if (
                generation != self._generation
                or self._resident is None
                or self._transition is not None
                or self._leases
                or self._demand
            ):
                return
            self._timer = None
            self._unload_locked()

    # -- the rest ---------------------------------------------------------

    def engine(self, model):
        """The engine of `model` while it is resident, else None."""
        with self._condition:
            if self._resident == model and self._transition is None:
                return self._engine
            return None

    @property
    def held_engine(self):
        """The engine loaded now, the one unloading too, read without the
        lock: safe in a signal handler."""
        return self._engine

    @property
    def resident(self):
        with self._condition:
            return self._resident

    def status(self):
        with self._condition:
            if self._transition is not None:
                state = self._transition
            elif self._resident is None:
                state = "idle"
            elif self._demand and self._demand[0] != self._resident:
                state = "draining"
            elif self._leases:
                state = "serving"
            else:
                state = "warm"
            return {
                "resident": self._resident,
                "loading": self._loading,
                "state": state,
                "leases": self._leases,
                "waiting": {model: len(self._waiters[model]) for model in self._demand},
                "ttl_seconds": None if math.isinf(self._ttl) else self._ttl,
            }

    def close(self):
        """Stop loading models and unload the resident one. A load in
        progress unloads its model once it ends."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._cancel_timer()
            self._condition.notify_all()
            if self._resident is not None and self._transition is None:
                self._unload_locked()
