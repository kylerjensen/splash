import math
import threading
import time
import unittest

from server import residency


class ManualTimer:
    """threading.Timer's stand-in, which fires only when the test says."""

    def __init__(self, interval, function, args):
        self.interval = interval
        self.function = function
        self.args = args
        self.cancelled = False
        self.daemon = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.function(*self.args)


class Engines:
    """Records loads and unloads; a model in `failing` fails to load, and a
    load of a model in `gates` waits for its event."""

    def __init__(self):
        self.events = []
        self.failing = set()
        self.gates = {}
        self.lock = threading.Lock()

    def load(self, model):
        if model in self.gates:
            self.gates[model].wait(5)
        with self.lock:
            self.events.append(("load", model))
        if model in self.failing:
            raise RuntimeError(f"{model} does not fit")
        return f"engine-{model}"

    def unload(self, model, engine):
        self.assertEqual(engine, f"engine-{model}")
        with self.lock:
            self.events.append(("unload", model))

    @staticmethod
    def assertEqual(first, second):
        if first != second:
            raise AssertionError(f"{first!r} != {second!r}")


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.002)


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.engines = Engines()
        self.timers = []

        def timer(interval, function, args):
            created = ManualTimer(interval, function, args)
            self.timers.append(created)
            return created

        self.timer_factory = timer

    def residency(self, ttl=300.0):
        models = residency.Residency(
            self.engines.load,
            self.engines.unload,
            ttl,
            poll=0.01,
            timer_factory=self.timer_factory,
        )
        self.addCleanup(models.close)
        return models

    def lease_in_thread(self, models, model, results, **options):
        def run():
            try:
                results.append(models.lease(model, **options))
            except BaseException as error:
                results.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def test_a_lease_loads_its_model_and_holds_it(self):
        models = self.residency()
        lease = models.lease("a")
        self.assertEqual(lease.engine, "engine-a")
        self.assertEqual(models.engine("a"), "engine-a")
        self.assertIsNone(models.engine("b"))
        self.assertEqual(models.status()["state"], "serving")
        # A second request for the resident model loads nothing.
        models.lease("a").release()
        lease.release()
        self.assertEqual(self.engines.events, [("load", "a")])
        self.assertEqual(models.status()["state"], "warm")

    def test_the_ttl_unloads_an_idle_model_and_a_lease_cancels_it(self):
        models = self.residency(ttl=300.0)
        models.lease("a").release()
        (first,) = self.timers
        self.assertEqual(first.interval, 300.0)
        lease = models.lease("a")
        self.assertTrue(first.cancelled)
        first.fire()
        self.assertEqual(models.resident, "a")
        lease.release()
        second = self.timers[-1]
        second.fire()
        self.assertIsNone(models.resident)
        self.assertEqual(self.engines.events, [("load", "a"), ("unload", "a")])
        self.assertEqual(models.status()["state"], "idle")
        # A request after the unload loads the model again.
        models.lease("a").release()
        self.assertEqual(self.engines.events[-1], ("load", "a"))

    def test_a_timer_of_an_earlier_residency_does_nothing(self):
        models = self.residency()
        models.lease("a").release()
        stale = self.timers[-1]
        models.lease("b").release()
        stale.cancelled = False
        stale.fire()
        self.assertEqual(models.resident, "b")

    def test_ttl_off_keeps_the_model_until_another_is_requested(self):
        models = self.residency(ttl=math.inf)
        models.lease("a").release()
        self.assertEqual(self.timers, [])
        self.assertIsNone(models.status()["ttl_seconds"])
        models.lease("b").release()
        self.assertEqual(
            self.engines.events, [("load", "a"), ("unload", "a"), ("load", "b")]
        )

    def test_another_model_waits_for_the_resident_ones_requests(self):
        models = self.residency()
        running = models.lease("a")
        results = []
        thread = self.lease_in_thread(models, "b", results)
        wait_until(lambda: models.status()["state"] == "draining")
        # The draining model takes no new request meanwhile.
        later = []
        self.lease_in_thread(models, "a", later)
        wait_until(lambda: models.status()["waiting"] == {"b": 1, "a": 1})
        self.assertEqual(results, [])
        self.assertEqual(later, [])
        self.assertEqual(self.engines.events, [("load", "a")])
        running.release()
        thread.join(2)
        (lease,) = results
        self.assertEqual(lease.engine, "engine-b")
        self.assertEqual(
            self.engines.events, [("load", "a"), ("unload", "a"), ("load", "b")]
        )
        # The request for a, queued behind b, gets its turn after b's.
        self.assertEqual(later, [])
        lease.release()
        wait_until(lambda: len(later) == 1)
        self.assertEqual(later[0].engine, "engine-a")
        later[0].release()

    def test_waiting_requests_for_one_model_share_its_load(self):
        models = self.residency()
        gate = self.engines.gates["a"] = threading.Event()
        results = []
        threads = [self.lease_in_thread(models, "a", results) for _ in range(3)]
        wait_until(lambda: models.status()["state"] == "loading")
        gate.set()
        for thread in threads:
            thread.join(2)
        self.assertEqual([lease.engine for lease in results], ["engine-a"] * 3)
        self.assertEqual(self.engines.events, [("load", "a")])
        self.assertEqual(models.status()["leases"], 3)
        for lease in results:
            lease.release()

    def test_a_failed_load_reaches_every_waiter_and_the_next_request_retries(self):
        models = self.residency()
        self.engines.failing.add("a")
        gate = self.engines.gates["a"] = threading.Event()
        results = []
        threads = [self.lease_in_thread(models, "a", results) for _ in range(2)]
        wait_until(lambda: models.status()["waiting"] == {"a": 2})
        gate.set()
        for thread in threads:
            thread.join(2)
        self.assertEqual(len(results), 2)
        for error in results:
            self.assertIsInstance(error, residency.ModelLoadFailed)
            self.assertIn("a does not fit", str(error))
        self.assertIsNone(models.resident)
        self.engines.failing.clear()
        models.lease("a").release()
        self.assertEqual(models.resident, "a")

    def test_a_waiting_request_ends_at_its_deadline_or_disconnect(self):
        models = self.residency()
        running = models.lease("a")
        with self.assertRaises(residency.LeaseTimeout):
            models.lease("b", deadline=time.monotonic() + 0.05)
        with self.assertRaises(ConnectionResetError):
            models.lease("b", disconnected=lambda: True)
        # Neither left a demand behind: a keeps serving, then idles.
        self.assertEqual(models.status()["waiting"], {})
        models.lease("a").release()
        running.release()
        self.assertEqual(models.resident, "a")
        self.assertEqual(models.status()["state"], "warm")

    def test_adopt_makes_a_loaded_model_resident(self):
        models = self.residency()
        models.adopt("a", "engine-a")
        self.assertEqual(models.engine("a"), "engine-a")
        self.assertEqual(len(self.timers), 1)
        models.lease("a").release()
        self.assertEqual(self.engines.events, [])
        with self.assertRaises(RuntimeError):
            models.adopt("b", "engine-b")

    def test_close_unloads_and_refuses_new_leases(self):
        models = self.residency()
        models.lease("a").release()
        models.close()
        self.assertEqual(self.engines.events, [("load", "a"), ("unload", "a")])
        with self.assertRaises(residency.ResidencyClosed):
            models.lease("a")

    def test_ttl_must_be_positive(self):
        with self.assertRaises(ValueError):
            residency.Residency(self.engines.load, self.engines.unload, 0)


if __name__ == "__main__":
    unittest.main()
