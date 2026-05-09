"""Multi-process supervisor: live together, die together."""

import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ManagedProcess:
    """A named, already-spawned subprocess managed by the Supervisor."""

    name: str
    proc: subprocess.Popen[bytes]


class Supervisor:
    """Runs N processes. Returns when any child dies or a signal triggers
    shutdown. On exit, terminates remaining children in declared order with
    a grace period before SIGKILL."""

    def __init__(
        self,
        processes: Sequence[ManagedProcess],
        grace_seconds: float = 25.0,
        poll_interval: float = 1.0,
    ) -> None:
        self._procs = list(processes)
        self._grace = grace_seconds
        self._poll = poll_interval
        self._shutdown = threading.Event()

    def run(self) -> int:
        """Block until a child exits or shutdown is signalled. Returns first exit code seen."""
        self._install_signal_handlers()

        rc = 0
        while not self._shutdown.is_set():
            for mp in self._procs:
                ret = mp.proc.poll()
                if ret is not None:
                    rc = ret
                    self._shutdown.set()
                    break
            if not self._shutdown.is_set():
                time.sleep(self._poll)

        self._terminate_all()
        return rc

    def shutdown(self) -> None:
        """Cooperative shutdown trigger: behaves like a SIGTERM to the supervisor."""
        self._shutdown.set()

    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
        except (ValueError, OSError):
            # Signal install fails when Supervisor.run() is invoked from a
            # non-main thread (e.g. in tests). The shutdown() method still
            # works as the cooperative path.
            pass

    def _on_signal(self, _signum: int, _frame: object) -> None:
        self._shutdown.set()

    def _terminate_all(self) -> None:
        # SIGTERM fires for every child up front, then we wait() on each in
        # declared order against a single shared deadline. Ordering matters:
        # processes are passed in front-to-back (haproxy, middleware, backend),
        # haproxy exits in milliseconds, and waiting on it first lets in-flight
        # requests drain through the middleware before the backend dies. Do
        # not split this into "terminate one, wait one" or restructure the
        # loops; the simultaneous SIGTERM is what bounds total shutdown time
        # to grace_seconds.
        for mp in self._procs:
            if mp.proc.poll() is None:
                mp.proc.terminate()
        deadline = time.monotonic() + self._grace
        for mp in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                mp.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                mp.proc.kill()
