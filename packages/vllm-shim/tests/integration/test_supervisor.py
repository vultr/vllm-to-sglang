"""Supervisor integration tests using real short-lived subprocesses."""

import subprocess
import sys
import time

from vllm_shim.cli.supervisor import ManagedProcess, Supervisor


def _spawn_sleeper(seconds: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def test_returns_when_first_child_dies() -> None:
    fast = _spawn_sleeper(0.2)
    slow1 = _spawn_sleeper(5)
    slow2 = _spawn_sleeper(5)
    sup = Supervisor(
        [
            ManagedProcess("fast", fast),
            ManagedProcess("slow1", slow1),
            ManagedProcess("slow2", slow2),
        ],
        grace_seconds=2.0,
        poll_interval=0.05,
    )
    start = time.monotonic()
    rc = sup.run()
    elapsed = time.monotonic() - start
    assert rc == 0
    assert elapsed < 4.0
    assert slow1.poll() is not None
    assert slow2.poll() is not None


def test_signal_triggers_shutdown_in_order() -> None:
    procs = [_spawn_sleeper(30) for _ in range(3)]
    sup = Supervisor(
        [
            ManagedProcess("haproxy", procs[0]),
            ManagedProcess("middleware", procs[1]),
            ManagedProcess("backend", procs[2]),
        ],
        grace_seconds=2.0,
        poll_interval=0.05,
    )

    import threading

    def fire() -> None:
        time.sleep(0.3)
        sup.shutdown()

    threading.Thread(target=fire, daemon=True).start()
    rc = sup.run()
    assert rc == 0
    for p in procs:
        assert p.poll() is not None
