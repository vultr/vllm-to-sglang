# Process supervisor

The shim runs three child processes (haproxy, middleware, backend) inside a single container. The `Supervisor` class (`packages/vllm-shim/src/vllm_shim/cli/supervisor.py`) is what keeps them alive together and tears them down together.

It's deliberately small (about 80 lines) because the supervision contract is narrow: if any child dies, all children die; if a signal arrives, all children die.

## Data model

```python
@dataclass(frozen=True)
class ManagedProcess:
    name: str
    proc: subprocess.Popen[bytes]

class Supervisor:
    def __init__(self, processes, grace_seconds=25.0, poll_interval=1.0): ...
    def run(self) -> int: ...
    def shutdown(self) -> None: ...
```

A `ManagedProcess` is just a name + an already-spawned `Popen`. The supervisor doesn't spawn anything itself; the entrypoint constructs the children, hands them in, and calls `run()`.

## Run loop

```python
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
```

Per tick (default 1 second):

- Poll each child. The first one with a return code wins; its exit code becomes the supervisor's return code, the shutdown event is set, and the loop exits.
- Otherwise sleep `poll_interval` and try again.

`shutdown()` is the cooperative path: anything that calls it triggers the same teardown without needing to mock a signal. Tests use it.

## Signal handling

`run()` installs SIGTERM and SIGINT handlers on entry. Both flip the shutdown event. The signal install is wrapped in `try/except (ValueError, OSError)` because Python only allows signal install from the main thread; tests that drive `run()` from a worker thread would otherwise crash here.

The signal handler does not call `_terminate_all` directly; it just sets the event. The run loop notices on the next poll iteration. This avoids reentrancy concerns inside the signal handler.

## Teardown ordering

This is the part that took the most thought. The relevant code is in `_terminate_all`, and there's a long comment in the source explaining why it's structured the way it is.

The shape:

```python
for mp in self._procs:
    if mp.proc.poll() is None:
        mp.proc.terminate()        # SIGTERM all of them up front
deadline = time.monotonic() + self._grace
for mp in self._procs:
    remaining = max(0.0, deadline - time.monotonic())
    try:
        mp.proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        mp.proc.kill()
```

Two things to notice:

### Simultaneous SIGTERMs, sequential waits

The naive structure is "terminate one, wait one, terminate next." That bounds total shutdown time to `n * grace_seconds`. The structure used here (terminate all, then wait all against a single deadline) bounds it to one `grace_seconds` regardless of how many children there are.

### Wait order matters even though SIGTERMs are simultaneous

The supervisor receives processes in declared order: `haproxy, middleware, backend`. Even though they all get SIGTERM at the same instant:

- haproxy exits in milliseconds (it's a stateless reverse proxy with fast shutdown).
- The middleware needs a few seconds to drain in-flight requests.
- The backend (SGLang or TRT-LLM) is slowest; it has to flush GPU buffers and tear down the inference engine.

By `wait()`ing on haproxy first, then the middleware, the supervisor's bookkeeping naturally serializes around the right order. By the time it's waiting on the backend, the middleware has likely already drained whatever requests were in flight. The backend dies last, by which point nothing should still be talking to it.

If you split this into "terminate one, wait one, terminate next," haproxy stays alive until the loop reaches it, and clients keep getting routed to a dying backend.

The source comment makes this explicit and warns against restructuring; it is load-bearing.

## Grace and kill

Each child gets up to (deadline - now) seconds to exit cleanly. If `wait()` times out, `kill()` sends SIGKILL. The default 25-second grace is set to comfortably exceed the backend's typical clean-shutdown time (5 to 15 seconds for SGLang, dominated by GPU teardown; TRT-LLM is in the same range).

If you bump it, also consider container-level shutdown grace (`terminationGracePeriodSeconds` in k8s, default 30). The supervisor's grace must be less than or equal to the container grace, or k8s will SIGKILL the supervisor itself before it can kill its children, leading to PID-1 zombies.

## Return code semantics

The supervisor's return code is the exit code of the *first* child to exit. That value propagates out of `entrypoint.main` and out of the `vllm` console script. Concretely:

- The backend exiting cleanly (e.g., on SIGTERM) → 0 → container exits 0.
- The backend crashing → its exit code → container exits non-zero → k8s notices.
- A signal-driven shutdown also returns 0, because the run loop never enters the "first child died" branch; it just runs `_terminate_all` and the initial `rc = 0` survives.

This is the right behavior for a k8s-style restart policy: clean shutdowns don't restart, crashes do.

## Tests

`packages/vllm-shim/tests/integration/test_supervisor.py` covers the two main shapes:

- `test_returns_when_first_child_dies` spawns three sleepers, the first short, and asserts the supervisor returns within the grace deadline and the others are reaped.
- `test_signal_triggers_shutdown_in_order` spawns three long sleepers and triggers `shutdown()` from a sidecar thread, asserting all three are reaped.

Real signal delivery isn't tested directly (it's tricky cross-platform); the cooperative `shutdown()` path exercises the same teardown code.
