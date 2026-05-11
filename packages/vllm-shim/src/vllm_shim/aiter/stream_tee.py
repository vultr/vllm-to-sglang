"""Background-thread tee from a binary stream to a sink + per-line callback.

The shim runs SGLang as a child process and inherits its stderr by
default so ``kubectl logs`` sees everything. To capture AITER shape
lines we need to *also* peek at the stream, without burying the
backend's normal output.

This tee reads the source pipe a line at a time, forwards each line's
raw bytes to a sink (typically ``sys.stderr.buffer``) so the operator
sees them in real time, and calls ``callback(line_str)`` for code that
wants to act on the content. Callback errors are swallowed: a broken
shape-capture step must not corrupt the backend's log stream.
"""

import contextlib
import threading
from collections.abc import Callable
from typing import IO


class StreamTee:
    """Daemon thread that copies bytes from ``source`` to ``sink``.

    For each line read, the decoded line (UTF-8, errors replaced) is
    passed to ``callback``. The thread exits when ``source`` reaches
    EOF (typically when the producing subprocess exits and closes its
    end of the pipe).
    """

    def __init__(
        self,
        source: IO[bytes],
        sink: IO[bytes],
        callback: Callable[[str], None],
    ) -> None:
        self._source = source
        self._sink = sink
        self._callback = callback
        # Daemon so a stalled producer can't keep the supervisor alive
        # past its own shutdown deadline.
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        for raw in self._source:
            self._sink.write(raw)
            self._sink.flush()
            line = raw.decode("utf-8", errors="replace")
            # Tee survival is more important than any single shape-capture
            # write. A broken callback (disk full, readonly volume, etc.)
            # must not silence the backend's stderr stream.
            with contextlib.suppress(Exception):
                self._callback(line)
