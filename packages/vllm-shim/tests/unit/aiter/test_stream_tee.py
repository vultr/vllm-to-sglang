"""Tests for the background-thread stream tee."""

import io
import os
import threading

from vllm_shim.aiter.stream_tee import StreamTee


def _tee_and_collect(payload: bytes) -> tuple[bytes, list[str]]:
    source = io.BytesIO(payload)
    sink = io.BytesIO()
    received: list[str] = []
    tee = StreamTee(source=source, sink=sink, callback=received.append)
    tee.start()
    tee.join(timeout=2.0)
    assert not tee.is_alive(), "tee thread should have exited at EOF"
    return sink.getvalue(), received


def test_forwards_every_byte_to_sink() -> None:
    payload = b"line one\nline two\nline three\n"
    sink_bytes, _ = _tee_and_collect(payload)
    assert sink_bytes == payload


def test_callback_invoked_once_per_line() -> None:
    payload = b"alpha\nbeta\ngamma\n"
    _, received = _tee_and_collect(payload)
    assert received == ["alpha\n", "beta\n", "gamma\n"]


def test_handles_invalid_utf8_via_replacement() -> None:
    # \xff is not valid UTF-8; the tee should pass it through to the
    # sink as-is and replace it in the callback string.
    payload = b"bad \xff byte\n"
    sink_bytes, received = _tee_and_collect(payload)
    assert sink_bytes == payload
    assert received == ["bad � byte\n"]


def test_callback_exception_does_not_kill_tee() -> None:
    payload = b"good\nthen-bad\nstill-good\n"
    source = io.BytesIO(payload)
    sink = io.BytesIO()
    seen: list[str] = []

    def callback(line: str) -> None:
        seen.append(line)
        if "bad" in line:
            raise RuntimeError("simulated capture failure")

    tee = StreamTee(source=source, sink=sink, callback=callback)
    tee.start()
    tee.join(timeout=2.0)
    assert not tee.is_alive()
    # All three lines reached the sink AND the callback, even though
    # the middle callback raised.
    assert sink.getvalue() == payload
    assert seen == ["good\n", "then-bad\n", "still-good\n"]


def test_thread_is_daemon() -> None:
    # The supervisor relies on this so a stalled producer can't keep
    # the process alive past its shutdown deadline.
    source = io.BytesIO(b"")
    sink = io.BytesIO()
    tee = StreamTee(source=source, sink=sink, callback=lambda _: None)
    # Reach into the private thread to assert daemon-ness; this is
    # the load-bearing invariant for shutdown, worth checking.
    assert tee._thread.daemon is True


def test_streams_realtime_not_in_one_batch() -> None:
    # If the tee accumulated input before invoking the callback, this
    # test would hang on the first wait(). A real os.pipe means the
    # reader thread genuinely blocks on input.
    r, w = os.pipe()
    source = os.fdopen(r, "rb", buffering=0)
    writer = os.fdopen(w, "wb", buffering=0)
    sink = io.BytesIO()
    got = threading.Event()
    received: list[str] = []

    def callback(line: str) -> None:
        received.append(line)
        got.set()

    tee = StreamTee(source=source, sink=sink, callback=callback)
    tee.start()
    writer.write(b"first\n")
    assert got.wait(timeout=2.0), "callback should fire before EOF"
    assert received == ["first\n"]
    writer.close()
    tee.join(timeout=2.0)
    source.close()
