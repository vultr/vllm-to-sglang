"""Tests for the Parallelism value object."""

from vllm_shim.values.parallelism import Parallelism


def test_defaults_are_all_one() -> None:
    p = Parallelism()
    assert (p.tp, p.ep, p.pp) == (1, 1, 1)


def test_path_segment_tp_only() -> None:
    assert Parallelism(tp=8).path_segment() == "tp8"


def test_path_segment_single_gpu_is_not_empty() -> None:
    # Even the trivial 1x1x1 topology must produce a non-empty segment so
    # the captured-shape directory layout is well-defined for single-GPU
    # deployments.
    assert Parallelism().path_segment() == "tp1"


def test_path_segment_tp_ep() -> None:
    assert Parallelism(tp=8, ep=8).path_segment() == "tp8-ep8"


def test_path_segment_tp_pp() -> None:
    assert Parallelism(tp=4, pp=2).path_segment() == "tp4-pp2"


def test_path_segment_all_three() -> None:
    assert Parallelism(tp=4, ep=4, pp=2).path_segment() == "tp4-ep4-pp2"


def test_path_segment_omits_ep_pp_when_one() -> None:
    # EP=1 / PP=1 are the off states; they must not clutter the path.
    assert Parallelism(tp=2, ep=1, pp=1).path_segment() == "tp2"


def test_is_hashable_and_frozen() -> None:
    # Used as a dict key (e.g., per-topology shape stores).
    a = Parallelism(tp=8)
    b = Parallelism(tp=8)
    assert hash(a) == hash(b)
    assert {a: 1}[b] == 1
