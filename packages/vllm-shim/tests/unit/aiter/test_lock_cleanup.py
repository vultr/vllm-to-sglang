"""Tests for orphaned AITER JIT lock cleanup at entrypoint startup."""

import os
from pathlib import Path

import pytest
from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, REASON_NO_SHIM_HOME
from vllm_shim.aiter.lock_cleanup import (
    CleanupPlan,
    _is_lock_file,
    cleanup_locks,
    plan_cleanup,
)
from vllm_shim.cli.rocm_probe import GpuAgent

_GPU = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")


# ---------- plan_cleanup ----------


def test_plan_disabled_when_no_gpu(tmp_path: Path) -> None:
    plan = plan_cleanup(shim_home=tmp_path, gpu=None)
    assert plan.enabled is False
    assert plan.root is None
    assert plan.reason == REASON_NO_GPU


def test_plan_disabled_when_shim_home_unresolvable() -> None:
    plan = plan_cleanup(shim_home=None, gpu=_GPU)
    assert plan.enabled is False
    assert plan.root is None
    assert plan.reason == REASON_NO_SHIM_HOME


def test_plan_uses_default_jit_path_when_no_operator_override(tmp_path: Path) -> None:
    plan = plan_cleanup(shim_home=tmp_path, gpu=_GPU, env={})
    assert plan.enabled is True
    assert plan.reason == REASON_ENABLED
    # On a host without the image-baked cache-key file the resolver
    # collapses to "default"; the cleanup root must follow.
    assert plan.root == tmp_path / "aiter" / "jit" / "default"


def test_plan_honors_operator_aiter_jit_dir_override(tmp_path: Path) -> None:
    # An operator who set AITER_JIT_DIR explicitly is overriding the
    # default JIT location; cleanup must scan their path or it would
    # leave their stale locks behind and clean an empty directory.
    elsewhere = tmp_path / "elsewhere"
    plan = plan_cleanup(
        shim_home=tmp_path, gpu=_GPU, env={"AITER_JIT_DIR": str(elsewhere)}
    )
    assert plan.enabled is True
    assert plan.root == elsewhere


def test_plan_empty_operator_value_falls_through_to_default(tmp_path: Path) -> None:
    # An empty AITER_JIT_DIR (truthiness-false) should be treated as
    # unset, same way docker -e VAR= or k8s value: "" gets exported.
    plan = plan_cleanup(shim_home=tmp_path, gpu=_GPU, env={"AITER_JIT_DIR": ""})
    assert plan.root == tmp_path / "aiter" / "jit" / "default"


def test_plan_no_env_arg_means_default_path(tmp_path: Path) -> None:
    # Callers that don't pass env should still get a default path,
    # not crash on a None.get() somewhere.
    plan = plan_cleanup(shim_home=tmp_path, gpu=_GPU)
    assert plan.enabled is True
    assert plan.root is not None


# ---------- _is_lock_file ----------


@pytest.mark.parametrize(
    "name",
    [
        "lock",                              # csrc/cpp_itfs/utils.py:160
        "lock_module_custom_all_reduce",     # the one in the user's wedged pod
        "lock_3rdparty_clone_composable_kernel",  # aiter/jit/core.py:716
        "lock_module_fused_moe",             # aiter/jit/core.py:744
        "mymodule.lock",                     # cpp_itfs func_name.lock pattern
        "compile.lock",                      # core.py:309 new_file_path.lock
    ],
)
def test_is_lock_file_matches_aiter_patterns(name: str) -> None:
    assert _is_lock_file(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "locksmith.txt",     # contains "lock" but not at start/end with the right delimiters
        "padlock.txt",       # contains "lock" mid-string
        "blockwise.py",      # superstring of "lock"
        "module.so",         # genuine build artifact, must not be swept
        "lockfile.cfg",      # no underscore after "lock"; user-named, leave alone
        "the.lock.bak",      # not "lock" and not ".lock" tail
        "",                  # degenerate case
    ],
)
def test_is_lock_file_rejects_non_lock_names(name: str) -> None:
    assert _is_lock_file(name) is False


# ---------- cleanup_locks ----------


def _enabled_plan(root: Path) -> CleanupPlan:
    return CleanupPlan(enabled=True, root=root, reason=REASON_ENABLED)


def test_cleanup_returns_empty_when_disabled(tmp_path: Path) -> None:
    plan = CleanupPlan(enabled=False, root=tmp_path, reason=REASON_NO_GPU)
    assert cleanup_locks(plan) == []


def test_cleanup_returns_empty_when_root_missing(tmp_path: Path) -> None:
    # First-ever pod on a fresh PV: the JIT dir doesn't exist yet
    # because AITER hasn't built anything. Cleanup must no-op cleanly.
    plan = _enabled_plan(tmp_path / "never-built")
    assert cleanup_locks(plan) == []


def test_cleanup_removes_lock_file_with_exact_name(tmp_path: Path) -> None:
    (tmp_path / "build").mkdir()
    lock = tmp_path / "build" / "lock"
    lock.write_text("")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert cleaned == [lock]
    assert not lock.exists()


def test_cleanup_removes_lock_module_prefix(tmp_path: Path) -> None:
    # The exact pattern that wedged the user's MXFP4 pod.
    (tmp_path / "build").mkdir()
    lock = tmp_path / "build" / "lock_module_custom_all_reduce"
    lock.write_text("")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert cleaned == [lock]
    assert not lock.exists()


def test_cleanup_removes_dot_lock_suffix(tmp_path: Path) -> None:
    (tmp_path / "build").mkdir()
    lock = tmp_path / "build" / "fmoe_compile.lock"
    lock.write_text("")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert cleaned == [lock]


def test_cleanup_recurses_into_subdirectories(tmp_path: Path) -> None:
    # AITER scatters locks across nested build dirs (per-module, per
    # 3rd-party, per cpp_itfs func). rglob must walk the whole tree.
    nested = tmp_path / "build" / "deep" / "deeper"
    nested.mkdir(parents=True)
    top = tmp_path / "lock"
    mid = tmp_path / "build" / "lock_module_a"
    bottom = nested / "b.lock"
    for f in (top, mid, bottom):
        f.write_text("")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert set(cleaned) == {top, mid, bottom}
    for f in (top, mid, bottom):
        assert not f.exists()


def test_cleanup_does_not_touch_non_lock_files(tmp_path: Path) -> None:
    # Compiled .so files, source files, and user-named files
    # containing "lock" mid-name must survive. Removing a .so would
    # force AITER to recompile on every pod restart, defeating the PV.
    (tmp_path / "build").mkdir()
    keepers = [
        tmp_path / "build" / "module.so",
        tmp_path / "build" / "kernel.hsaco",
        tmp_path / "build" / "padlock.txt",
        tmp_path / "build" / "locksmith.py",
        tmp_path / "build" / "blockwise.py",
    ]
    lock = tmp_path / "build" / "lock"
    for f in [*keepers, lock]:
        f.write_text("contents")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert cleaned == [lock]
    for f in keepers:
        assert f.exists()
        assert f.read_text() == "contents"


def test_cleanup_skips_directories_named_like_locks(tmp_path: Path) -> None:
    # A directory that happens to match the name patterns must not be
    # removed (rmdir vs unlink semantics differ; we only unlink files).
    dir_named_lock = tmp_path / "lock_module_oops"
    dir_named_lock.mkdir()
    real_lock = tmp_path / "lock"
    real_lock.write_text("")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert cleaned == [real_lock]
    assert dir_named_lock.is_dir()


def test_cleanup_is_best_effort_on_individual_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If one lock file is un-removable (permission, race), the others
    # in the same scan should still be cleaned. Otherwise a single
    # un-removable lock anywhere on the PV would block pod startup.
    (tmp_path / "build").mkdir()
    bad = tmp_path / "build" / "lock_module_unremovable"
    good = tmp_path / "build" / "lock_module_clean"
    bad.write_text("")
    good.write_text("")

    real_unlink = Path.unlink

    def selective_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == bad:
            raise PermissionError("simulated")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", selective_unlink)
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert cleaned == [good]
    assert not good.exists()
    assert bad.exists()  # the un-removable one stays


def test_cleanup_returned_paths_are_absolute(tmp_path: Path) -> None:
    # The returned list is what callers log and surface; relative
    # paths would be ambiguous across the entrypoint's cwd changes.
    (tmp_path / "build").mkdir()
    lock = tmp_path / "build" / "lock"
    lock.write_text("")
    cleaned = cleanup_locks(_enabled_plan(tmp_path))
    assert all(os.path.isabs(p) for p in cleaned)
