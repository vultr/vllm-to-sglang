"""Tests for the HIP_ONLINE_TUNING symlink anchor."""

import os
import sys
from pathlib import Path

import pytest
from vllm_shim.aiter.hip_online_tuning import (
    FILENAME,
    REASON_ALREADY_LINKED,
    REASON_BLOCKED,
    REASON_DISABLED,
    REASON_NO_GPU,
    REASON_NO_SHIM_HOME,
    REASON_WILL_LINK,
    apply_hip_online_tuning,
    plan_hip_online_tuning,
)
from vllm_shim.cli.rocm_probe import GpuAgent

_MI300X = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")

# Symlink creation on Windows needs admin or developer mode. The shim
# only runs on Linux in production, so we skip the side-effect tests
# rather than wire up an OS-specific fixture.
_skip_no_symlink = pytest.mark.skipif(
    sys.platform == "win32",
    reason="symlinks need admin / developer mode on Windows",
)


def test_plan_disabled_when_env_var_unset(tmp_path: Path) -> None:
    plan = plan_hip_online_tuning({}, _MI300X, tmp_path, tmp_path)
    assert plan.enabled is False
    assert plan.reason == REASON_DISABLED
    assert plan.storage is None and plan.target is None


def test_plan_disabled_when_env_var_is_zero(tmp_path: Path) -> None:
    # AITER's C++ side accepts only "1" or "true"; mirror that exactly
    # so the shim doesn't anchor a file AITER won't read.
    plan = plan_hip_online_tuning(
        {"HIP_ONLINE_TUNING": "0"}, _MI300X, tmp_path, tmp_path
    )
    assert plan.enabled is False
    assert plan.reason == REASON_DISABLED


def test_plan_accepts_true_string(tmp_path: Path) -> None:
    plan = plan_hip_online_tuning(
        {"HIP_ONLINE_TUNING": "true"}, _MI300X, tmp_path, tmp_path / "cwd"
    )
    assert plan.enabled is True


def test_plan_disabled_when_no_gpu(tmp_path: Path) -> None:
    # CUDA host or dev box: the env var has no meaning since AITER's
    # gradlib isn't loaded. Anchoring would create a useless symlink.
    plan = plan_hip_online_tuning(
        {"HIP_ONLINE_TUNING": "1"}, None, tmp_path, tmp_path
    )
    assert plan.enabled is False
    assert plan.reason == REASON_NO_GPU


def test_plan_disabled_when_no_shim_home(tmp_path: Path) -> None:
    # Operator opted in but the shim couldn't resolve a PV. The plan
    # surfaces the reason so the launch info explains why anchoring
    # didn't happen; apply() will no-op.
    plan = plan_hip_online_tuning(
        {"HIP_ONLINE_TUNING": "1"}, _MI300X, None, tmp_path
    )
    assert plan.enabled is False
    assert plan.reason == REASON_NO_SHIM_HOME


def test_plan_will_link_when_target_missing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    assert plan.enabled is True
    assert plan.reason == REASON_WILL_LINK
    assert plan.storage == home / FILENAME
    assert plan.target == cwd / FILENAME


def test_plan_blocks_on_existing_regular_file(tmp_path: Path) -> None:
    # A real file at the target is almost always operator data from a
    # pre-shim run. Refusing to clobber it (and explaining why in the
    # info dump) lets the operator decide whether to move it onto the PV.
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / FILENAME).write_text("M,N,K,algo\n")
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    assert plan.enabled is True
    assert plan.reason == REASON_BLOCKED


@_skip_no_symlink
def test_plan_already_linked_when_symlink_correct(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    storage = home / FILENAME
    storage.touch()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / FILENAME).symlink_to(storage)
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    assert plan.enabled is True
    assert plan.reason == REASON_ALREADY_LINKED


@_skip_no_symlink
def test_plan_will_link_when_symlink_points_elsewhere(tmp_path: Path) -> None:
    # An incorrect existing symlink (image upgrade moved the storage
    # path, operator hand-linked somewhere else) gets replaced rather
    # than left dangling.
    home = tmp_path / "home"
    home.mkdir()
    other = tmp_path / "elsewhere.csv"
    other.touch()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / FILENAME).symlink_to(other)
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    assert plan.reason == REASON_WILL_LINK


def test_apply_noop_when_disabled(tmp_path: Path) -> None:
    plan = plan_hip_online_tuning({}, _MI300X, tmp_path, tmp_path)
    apply_hip_online_tuning(plan)
    # No storage file, no symlink: pure no-op.
    assert not (tmp_path / FILENAME).exists()


def test_apply_noop_when_blocked(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / FILENAME).write_text("M,N,K,algo\n")
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    apply_hip_online_tuning(plan)
    # Storage must not be touched and the existing file must stay
    # exactly as the operator left it.
    assert not (home / FILENAME).exists()
    assert (cwd / FILENAME).read_text() == "M,N,K,algo\n"


@_skip_no_symlink
def test_apply_creates_symlink_and_storage_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    apply_hip_online_tuning(plan)
    assert (home / FILENAME).is_file()
    assert (cwd / FILENAME).is_symlink()
    assert os.readlink(cwd / FILENAME) == str(home / FILENAME)


@_skip_no_symlink
def test_apply_is_idempotent_on_already_linked(tmp_path: Path) -> None:
    # Restart case: the previous pod already created the symlink, so a
    # second apply must not perturb it (and must not raise on the
    # existing symlink). Storage gets a touch but its mtime not the
    # contents - tuning data accumulated by the previous pod survives.
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    apply_hip_online_tuning(
        plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    )
    (home / FILENAME).write_text("M,N,K,algo\nrow1\n")
    apply_hip_online_tuning(
        plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    )
    assert (cwd / FILENAME).is_symlink()
    assert (home / FILENAME).read_text() == "M,N,K,algo\nrow1\n"


@_skip_no_symlink
def test_apply_replaces_wrong_pointing_symlink(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    other = tmp_path / "elsewhere.csv"
    other.write_text("stale\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / FILENAME).symlink_to(other)
    plan = plan_hip_online_tuning({"HIP_ONLINE_TUNING": "1"}, _MI300X, home, cwd)
    apply_hip_online_tuning(plan)
    assert (cwd / FILENAME).is_symlink()
    assert os.readlink(cwd / FILENAME) == str(home / FILENAME)
    # The original elsewhere file must remain - we replaced the link,
    # not the file it pointed at.
    assert other.read_text() == "stale\n"
