"""Tests for AITER tuned-config restoration."""

from pathlib import Path

from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, REASON_NO_HF_HOME
from vllm_shim.aiter.restore import (
    AITER_CONFIG_DIR,
    plan_restore,
    restore_configs,
)
from vllm_shim.cli.rocm_probe import GpuAgent

_GPU = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")


def _enabled_plan(tmp_path: Path) -> tuple[Path, Path]:
    """Build a (source, target) pair under tmp_path."""
    return tmp_path / "src", tmp_path / "tgt"


# ---------- plan_restore ----------


def test_plan_disabled_when_no_gpu(tmp_path: Path) -> None:
    plan = plan_restore(hf_home=tmp_path, gpu=None, target=tmp_path / "tgt")
    assert plan.enabled is False
    assert plan.source is None
    assert plan.reason == REASON_NO_GPU


def test_plan_disabled_when_hf_home_unresolvable(tmp_path: Path) -> None:
    plan = plan_restore(hf_home=None, gpu=_GPU, target=tmp_path / "tgt")
    assert plan.enabled is False
    assert plan.source is None
    assert plan.reason == REASON_NO_HF_HOME


def test_plan_source_is_bucket_partitioned(tmp_path: Path) -> None:
    plan = plan_restore(hf_home=tmp_path, gpu=_GPU)
    assert plan.enabled is True
    assert plan.reason == REASON_ENABLED
    # Bucket only - tuned configs are shape-keyed and reusable across
    # models, so no model/parallelism in the path.
    assert plan.source == tmp_path / "vllm-shim" / "aiter-configs" / "gfx942-304cu"


def test_plan_target_defaults_to_aiter_config_dir() -> None:
    # The default target is the hardcoded read location AITER itself uses;
    # operators have no way to point AITER elsewhere, so we must not either.
    plan = plan_restore(hf_home=Path("/data/hf"), gpu=_GPU)
    assert plan.target == AITER_CONFIG_DIR


# ---------- restore_configs ----------


def test_restore_no_op_when_disabled(tmp_path: Path) -> None:
    _src, tgt = _enabled_plan(tmp_path)
    plan = plan_restore(hf_home=None, gpu=_GPU, target=tgt)
    assert restore_configs(plan) == []
    assert not tgt.exists()


def test_restore_creates_target_dir_when_enabled_even_if_source_missing(
    tmp_path: Path,
) -> None:
    _src, tgt = _enabled_plan(tmp_path)
    # Source dir doesn't exist (first-ever run, nothing tuned yet).
    plan = plan_restore(hf_home=tmp_path / "hf-empty", gpu=_GPU, target=tgt)
    assert restore_configs(plan) == []
    # AITER will write to the target during a fine-tuning run, so it
    # must exist even when we have nothing to seed it with.
    assert tgt.is_dir()


def test_restore_symlinks_every_file_in_source(tmp_path: Path) -> None:
    src = tmp_path / "vllm-shim" / "aiter-configs" / "gfx942-304cu"
    src.mkdir(parents=True)
    (src / "bf16_gemm.csv").write_text("M,N,K\n1,2,3\n")
    (src / "fp8_blockscale_gemm.csv").write_text("M,N,K\n4,5,6\n")
    tgt = tmp_path / "tgt"

    plan = plan_restore(hf_home=tmp_path, gpu=_GPU, target=tgt)
    restored = restore_configs(plan)

    assert restored == ["bf16_gemm.csv", "fp8_blockscale_gemm.csv"]
    assert (tgt / "bf16_gemm.csv").is_symlink()
    # Reading through the symlink must surface the source content; this
    # is the load-bearing invariant for AITER actually picking the
    # configs up rather than seeing an empty target file.
    assert (tgt / "bf16_gemm.csv").read_text() == "M,N,K\n1,2,3\n"


def test_restore_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "vllm-shim" / "aiter-configs" / "gfx942-304cu"
    src.mkdir(parents=True)
    (src / "bf16_gemm.csv").write_text("content")
    tgt = tmp_path / "tgt"

    plan = plan_restore(hf_home=tmp_path, gpu=_GPU, target=tgt)
    assert restore_configs(plan) == ["bf16_gemm.csv"]
    # Second run: nothing new to restore.
    assert restore_configs(plan) == []
    assert (tgt / "bf16_gemm.csv").is_symlink()


def test_restore_leaves_pre_existing_files_alone(tmp_path: Path) -> None:
    src = tmp_path / "vllm-shim" / "aiter-configs" / "gfx942-304cu"
    src.mkdir(parents=True)
    (src / "bf16_gemm.csv").write_text("from-source")
    tgt = tmp_path / "tgt"
    tgt.mkdir()
    # Operator (or a previous AITER run) already wrote a real file at
    # the destination; restore must not clobber it.
    (tgt / "bf16_gemm.csv").write_text("from-operator")

    plan = plan_restore(hf_home=tmp_path, gpu=_GPU, target=tgt)
    assert restore_configs(plan) == []
    assert (tgt / "bf16_gemm.csv").read_text() == "from-operator"
    assert not (tgt / "bf16_gemm.csv").is_symlink()


def test_restore_skips_non_files(tmp_path: Path) -> None:
    src = tmp_path / "vllm-shim" / "aiter-configs" / "gfx942-304cu"
    src.mkdir(parents=True)
    (src / "bf16_gemm.csv").write_text("content")
    (src / "nested").mkdir()  # subdirectory should be ignored
    tgt = tmp_path / "tgt"

    plan = plan_restore(hf_home=tmp_path, gpu=_GPU, target=tgt)
    assert restore_configs(plan) == ["bf16_gemm.csv"]
    assert not (tgt / "nested").exists()


def test_restore_returns_basenames_in_sorted_order(tmp_path: Path) -> None:
    # Stable ordering keeps the launch-info dump deterministic across
    # platforms / filesystem iteration order.
    src = tmp_path / "vllm-shim" / "aiter-configs" / "gfx942-304cu"
    src.mkdir(parents=True)
    for name in ("zeta.csv", "alpha.csv", "mu.csv"):
        (src / name).write_text("x")
    tgt = tmp_path / "tgt"

    plan = plan_restore(hf_home=tmp_path, gpu=_GPU, target=tgt)
    assert restore_configs(plan) == ["alpha.csv", "mu.csv", "zeta.csv"]
