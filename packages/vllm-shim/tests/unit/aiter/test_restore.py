"""Tests for AITER tuned-config restoration via env-var override."""

from pathlib import Path

from vllm_shim.aiter.capture import REASON_ENABLED, REASON_NO_GPU, REASON_NO_HF_HOME
from vllm_shim.aiter.restore import plan_restore, restore_configs
from vllm_shim.cli.rocm_probe import GpuAgent

_GPU = GpuAgent(gfx_target="gfx942", compute_units=304, marketing_name="MI300X")


def _seed(hf_home: Path, *names: str) -> Path:
    """Create the bucket directory under hf_home and drop empty CSVs."""
    bucket_dir = hf_home / "vllm-shim" / "aiter-configs" / "gfx942-304cu"
    bucket_dir.mkdir(parents=True)
    for name in names:
        (bucket_dir / name).write_text("M,N,K\n")
    return bucket_dir


# ---------- plan_restore ----------


def test_plan_disabled_when_no_gpu(tmp_path: Path) -> None:
    plan = plan_restore(hf_home=tmp_path, gpu=None)
    assert plan.enabled is False
    assert plan.source is None
    assert plan.reason == REASON_NO_GPU


def test_plan_disabled_when_hf_home_unresolvable() -> None:
    plan = plan_restore(hf_home=None, gpu=_GPU)
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


# ---------- restore_configs ----------


def test_restore_returns_empty_when_disabled(tmp_path: Path) -> None:
    plan = plan_restore(hf_home=None, gpu=_GPU)
    assert restore_configs(plan) == {}


def test_restore_returns_empty_when_source_missing(tmp_path: Path) -> None:
    # First-ever run, nothing tuned yet: the source directory doesn't
    # exist. Restore should be a clean no-op (not raise).
    plan = plan_restore(hf_home=tmp_path / "hf-empty", gpu=_GPU)
    assert restore_configs(plan) == {}


def test_restore_maps_known_targets_to_their_env_vars(tmp_path: Path) -> None:
    src = _seed(tmp_path, "bf16_tuned_gemm.csv", "a8w8_tuned_gemm.csv")
    plan = plan_restore(hf_home=tmp_path, gpu=_GPU)
    overrides = restore_configs(plan)
    # Each tuned-config CSV becomes an AITER_CONFIG_* override pointing
    # at the absolute path under our persistent store.
    assert overrides == {
        "AITER_CONFIG_GEMM_BF16": str(src / "bf16_tuned_gemm.csv"),
        "AITER_CONFIG_GEMM_A8W8": str(src / "a8w8_tuned_gemm.csv"),
    }


def test_restore_covers_all_known_aiter_targets(tmp_path: Path) -> None:
    # Sanity: every basename in the mapping table produces an override.
    # When AITER adds a new tuned-config target, we want this test to
    # fail loudly so we update _TARGET_ENV in lockstep.
    src = _seed(
        tmp_path,
        "bf16_tuned_gemm.csv",
        "a4w4_blockscale_tuned_gemm.csv",
        "a8w8_tuned_gemm.csv",
        "a8w8_bpreshuffle_tuned_gemm.csv",
        "a8w8_blockscale_tuned_gemm.csv",
        "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "bf16_tuned_batched_gemm.csv",
        "a8w8_tuned_batched_gemm.csv",
        "tuned_fmoe.csv",
    )
    plan = plan_restore(hf_home=tmp_path, gpu=_GPU)
    overrides = restore_configs(plan)
    assert set(overrides.keys()) == {
        "AITER_CONFIG_GEMM_BF16",
        "AITER_CONFIG_GEMM_A4W4",
        "AITER_CONFIG_GEMM_A8W8",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        "AITER_CONFIG_BF16_BATCHED_GEMM",
        "AITER_CONFIG_A8W8_BATCHED_GEMM",
        "AITER_CONFIG_FMOE",
    }
    assert all(v == str(src / Path(v).name) for v in overrides.values())


def test_restore_skips_unknown_targets(tmp_path: Path) -> None:
    # A future AITER target (or an operator-dropped extra file) that we
    # don't know about should be silently ignored rather than blow up.
    _seed(tmp_path, "bf16_tuned_gemm.csv", "future_kernel_tuned_gemm.csv")
    plan = plan_restore(hf_home=tmp_path, gpu=_GPU)
    overrides = restore_configs(plan)
    assert "AITER_CONFIG_GEMM_BF16" in overrides
    assert len(overrides) == 1


def test_restore_skips_subdirectories(tmp_path: Path) -> None:
    src = _seed(tmp_path, "bf16_tuned_gemm.csv")
    (src / "nested").mkdir()  # subdir should be ignored
    plan = plan_restore(hf_home=tmp_path, gpu=_GPU)
    overrides = restore_configs(plan)
    assert set(overrides.keys()) == {"AITER_CONFIG_GEMM_BF16"}


def test_restore_does_no_filesystem_writes(tmp_path: Path) -> None:
    # The function is read-only: it must not create directories or
    # files (especially not write to AITER's hardcoded /tmp paths).
    src = _seed(tmp_path, "bf16_tuned_gemm.csv")
    before = sorted(p.name for p in src.iterdir())
    plan = plan_restore(hf_home=tmp_path, gpu=_GPU)
    restore_configs(plan)
    after = sorted(p.name for p in src.iterdir())
    assert before == after
