"""Tests for the ``vllm-shim-tune`` orchestration."""

from collections.abc import Callable
from pathlib import Path

import pytest
from vllm_shim.aiter import tune
from vllm_shim.aiter.tune import (
    TuneResult,
    build_command,
    canonicalize_shape_files,
    discover_shape_files,
    known_targets,
    main,
    merge_shape_files,
    tune_target,
)

_HEADER = "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
_ROW_A = "1024,7168,512,False,torch.bfloat16,torch.bfloat16,False,False"
_ROW_B = "2048,7168,512,False,torch.bfloat16,torch.bfloat16,False,False"
_ROW_C = "4096,7168,512,False,torch.bfloat16,torch.bfloat16,False,False"


def _write(path: Path, *rows: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join((_HEADER, *rows)) + "\n")
    return path


# ---------- known_targets / _SPECS contract ----------


def test_known_targets_covers_all_restorable_targets() -> None:
    # Same keyspace as restore._TARGET_ENV minus the deliberately
    # excluded ``tuned_fmoe``. If a tuner is added for fmoe, this
    # assertion needs to flip and tuned_fmoe should appear here too.
    from vllm_shim.aiter.restore import _TARGET_ENV

    assert set(known_targets()) == set(_TARGET_ENV.keys()) - {"tuned_fmoe"}


def test_blockscale_bpreshuffle_reuses_blockscale_script() -> None:
    # Important AITER quirk worth pinning: the bpreshuffle variant is
    # the same script with one extra flag, not a separate file.
    plain = tune._SPECS["a8w8_blockscale_tuned_gemm"]
    pre = tune._SPECS["a8w8_blockscale_bpreshuffle_tuned_gemm"]
    assert plain.script == pre.script
    assert "--preshuffle" in pre.extra_args
    assert "--preshuffle" not in plain.extra_args


# ---------- discover_shape_files ----------


def test_discover_returns_empty_when_bucket_dir_missing(tmp_path: Path) -> None:
    assert discover_shape_files(tmp_path, "gfx942-304cu", "bf16_tuned_gemm") == []


def test_discover_walks_model_and_parallelism(tmp_path: Path) -> None:
    bucket = tmp_path / "aiter" / "shapes" / "gfx942-304cu"
    a = _write(bucket / "modelA" / "tp8" / "bf16_tuned_gemm.csv", _ROW_A)
    b = _write(bucket / "modelB" / "tp4-ep4" / "bf16_tuned_gemm.csv", _ROW_B)
    _write(bucket / "modelB" / "tp4-ep4" / "a8w8_tuned_gemm.csv", _ROW_C)
    found = discover_shape_files(tmp_path, "gfx942-304cu", "bf16_tuned_gemm")
    assert found == sorted([a, b])


# ---------- merge_shape_files ----------


def test_merge_dedups_across_files(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "f1.csv", _ROW_A, _ROW_B)
    f2 = _write(tmp_path / "f2.csv", _ROW_A, _ROW_C)
    dest = tmp_path / "merged" / "out.csv"
    rows = merge_shape_files([f1, f2], dest)
    assert rows == 3  # A, B, C (one A dropped)
    text = dest.read_text().splitlines()
    assert text[0] == _HEADER
    assert set(text[1:]) == {_ROW_A, _ROW_B, _ROW_C}


def test_merge_empty_list_writes_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "merged" / "out.csv"
    assert merge_shape_files([], dest) == 0
    assert not dest.exists()


def test_merge_rejects_header_mismatch(tmp_path: Path) -> None:
    f1 = _write(tmp_path / "f1.csv", _ROW_A)
    f2 = tmp_path / "f2.csv"
    f2.write_text("M,N,K\n1024,7168,512\n")
    with pytest.raises(ValueError, match="header mismatch"):
        merge_shape_files([f1, f2], tmp_path / "out.csv")


# ---------- canonicalize_shape_files ----------


def test_canonicalize_rewrites_quoted_dtype_in_place(tmp_path: Path) -> None:
    # Pre-fix capture data on operator volumes stored ``dtype`` and
    # ``outdtype`` with literal repr quotes (``'torch.bfloat16'``).
    # AITER's tuner reads those values and dies with "Invalid device
    # string". The canonicalize pass patches source files in place so
    # subsequent runs see clean data and operators don't have to
    # scrub the PV by hand.
    quoted_row = "1024,7168,512,False,'torch.bfloat16','torch.bfloat16',False,False"
    f = _write(tmp_path / "pre-fix.csv", quoted_row)
    rewritten = canonicalize_shape_files([f])
    assert rewritten == 1
    # On-disk file is now canonical; the quotes are gone.
    text = f.read_text().splitlines()
    assert text[0] == _HEADER
    assert text[1] == _ROW_A
    assert "'torch" not in f.read_text()


def test_canonicalize_skips_already_clean_files(tmp_path: Path) -> None:
    # Idempotent: an already-canonical file is not rewritten. This
    # matters because tune_target runs canonicalize_shape_files on
    # every invocation; we don't want needless disk churn or atime
    # noise on already-clean PVs.
    f = _write(tmp_path / "clean.csv", _ROW_A)
    mtime_before = f.stat().st_mtime_ns
    rewritten = canonicalize_shape_files([f])
    assert rewritten == 0
    assert f.stat().st_mtime_ns == mtime_before


def test_canonicalize_collapses_duplicate_rows(tmp_path: Path) -> None:
    # Once padding is applied, the canonical row tuple is the only
    # thing that matters for tune fidelity; identical rows are pure
    # waste. Canonicalize collapses them so the on-disk file shrinks
    # toward its minimal representation.
    f = _write(tmp_path / "dup.csv", _ROW_A, _ROW_A)
    canonicalize_shape_files([f])
    lines = f.read_text().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_canonicalize_pads_raw_m_and_dedups(tmp_path: Path) -> None:
    # Raw M values from pre-bucketing capture (e.g. m=257) round up
    # to AITER's tuning-grid bucket (272 in this case). Two raw rows
    # captured at different real M values may collapse onto the same
    # bucket; canonicalize dedups so the resulting file has one row
    # per (padded_m, ...) tuple.
    raw_257 = "257,7168,512,False,torch.bfloat16,torch.bfloat16,False,False"
    raw_258 = "258,7168,512,False,torch.bfloat16,torch.bfloat16,False,False"
    f = _write(tmp_path / "raw.csv", raw_257, raw_258)
    rewritten = canonicalize_shape_files([f])
    assert rewritten == 1
    lines = f.read_text().splitlines()
    # Both raw values pad to 272 (M<=1024 -> multiple of 32, 257 -> 288, 258 -> 288).
    # Wait: 257 rounds to nearest 32 mult: (257+31)//32*32 = 288.
    # Verify the actual bucket and confirm dedup.
    assert lines[0] == _HEADER
    assert len(lines) == 2  # header + one canonical row
    # The M column should now be 288 (raw 257/258 both pad to 288).
    assert lines[1].startswith("288,")


def test_canonicalize_uses_atomic_write(tmp_path: Path) -> None:
    # Tmp file is renamed over the original. After a successful
    # canonicalize call, no stray ``.tmp`` should remain in the dir.
    quoted_row = "1024,7168,512,False,'torch.bfloat16','torch.bfloat16',False,False"
    f = _write(tmp_path / "pre-fix.csv", quoted_row)
    canonicalize_shape_files([f])
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_canonicalize_handles_empty_file(tmp_path: Path) -> None:
    # Truncated / empty CSV (e.g. a partial capture write that lost
    # the header) must not raise. We just skip it.
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert canonicalize_shape_files([empty]) == 0


def test_tune_target_canonicalizes_before_merge(tmp_path: Path) -> None:
    # End-to-end: a pre-fix shape file under shapes/<bucket>/<model>/
    # <para>/ gets cleaned in place by tune_target before the merge
    # produces the untuned input. The source file on disk should be
    # canonical after tune_target returns, and the merged untuned
    # file should also be clean.
    quoted = "1024,7168,512,False,'torch.bfloat16','torch.bfloat16',False,False"
    source = tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "bf16_tuned_gemm.csv"
    _write(source, quoted)

    runner, _calls = _spy_runner()
    result = tune_target(
        "bf16_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=False,
        run=runner,
    )
    assert result.returncode == 0
    # Source file got patched.
    assert "'torch" not in source.read_text()
    # Untuned file (the tuner input) is also clean.
    assert "'torch" not in result.untuned_file.read_text()


# ---------- build_command ----------


def test_build_command_gradlib_shape() -> None:
    spec = tune._SPECS["bf16_tuned_gemm"]
    python = Path("/opt/venv/bin/python")
    aiter_root = Path("/sgl-workspace/aiter")
    untuned = Path("/tmp/untuned.csv")
    tuned = Path("/data/tuned.csv")
    cmd = build_command(
        spec,
        python=python,
        aiter_root=aiter_root,
        untuned_file=untuned,
        tuned_file=tuned,
        retune=False,
    )
    # gradlib uses --input_file / --tuned_file; csrc spelling would
    # silently mis-bind. Pin both flag spellings + their pairing.
    # Paths are stringified through ``str(Path(...))`` so the test
    # passes on both POSIX and Windows separator conventions.
    assert cmd == [
        str(python),
        str(aiter_root / "gradlib" / "gradlib" / "gemm_tuner.py"),
        "--input_file",
        str(untuned),
        "--tuned_file",
        str(tuned),
    ]


def test_build_command_csrc_passes_libtype(tmp_path: Path) -> None:
    spec = tune._SPECS["a8w8_tuned_gemm"]
    cmd = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=False,
    )
    assert "--untune_file" in cmd
    assert "--tune_file" in cmd
    i = cmd.index("--libtype")
    assert cmd[i : i + 2] == ["--libtype", "all"]


def test_build_command_blockscale_bpreshuffle_includes_preshuffle() -> None:
    spec = tune._SPECS["a8w8_blockscale_bpreshuffle_tuned_gemm"]
    cmd = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=False,
    )
    assert "--preshuffle" in cmd


def test_build_command_retune_appends_all() -> None:
    spec = tune._SPECS["a8w8_tuned_gemm"]
    cmd = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=True,
    )
    # --all is AITER's incremental-bypass flag; without it the tuner
    # already skips already-tuned shapes (which is what we want by
    # default).
    assert cmd[-1] == "--all"


# ---------- tune_target ----------


def _spy_runner() -> tuple[Callable[[list[str]], int], list[list[str]]]:
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    return runner, calls


def test_tune_target_no_shapes_yields_skipped(tmp_path: Path) -> None:
    # Pre-tuner pod: shapes directory empty. The orchestrator must
    # report the skip rather than invoke AITER with an empty input.
    runner, calls = _spy_runner()
    result = tune_target(
        "bf16_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=False,
        run=runner,
    )
    assert result.skipped_reason == "no captured shapes"
    assert result.rows_in == 0
    assert calls == []


def test_tune_target_dry_run_builds_command_but_does_not_invoke(tmp_path: Path) -> None:
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "bf16_tuned_gemm.csv",
        _ROW_A,
    )
    runner, calls = _spy_runner()
    result = tune_target(
        "bf16_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=True,
        run=runner,
    )
    assert result.skipped_reason == "dry-run"
    assert result.rows_in == 1
    assert calls == []
    assert "--input_file" in result.command


def test_tune_target_invokes_runner_when_not_dry(tmp_path: Path) -> None:
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "a8w8_tuned_gemm.csv",
        _ROW_A,
        _ROW_B,
    )
    runner, calls = _spy_runner()
    result = tune_target(
        "a8w8_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=False,
        run=runner,
    )
    assert result.returncode == 0
    assert result.rows_in == 2
    assert len(calls) == 1
    # The merged untuned CSV must exist on disk before the tuner is
    # invoked; otherwise AITER's argparse-bound file open will fail.
    assert result.untuned_file.exists()


def test_tune_target_runner_failure_surfaces_returncode(tmp_path: Path) -> None:
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "bf16_tuned_gemm.csv",
        _ROW_A,
    )

    def failing(_: list[str]) -> int:
        return 7

    result = tune_target(
        "bf16_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=False,
        run=failing,
    )
    assert result.returncode == 7


# ---------- main() ----------


def test_main_errors_when_shim_home_unresolvable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tune, "resolve_shim_home", lambda: None)
    rc = main(["--bucket", "gfx942-304cu"])
    assert rc == 1
    assert "could not resolve VLLM_SHIM_HOME" in capsys.readouterr().err


def test_main_errors_when_bucket_unresolvable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No --bucket and no ROCm GPU: explicit failure rather than silent
    # tuning under the wrong bucket key.
    monkeypatch.setattr(tune, "probe_rocm", lambda: None)
    rc = main(["--shim-home", str(tmp_path)])
    assert rc == 1
    assert "no ROCm GPU detected" in capsys.readouterr().err


def test_main_list_prints_per_target_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "bf16_tuned_gemm.csv",
        _ROW_A,
    )
    (tmp_path / "aiter" / "configs" / "gfx942-304cu").mkdir(parents=True)
    (tmp_path / "aiter" / "configs" / "gfx942-304cu" / "a8w8_tuned_gemm.csv").write_text(
        "header\n"
    )
    rc = main(
        ["--shim-home", str(tmp_path), "--bucket", "gfx942-304cu", "--list"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # Captured but not tuned:
    assert "bf16_tuned_gemm: 1 shape file(s), untuned" in out
    # Tuned but no captures (legacy from a previous bucket):
    assert "a8w8_tuned_gemm: 0 shape file(s), tuned" in out


def test_main_dry_run_iterates_only_chosen_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "a8w8_tuned_gemm.csv",
        _ROW_A,
    )
    invoked: list[list[str]] = []

    def fake_tune_target(target: str, **kw: object) -> TuneResult:
        # Capture the per-target call so we can assert on its identity.
        invoked.append([target])
        return TuneResult(
            target=target,
            untuned_file=Path("u.csv"),
            tuned_file=Path("t.csv"),
            rows_in=1,
            command=["echo"],
            returncode=None,
            skipped_reason="dry-run",
        )

    monkeypatch.setattr(tune, "tune_target", fake_tune_target)
    rc = main(
        [
            "--shim-home",
            str(tmp_path),
            "--bucket",
            "gfx942-304cu",
            "--target",
            "a8w8_tuned_gemm",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert invoked == [["a8w8_tuned_gemm"]]
    assert "skipped (dry-run)" in capsys.readouterr().err
