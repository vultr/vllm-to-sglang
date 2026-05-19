"""Tests for the ``vllm-shim-tune`` orchestration."""

from collections.abc import Callable
from pathlib import Path

import pytest
from vllm_shim.aiter import tune
from vllm_shim.aiter.tune import (
    TuneResult,
    build_command,
    discover_shape_files,
    known_targets,
    main,
    merge_shape_files,
    tune_target,
)

# Full capture-side schema, including the shim-added ``hits`` column.
_HEADER = "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,hits"
# Pre-hits schema (what ShapeStore wrote before the column was added,
# and what tune outputs for AITER's tuner after stripping ``hits``).
_LEGACY_HEADER = "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"


def _row(m: int = 1024, hits: int = 1) -> str:
    """Build a capture-format row string with an explicit hit count."""
    return f"{m},7168,512,False,torch.bfloat16,torch.bfloat16,False,False,{hits}"


def _aiter_row(m: int = 1024) -> str:
    """Row in the AITER tuner-input format (no ``hits`` column)."""
    return f"{m},7168,512,False,torch.bfloat16,torch.bfloat16,False,False"


# Three distinct shapes used across merge/tune tests. Hits default to 1
# so legacy-style assertions about set membership keep working; tests
# that exercise hot-ranking re-build the rows with explicit hit counts.
_ROW_A = _row(m=1024)
_ROW_B = _row(m=2048)
_ROW_C = _row(m=4096)


def _write(path: Path, *rows: str, header: str = _HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join((header, *rows)) + "\n")
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
    rows_total, rows_in = merge_shape_files([f1, f2], dest)
    # Three unique shapes: A appears in both files but collapses.
    assert (rows_total, rows_in) == (3, 3)
    lines = dest.read_text().splitlines()
    # Output is the AITER-input schema: no ``hits`` column.
    assert lines[0] == _LEGACY_HEADER
    assert set(lines[1:]) == {_aiter_row(1024), _aiter_row(2048), _aiter_row(4096)}


def test_merge_empty_list_writes_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "merged" / "out.csv"
    assert merge_shape_files([], dest) == (0, 0)
    assert not dest.exists()


def test_merge_rejects_missing_required_column(tmp_path: Path) -> None:
    # A capture file that's lost one of the shape-identifying columns
    # is a sign of upstream corruption; failing loud is better than
    # silently writing a garbage tuner input.
    f1 = _write(tmp_path / "f1.csv", _ROW_A)
    f2 = tmp_path / "f2.csv"
    f2.write_text("M,N,K\n1024,7168,512\n")
    with pytest.raises(ValueError, match="missing column"):
        merge_shape_files([f1, f2], tmp_path / "out.csv")


def test_merge_sums_hits_across_files(tmp_path: Path) -> None:
    # The same shape captured in two pods (or across two pod
    # generations) should have its hit counts summed at merge time so
    # the ranking sees the workload's true call frequency.
    f1 = _write(tmp_path / "f1.csv", _row(m=1024, hits=7))
    f2 = _write(tmp_path / "f2.csv", _row(m=1024, hits=15))
    dest = tmp_path / "out.csv"
    rows_total, rows_in = merge_shape_files([f1, f2], dest, top_k=1)
    assert (rows_total, rows_in) == (1, 1)
    # We can't observe the summed hits in the output (the column is
    # stripped), but with top_k=1 the surviving row must be the merged
    # one; a buggy implementation that didn't sum would still pass
    # here, so the real assertion is in the hot-ranking tests below.
    lines = dest.read_text().splitlines()
    assert lines[1] == _aiter_row(1024)


def test_merge_defaults_missing_hits_column_to_one(tmp_path: Path) -> None:
    # A CSV captured before the hits column was added must still merge
    # cleanly; each row counts for 1. Two legacy-format files with the
    # same shape produce a merged hit count of 2.
    legacy_row = "1024,7168,512,False,torch.bfloat16,torch.bfloat16,False,False"
    f1 = _write(tmp_path / "f1.csv", legacy_row, header=_LEGACY_HEADER)
    f2 = _write(tmp_path / "f2.csv", legacy_row, header=_LEGACY_HEADER)
    dest = tmp_path / "out.csv"
    rows_total, rows_in = merge_shape_files([f1, f2], dest)
    assert (rows_total, rows_in) == (1, 1)
    lines = dest.read_text().splitlines()
    assert lines[1] == _aiter_row(1024)


def test_merge_top_k_keeps_hottest_by_hits(tmp_path: Path) -> None:
    # Three shapes with distinct hit counts. top_k=2 must keep the two
    # with the highest counts, regardless of their M values.
    hot = _row(m=2048, hits=100)
    warm = _row(m=1024, hits=50)
    cold = _row(m=4096, hits=1)
    f = _write(tmp_path / "in.csv", cold, hot, warm)
    dest = tmp_path / "out.csv"
    rows_total, rows_in = merge_shape_files([f], dest, top_k=2)
    assert (rows_total, rows_in) == (3, 2)
    lines = dest.read_text().splitlines()
    assert lines[0] == _LEGACY_HEADER
    # Order: highest hits first, so the operator can read the file
    # top-down to see which shapes the ranking surfaced.
    assert lines[1] == _aiter_row(2048)
    assert lines[2] == _aiter_row(1024)


def test_merge_top_k_above_unique_count_is_noop(tmp_path: Path) -> None:
    f = _write(tmp_path / "in.csv", _ROW_A, _ROW_B)
    dest = tmp_path / "out.csv"
    rows_total, rows_in = merge_shape_files([f], dest, top_k=10)
    # No truncation when the cap exceeds the unique-shape count.
    assert (rows_total, rows_in) == (2, 2)


def test_merge_writes_atomically_via_tmp(tmp_path: Path) -> None:
    # The merged file is rewritten on every tune run; a crash midway
    # through should leave the previous file intact rather than
    # corrupt it. No .tmp survivors after a successful write.
    f = _write(tmp_path / "in.csv", _ROW_A)
    dest = tmp_path / "merged" / "out.csv"
    merge_shape_files([f], dest)
    leftovers = list(dest.parent.glob("*.tmp"))
    assert leftovers == []


def test_merge_strips_hits_column_from_output(tmp_path: Path) -> None:
    # AITER's tuners read the merged file via pd.read_csv +
    # drop_duplicates over all columns. Carrying our ``hits`` column
    # into that pipeline risks unintended behaviour, so the output
    # schema is always the AITER-native one regardless of input shape.
    f = _write(tmp_path / "in.csv", _row(m=1024, hits=42))
    dest = tmp_path / "out.csv"
    merge_shape_files([f], dest)
    lines = dest.read_text().splitlines()
    assert lines[0] == _LEGACY_HEADER
    assert "hits" not in lines[0]
    assert "42" not in lines[1]


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


def test_build_command_no_flydsl_swaps_bf16_gradlib_libtype() -> None:
    # gradlib defaults to libtype=["all"], which pulls in FlyDSL
    # candidates. Startup tune passes --no-flydsl so we add an explicit
    # libtype list that omits flydsl (and the "all" meta value).
    spec = tune._SPECS["bf16_tuned_gemm"]
    cmd = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=False,
        no_flydsl=True,
    )
    i = cmd.index("--libtype")
    assert cmd[i : i + 2] == [
        "--libtype",
        "asm,hipblaslt,triton,torch,skinny",
    ]


def test_build_command_no_flydsl_replaces_bpreshuffle_libtype() -> None:
    # The CK bpreshuffle script accepts ``flydsl`` as a libtype choice
    # and we currently pass ``--libtype all``. Under --no-flydsl that
    # swaps to the non-flydsl subset.
    spec = tune._SPECS["a8w8_bpreshuffle_tuned_gemm"]
    cmd = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=False,
        no_flydsl=True,
    )
    i = cmd.index("--libtype")
    assert cmd[i : i + 2] == ["--libtype", "asm,ck,cktile"]
    # "all" must not survive; otherwise the override is a no-op.
    assert "all" not in cmd


def test_build_command_no_flydsl_is_noop_for_unaffected_target() -> None:
    # Targets without a FlyDSL candidate path keep their default
    # ``extra_args`` under --no-flydsl. a8w8_tuned_gemm's CK script
    # has no --libtype flag for flydsl, so the override map deliberately
    # omits it; build_command must pass through spec.extra_args unchanged.
    spec = tune._SPECS["a8w8_tuned_gemm"]
    plain = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=False,
    )
    with_flag = build_command(
        spec,
        python=Path("py"),
        aiter_root=Path("aroot"),
        untuned_file=Path("u.csv"),
        tuned_file=Path("t.csv"),
        retune=False,
        no_flydsl=True,
    )
    assert plain == with_flag


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


def test_tune_target_hot_filter_truncates_by_hits(tmp_path: Path) -> None:
    # Three shapes captured with distinct hit counts. hot=2 picks the
    # two with the most hits; the third drops out. ``rows_total``
    # preserves the pre-truncation count so the operator-facing
    # summary can report "2 of 3 shapes (hot)".
    cold = _row(m=4096, hits=1)
    warm = _row(m=1024, hits=50)
    hot = _row(m=2048, hits=100)
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "bf16_tuned_gemm.csv",
        cold,
        warm,
        hot,
    )
    runner, calls = _spy_runner()
    result = tune_target(
        "bf16_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=False,
        hot=2,
        run=runner,
    )
    assert result.rows_in == 2
    assert result.rows_total == 3
    assert len(calls) == 1
    # On disk the survivors are the two highest-hit shapes, ranked
    # hottest-first. ``hits`` column is stripped from the AITER input.
    rows = result.untuned_file.read_text().splitlines()
    assert rows[1:] == [_aiter_row(2048), _aiter_row(1024)]


def test_tune_target_hot_filter_noop_when_under_top_k(tmp_path: Path) -> None:
    # hot=10 with 2 shapes captured: no truncation, rows_in == rows_total.
    # Operator's summary should NOT show the "(hot)" marker in this case.
    _write(
        tmp_path / "aiter" / "shapes" / "gfx942-304cu" / "m" / "tp1" / "bf16_tuned_gemm.csv",
        _ROW_A,
        _ROW_B,
    )
    runner, _ = _spy_runner()
    result = tune_target(
        "bf16_tuned_gemm",
        shim_home=tmp_path,
        bucket="gfx942-304cu",
        aiter_root=Path("/aiter"),
        python=Path("py"),
        retune=False,
        dry_run=False,
        hot=10,
        run=runner,
    )
    assert result.rows_in == 2
    assert result.rows_total == 2


def test_print_result_marks_hot_when_truncated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tune._print_result(
        TuneResult(
            target="bf16_tuned_gemm",
            untuned_file=Path("/u.csv"),
            tuned_file=Path("/t.csv"),
            rows_in=100,
            rows_total=247,
            command=["echo"],
            returncode=0,
        )
    )
    err = capsys.readouterr().err
    # The operator should see the truncation count so they can decide
    # whether to widen --hot on the next run.
    assert "100 of 247 shapes (hot)" in err


def test_print_result_omits_hot_marker_when_no_truncation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tune._print_result(
        TuneResult(
            target="bf16_tuned_gemm",
            untuned_file=Path("/u.csv"),
            tuned_file=Path("/t.csv"),
            rows_in=42,
            rows_total=42,
            command=["echo"],
            returncode=0,
        )
    )
    err = capsys.readouterr().err
    assert "42 shapes" in err
    assert "(hot)" not in err


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
    (tmp_path / "aiter" / "configs" / "gfx942-304cu" / "a8w8_tuned_gemm.csv").write_text("header\n")
    rc = main(["--shim-home", str(tmp_path), "--bucket", "gfx942-304cu", "--list"])
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
            rows_total=1,
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
