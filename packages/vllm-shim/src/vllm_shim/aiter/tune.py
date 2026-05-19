"""Run AITER's tuner over captured shapes to produce tuned configs.

Bridges two CSV directories under ``$VLLM_SHIM_HOME/aiter/``:

- Input: ``shapes/<bucket>/<model>/<parallelism>/<target>.csv`` (the
  miss lines our stderr-tee captured at serve time, partitioned by
  bucket+model+parallelism).
- Output: ``configs/<bucket>/<target>.csv`` (tuned configs the shim's
  restore step points AITER at via ``AITER_CONFIG_*`` env vars,
  partitioned by bucket only because tuned rows are shape-keyed and
  reusable across models).

Per (bucket, target) we union the per-(model,parallelism) shape files
into a single deduped untuned CSV, then shell out to AITER's tuner
with that file as input and the bucket's tuned CSV as output. AITER's
tuner already does incremental tuning by default (it diffs the input
against the existing output and only tunes the new shapes), so the
shim's ``--retune`` flag just maps to AITER's own ``--all`` flag.

There are two CLI conventions among AITER's tuners and three classes
of target. The mapping is hardcoded - AITER doesn't expose it as
metadata, the only sources of truth are the script files themselves,
and adding a new target is a manual edit anyway. We exclude
``tuned_fmoe`` for now: it uses a third CLI (``gemm_moe_tune.py``
with ``-i -o -o2 --last``) and a different capture-log schema than
our stderr-tee parses today.

This module is an operator tool, not a server-time path. It's invoked
via the ``vllm-shim-tune`` console script, typically from a shell in
a pod that has ROCm + AITER available.
"""

import argparse
import csv
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vllm_shim.aiter.capture import resolve_shim_home
from vllm_shim.aiter.shape_store import HITS_COLUMN, SHAPE_COLUMNS, parse_hits
from vllm_shim.cli.rocm_probe import bucket as bucket_for_gpu
from vllm_shim.cli.rocm_probe import probe as probe_rocm

# Default to the layout baked into the SGLang-ROCm shim image. ``/opt/venv``
# is the Python 3.10 env that ships AITER's tuner; ``/sgl-workspace/aiter``
# is the source tree the tuner scripts live in.
DEFAULT_AITER_ROOT = Path("/sgl-workspace/aiter")
DEFAULT_PYTHON = Path("/opt/venv/bin/python")


@dataclass(frozen=True, slots=True)
class TunerSpec:
    """How to invoke AITER's tuner for one tuned-config target.

    ``script`` is relative to the AITER repo root. ``input_flag`` and
    ``output_flag`` differ between the two tuner families (gradlib's
    ``GemmTuner`` uses ``--input_file``/``--tuned_file``; the csrc
    tuners inherit ``-i``/``-o`` from ``aiter.utility.base_tuner``).
    ``extra_args`` holds target-specific knobs (``--libtype``,
    ``--preshuffle``); they're appended verbatim to the command line.
    """

    target: str
    script: str
    input_flag: str
    output_flag: str
    extra_args: tuple[str, ...] = ()


# Hardcoded mapping: AITER target basename -> how to tune it. Keep in
# sync with ``vllm_shim.aiter.restore._TARGET_ENV`` (same keyspace).
# ``tuned_fmoe`` is deliberately absent; see module docstring.
_SPECS: dict[str, TunerSpec] = {
    "bf16_tuned_gemm": TunerSpec(
        target="bf16_tuned_gemm",
        script="gradlib/gradlib/gemm_tuner.py",
        input_flag="--input_file",
        output_flag="--tuned_file",
    ),
    "a4w4_blockscale_tuned_gemm": TunerSpec(
        target="a4w4_blockscale_tuned_gemm",
        script="csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all"),
    ),
    "a8w8_tuned_gemm": TunerSpec(
        target="a8w8_tuned_gemm",
        script="csrc/ck_gemm_a8w8/gemm_a8w8_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all"),
    ),
    "a8w8_bpreshuffle_tuned_gemm": TunerSpec(
        target="a8w8_bpreshuffle_tuned_gemm",
        script="csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all"),
    ),
    "a8w8_blockscale_tuned_gemm": TunerSpec(
        target="a8w8_blockscale_tuned_gemm",
        script="csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all"),
    ),
    # Same script as blockscale, with the preshuffle flag flipped. The
    # in-AITER variant identification happens via the flag, not the
    # script path; both write into the script's own default output
    # name unless overridden, which is why we always pass --tune_file.
    "a8w8_blockscale_bpreshuffle_tuned_gemm": TunerSpec(
        target="a8w8_blockscale_bpreshuffle_tuned_gemm",
        script="csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all", "--preshuffle"),
    ),
    "bf16_tuned_batched_gemm": TunerSpec(
        target="bf16_tuned_batched_gemm",
        script="csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all"),
    ),
    "a8w8_tuned_batched_gemm": TunerSpec(
        target="a8w8_tuned_batched_gemm",
        script="csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py",
        input_flag="--untune_file",
        output_flag="--tune_file",
        extra_args=("--libtype", "all"),
    ),
}


# Per-target override for ``extra_args`` when ``--no-flydsl`` is set.
# Two targets enumerate FlyDSL kernels as tuning candidates: gradlib's
# bf16 tuner (default libtype=["all"]) and the CK bpreshuffle a8w8
# script (we pass --libtype all). FlyDSL candidates JIT-compile per
# kernel during benchmarking, which dominates per-shape tune wall-time;
# the startup-tune path swaps to a libtype list that excludes flydsl
# (and the "all" meta value that re-includes it). Other targets have
# no FlyDSL path at all, so they're absent from this map and the flag
# is a no-op for them.
_NO_FLYDSL_EXTRA_ARGS: dict[str, tuple[str, ...]] = {
    "bf16_tuned_gemm": ("--libtype", "asm,hipblaslt,triton,torch,skinny"),
    "a8w8_bpreshuffle_tuned_gemm": ("--libtype", "asm,ck,cktile"),
}


def known_targets() -> tuple[str, ...]:
    """Stable ordering for ``--list`` output and default iteration."""
    return tuple(_SPECS.keys())


def shapes_root(shim_home: Path, bucket: str) -> Path:
    """Directory whose subtree holds captured shape CSVs for one bucket."""
    return shim_home / "aiter" / "shapes" / bucket


def configs_root(shim_home: Path, bucket: str) -> Path:
    """Directory tuned-config CSVs for one bucket are written into."""
    return shim_home / "aiter" / "configs" / bucket


def discover_shape_files(shim_home: Path, bucket: str, target: str) -> list[Path]:
    """All ``<target>.csv`` files under ``shapes/<bucket>/*/*/``.

    Sorted deterministically so dry-run output and merged-file row
    order don't depend on filesystem walk order.
    """
    root = shapes_root(shim_home, bucket)
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*/*/{target}.csv"))


def merge_shape_files(
    files: list[Path],
    destination: Path,
    *,
    top_k: int | None = None,
) -> tuple[int, int]:
    """Read per-source shape CSVs and write AITER's untuned input.

    Inputs are the per-(model, parallelism) shape capture CSVs written
    by ``ShapeStore``. They carry the shim's full schema including the
    ``hits`` column; this function:

    1. Sums hits per shape key across every input file.
    2. Optionally truncates to the ``top_k`` shapes with the highest
       summed hits, descending. Ties break by insertion order.
    3. Writes ``destination`` with just the AITER-required columns
       (``hits`` is dropped). AITER's tuners use ``pd.read_csv`` and
       ``drop_duplicates`` over all columns; stripping ``hits`` keeps
       the tuner contract unchanged whether or not we ranked.

    Returns ``(rows_total, rows_in)``: unique-shape count after merge,
    and the row count actually written to ``destination``. They differ
    only when ``top_k`` truncates.

    Missing ``hits`` cells default to 1 via ``parse_hits``, so CSVs
    captured before the column existed read back as "seen at least
    once" rather than being treated as zero-weight.
    """
    aggregated: dict[tuple[str, ...], int] = {}
    for path in files:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            missing = [c for c in SHAPE_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"missing column(s) {missing} in {path}: got {reader.fieldnames}")
            for row in reader:
                key = tuple(row[c] for c in SHAPE_COLUMNS)
                hits = parse_hits(row.get(HITS_COLUMN))
                aggregated[key] = aggregated.get(key, 0) + hits
    rows_total = len(aggregated)
    if rows_total == 0:
        return 0, 0
    items = list(aggregated.items())
    if top_k is not None and top_k < rows_total:
        # Stable sort by -hits keeps the input-encounter order for
        # equally-hot shapes, so operator-side debugging can correlate
        # rows back to the source files in deterministic order.
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:top_k]
    rows_in = len(items)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SHAPE_COLUMNS)
        for key, _hits in items:
            writer.writerow(list(key))
    tmp.replace(destination)
    return rows_total, rows_in


def build_command(
    spec: TunerSpec,
    *,
    python: Path,
    aiter_root: Path,
    untuned_file: Path,
    tuned_file: Path,
    retune: bool,
    no_flydsl: bool = False,
) -> list[str]:
    """Assemble the argv that invokes AITER's tuner for ``spec``.

    ``retune`` maps to AITER's own ``--all`` flag, which forces a
    re-tune of every shape in the input even when an existing tuned
    file is present. Without it, AITER diffs input against output and
    only tunes the new rows (incremental tuning is the default).

    ``no_flydsl`` swaps ``spec.extra_args`` for the entry in
    ``_NO_FLYDSL_EXTRA_ARGS`` when one exists; targets without a
    FlyDSL path keep their default args.
    """
    extra_args = (
        _NO_FLYDSL_EXTRA_ARGS.get(spec.target, spec.extra_args) if no_flydsl else spec.extra_args
    )
    cmd: list[str] = [
        str(python),
        str(aiter_root / spec.script),
        spec.input_flag,
        str(untuned_file),
        spec.output_flag,
        str(tuned_file),
        *extra_args,
    ]
    if retune:
        cmd.append("--all")
    return cmd


@dataclass(frozen=True, slots=True)
class TuneResult:
    """Outcome of tuning one target. ``returncode`` is None on dry-run / skip.

    ``rows_in`` is the row count actually fed to AITER's tuner (after
    ``--hot`` truncation). ``rows_total`` is the merged count before
    truncation; equal to ``rows_in`` when ``--hot`` wasn't set or
    didn't trigger.
    """

    target: str
    untuned_file: Path
    tuned_file: Path
    rows_in: int
    rows_total: int
    command: list[str]
    returncode: int | None
    skipped_reason: str | None = None


def tune_target(
    target: str,
    *,
    shim_home: Path,
    bucket: str,
    aiter_root: Path,
    python: Path,
    retune: bool,
    dry_run: bool,
    hot: int | None = None,
    no_flydsl: bool = False,
    run: Callable[[list[str]], int] | None = None,
) -> TuneResult:
    """Tune one target end-to-end: discover, merge, invoke.

    ``run`` is an injection point for tests; defaults to
    ``subprocess.run`` with stdout/stderr inherited from the parent
    so the operator sees AITER's progress in real time.
    """
    spec = _SPECS[target]
    files = discover_shape_files(shim_home, bucket, target)
    tuned_file = configs_root(shim_home, bucket) / f"{target}.csv"
    untuned_file = shim_home / "aiter" / "untuned" / bucket / f"{target}.csv"
    if not files:
        return TuneResult(
            target=target,
            untuned_file=untuned_file,
            tuned_file=tuned_file,
            rows_in=0,
            rows_total=0,
            command=[],
            returncode=None,
            skipped_reason="no captured shapes",
        )
    rows_total, rows = merge_shape_files(files, untuned_file, top_k=hot)
    cmd = build_command(
        spec,
        python=python,
        aiter_root=aiter_root,
        untuned_file=untuned_file,
        tuned_file=tuned_file,
        retune=retune,
        no_flydsl=no_flydsl,
    )
    if dry_run:
        return TuneResult(
            target=target,
            untuned_file=untuned_file,
            tuned_file=tuned_file,
            rows_in=rows,
            rows_total=rows_total,
            command=cmd,
            returncode=None,
            skipped_reason="dry-run",
        )
    tuned_file.parent.mkdir(parents=True, exist_ok=True)
    runner = run or _default_runner
    rc = runner(cmd)
    return TuneResult(
        target=target,
        untuned_file=untuned_file,
        tuned_file=tuned_file,
        rows_in=rows,
        rows_total=rows_total,
        command=cmd,
        returncode=rc,
    )


def _default_runner(cmd: list[str]) -> int:
    """Run AITER's tuner with inherited stdout/stderr; return its rc."""
    return subprocess.run(cmd, check=False).returncode


def _resolve_bucket(explicit: str | None) -> str | None:
    """Use ``--bucket`` if given, else infer from a probed ROCm GPU."""
    if explicit:
        return explicit
    gpu = probe_rocm()
    return bucket_for_gpu(gpu) if gpu else None


def _print_list(shim_home: Path, bucket: str) -> int:
    """``--list`` output: target -> (file count, total shape rows)."""
    sys.stdout.write(f"bucket: {bucket}\n")
    sys.stdout.write(f"shim home: {shim_home}\n")
    sys.stdout.write(f"shapes root: {shapes_root(shim_home, bucket)}\n")
    sys.stdout.write(f"configs root: {configs_root(shim_home, bucket)}\n\n")
    for target in known_targets():
        files = discover_shape_files(shim_home, bucket, target)
        tuned = configs_root(shim_home, bucket) / f"{target}.csv"
        tuned_marker = "tuned" if tuned.exists() else "untuned"
        sys.stdout.write(f"  {target}: {len(files)} shape file(s), {tuned_marker}\n")
    return 0


def _print_result(result: TuneResult) -> None:
    if result.skipped_reason:
        sys.stderr.write(f"  {result.target}: skipped ({result.skipped_reason})\n")
        if result.command:
            sys.stderr.write(f"    would run: {' '.join(result.command)}\n")
        return
    status = "ok" if result.returncode == 0 else f"failed (rc={result.returncode})"
    if result.rows_in < result.rows_total:
        shapes = f"{result.rows_in} of {result.rows_total} shapes (hot)"
    else:
        shapes = f"{result.rows_in} shapes"
    sys.stderr.write(f"  {result.target}: {shapes} -> {result.tuned_file} [{status}]\n")


def main(argv: list[str] | None = None) -> int:
    """``vllm-shim-tune`` entry point."""
    parser = argparse.ArgumentParser(
        prog="vllm-shim-tune",
        description=(
            "Run AITER's tuner over shapes captured by the vllm-shim "
            "serve-time tee. Writes tuned configs the shim's restore "
            "step will pick up on the next launch."
        ),
    )
    parser.add_argument(
        "--bucket",
        help=("Hardware bucket (e.g. gfx942-304cu). Inferred from rocminfo when omitted."),
    )
    parser.add_argument(
        "--target",
        choices=known_targets(),
        help="Tune only this target. Defaults to all known targets.",
    )
    parser.add_argument(
        "--aiter-root",
        type=Path,
        default=DEFAULT_AITER_ROOT,
        help=f"Path to the AITER source tree (default: {DEFAULT_AITER_ROOT}).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help=(f"Python interpreter that has AITER installed (default: {DEFAULT_PYTHON})."),
    )
    parser.add_argument(
        "--shim-home",
        type=Path,
        help=(
            "Override VLLM_SHIM_HOME for this invocation. Defaults to "
            "$VLLM_SHIM_HOME, then ~/.vllm-shim."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print captured/tuned state for each target and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the AITER command for each target without running it.",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help=(
            "Force re-tune of every shape (passes --all to AITER's "
            "tuner). Default is incremental: skip shapes that already "
            "have a tuned row."
        ),
    )
    parser.add_argument(
        "--hot",
        type=int,
        metavar="N",
        help=(
            "Tune only the N hottest shapes per target, ranked by the "
            "summed 'hits' column captured at serve time (one increment "
            "per AITER miss line). Default: tune every captured shape. "
            "Composes with --retune; with the default (incremental) "
            "mode AITER skips already-tuned rows, so growing --hot "
            "across runs converges on the full set."
        ),
    )
    parser.add_argument(
        "--no-flydsl",
        action="store_true",
        help=(
            "Exclude FlyDSL candidates from the tuner's libtype list "
            "for the two targets that include them (bf16_tuned_gemm, "
            "a8w8_bpreshuffle_tuned_gemm). FlyDSL candidates JIT-compile "
            "per kernel and dominate per-shape benchmark time; dropping "
            "them shrinks the candidate pool so a time-budgeted run "
            "covers more shapes. The vllm-shim startup-tune path passes "
            "this automatically; operator-driven runs can opt in to "
            "compare timings or trim the candidate set."
        ),
    )
    args = parser.parse_args(argv)

    shim_home = args.shim_home or resolve_shim_home()
    if shim_home is None:
        sys.stderr.write("could not resolve VLLM_SHIM_HOME\n")
        return 1
    bucket = _resolve_bucket(args.bucket)
    if bucket is None:
        sys.stderr.write(
            "no ROCm GPU detected and no --bucket supplied; pass "
            "--bucket explicitly to tune from an offline shape capture\n"
        )
        return 1

    if args.list:
        return _print_list(shim_home, bucket)

    targets = (args.target,) if args.target else known_targets()
    sys.stderr.write(f"tuning bucket={bucket} shim_home={shim_home} aiter_root={args.aiter_root}\n")
    failures = 0
    for target in targets:
        result = tune_target(
            target,
            shim_home=shim_home,
            bucket=bucket,
            aiter_root=args.aiter_root,
            python=args.python,
            retune=args.retune,
            dry_run=args.dry_run,
            hot=args.hot,
            no_flydsl=args.no_flydsl,
        )
        _print_result(result)
        if result.returncode not in (None, 0):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
