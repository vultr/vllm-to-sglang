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
from vllm_shim.aiter.shape_store import padded_m_gl0
from vllm_shim.cli.rocm_probe import bucket as bucket_for_gpu
from vllm_shim.cli.rocm_probe import probe as probe_rocm

# Column containing AITER's M (batch * seq) dimension. Pre-bucketing
# capture writes raw values that have to be rounded up to AITER's
# tuning grid before the tuner can do anything useful with them.
_M_COLUMN: str = "M"

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


def canonicalize_shape_files(files: list[Path]) -> int:
    """Rewrite shape CSVs in place to the canonical persisted form.

    Pads raw M values onto AITER's tuning grid. Pre-bucketing capture
    stored the raw M from the log line; AITER's runtime lookup pads M
    into discrete buckets before checking the tuned config, so off-grid
    raw values are dead weight for the tuner. After padding, rows that
    collapse to identical tuples are deduplicated so the file shrinks
    toward its canonical minimum. Returns the count of files actually
    rewritten; idempotent on already-clean files.

    Writes go through a sibling ``.tmp`` file plus an atomic
    ``Path.replace`` so a crash mid-rewrite never leaves a half-
    written CSV in place. Safe to run at startup-tune time because the
    backend hasn't launched yet, so no capture process is appending
    to these files concurrently.
    """
    rewritten = 0
    for path in files:
        if _canonicalize_one(path):
            rewritten += 1
    return rewritten


def _canonicalize_one(path: Path) -> bool:
    """Rewrite ``path`` if any row needs M padding or dedup.

    Returns True when the file was rewritten, False when every row is
    already canonical (or the file is empty / malformed in a way that
    doesn't include the M column).
    """
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
        try:
            m_col = header.index(_M_COLUMN)
        except ValueError:
            return False
        rows = list(reader)
    cleaned: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    changed = False
    for row in rows:
        new_row = list(row)
        if m_col < len(new_row):
            try:
                m_raw = int(new_row[m_col])
            except ValueError:
                m_raw = None
            if m_raw is not None:
                padded = padded_m_gl0(m_raw)
                if padded != m_raw:
                    new_row[m_col] = str(padded)
                    changed = True
        key = tuple(new_row)
        if key in seen:
            # Two raw rows collapsed onto the same padded bucket
            # (or were already identical). Drop the duplicate;
            # the rewrite below shrinks the file accordingly.
            changed = True
            continue
        seen.add(key)
        cleaned.append(new_row)
    if not changed:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned)
    tmp.replace(path)
    return True


def apply_hot_filter(path: Path, top_k: int) -> tuple[int, int]:
    """Truncate ``path`` to the top-k heuristically-hottest shapes.

    Hot = smallest ``(M, N, K)`` first. In autoregressive LLM serving
    the decode phase runs at ``M = active_batch`` (small, typically
    1-256) and calls every layer's GEMMs once per token; prefill
    shapes are large M but run once per request. Call count therefore
    skews orders of magnitude toward small-M shapes, so sorting M
    ascending and truncating keeps the long tail that dominates
    real-world performance at a fraction of the tuning cost.

    Returns ``(rows_before, rows_after)``. When ``rows_before <=
    top_k`` the file is left untouched. Idempotent and safe to run on
    files lacking the schema columns (no-op + same-length return).

    The heuristic is a stand-in for measured call frequency. Workloads
    dominated by long-context single-shot prefill (rather than
    chat-style decode) should set ``top_k`` wide enough to cover
    their shapes, or skip the filter entirely.
    """
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return (0, 0)
        rows = list(reader)
    rows_before = len(rows)
    if rows_before <= top_k:
        return (rows_before, rows_before)
    try:
        m_col = header.index(_M_COLUMN)
        n_col = header.index("N")
        k_col = header.index("K")
    except ValueError:
        # File lacks the canonical schema (corrupt or pre-shim);
        # leave it alone rather than reorder rows of unknown meaning.
        return (rows_before, rows_before)

    def sort_key(row: list[str]) -> tuple[int, int, int]:
        try:
            return (int(row[m_col]), int(row[n_col]), int(row[k_col]))
        except (ValueError, IndexError):
            # Malformed row sorts to the end so the well-formed shapes
            # at the top of the order are the ones that actually get
            # tuned. Stable sort within the bucket preserves capture
            # order for any operator-side debugging.
            return (sys.maxsize, sys.maxsize, sys.maxsize)

    rows.sort(key=sort_key)
    rows = rows[:top_k]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)
    return (rows_before, top_k)


def merge_shape_files(files: list[Path], destination: Path) -> int:
    """Concatenate shape CSVs into ``destination`` with row-level dedup.

    Every captured shape file shares the schema defined in
    ``shape_store._HEADER``. We dedup on the full row tuple so two
    captures with different ``bias`` or ``dtype`` for the same
    ``(M, N, K)`` both survive - the tuner treats them as separate
    inputs. Returns the row count written (excluding the header).

    Callers running this against operator volumes should invoke
    ``canonicalize_shape_files`` first to pad pre-bucketing M values
    onto AITER's grid. This function assumes its inputs are already
    canonical.
    """
    seen: set[tuple[str, ...]] = set()
    header: list[str] | None = None
    rows: list[list[str]] = []
    for path in files:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            try:
                file_header = next(reader)
            except StopIteration:
                continue
            if header is None:
                header = file_header
            elif file_header != header:
                # Different shape captures should all use shape_store's
                # canonical header; if they don't, something upstream
                # is wrong and the tuner output would be garbage.
                raise ValueError(
                    f"header mismatch in {path}: expected {header}, got {file_header}"
                )
            for row in reader:
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    if header is None:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


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
        _NO_FLYDSL_EXTRA_ARGS.get(spec.target, spec.extra_args)
        if no_flydsl
        else spec.extra_args
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
    # Pad pre-bucketing M values in place before feeding the merge
    # step. After this returns, every source CSV is on AITER's grid
    # and downstream readers (this run plus any future operator-driven
    # invocations) get canonical data. Idempotent on already-clean files.
    canonicalize_shape_files(files)
    rows = merge_shape_files(files, untuned_file)
    rows_total = rows
    if hot is not None and rows > 0:
        rows_total, rows = apply_hot_filter(untuned_file, hot)
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
        sys.stdout.write(
            f"  {target}: {len(files)} shape file(s), {tuned_marker}\n"
        )
    return 0


def _print_result(result: TuneResult) -> None:
    if result.skipped_reason:
        sys.stderr.write(
            f"  {result.target}: skipped ({result.skipped_reason})\n"
        )
        if result.command:
            sys.stderr.write(f"    would run: {' '.join(result.command)}\n")
        return
    status = "ok" if result.returncode == 0 else f"failed (rc={result.returncode})"
    if result.rows_in < result.rows_total:
        shapes = f"{result.rows_in} of {result.rows_total} shapes (hot)"
    else:
        shapes = f"{result.rows_in} shapes"
    sys.stderr.write(
        f"  {result.target}: {shapes} -> {result.tuned_file} [{status}]\n"
    )


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
        help=(
            "Hardware bucket (e.g. gfx942-304cu). Inferred from "
            "rocminfo when omitted."
        ),
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
        help=(
            "Python interpreter that has AITER installed "
            f"(default: {DEFAULT_PYTHON})."
        ),
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
            "Tune only the N hottest shapes per target. 'Hot' is "
            "currently a heuristic: smallest (M, N, K) first, which "
            "approximates the decode-phase GEMMs that dominate call "
            "count in autoregressive serving. Default: tune every "
            "captured shape. Composes with --retune; with the default "
            "(incremental) mode AITER skips already-tuned rows, so "
            "growing --hot across runs converges on the full set."
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
    sys.stderr.write(
        f"tuning bucket={bucket} shim_home={shim_home} aiter_root={args.aiter_root}\n"
    )
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
