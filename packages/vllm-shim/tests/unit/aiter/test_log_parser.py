"""Tests for the AITER shape-not-found log line parser."""

from vllm_shim.aiter.log_parser import AiterShape, parse_line

CANONICAL = (
    "[aiter] shape is M:1024, N:7168, K:512 "
    "dtype=torch.bfloat16 otype=torch.bfloat16 "
    "bias=False scaleAB=False bpreshuffle=False, "
    "not found tuned config in /tmp/aiter_configs/bf16_gemm.csv, "
    "will use default config!"
)


def test_parses_canonical_line() -> None:
    shape = parse_line(CANONICAL)
    assert shape == AiterShape(
        m=1024,
        n=7168,
        k=512,
        dtype="torch.bfloat16",
        otype="torch.bfloat16",
        bias=False,
        scale_ab=False,
        bpreshuffle=False,
        target="bf16_gemm",
    )


def test_target_is_csv_stem_not_full_path() -> None:
    # The directory is hardcoded by AITER and unrelated to where we
    # want to *write* captured shapes; only the basename varies.
    shape = parse_line(CANONICAL)
    assert shape is not None
    assert shape.target == "bf16_gemm"
    assert "/" not in shape.target
    assert not shape.target.endswith(".csv")


def test_returns_none_on_empty_line() -> None:
    assert parse_line("") is None


def test_returns_none_on_unrelated_line() -> None:
    assert parse_line("INFO 11-05 12:00:00 sglang ready on :8001\n") is None


def test_returns_none_on_other_aiter_lines() -> None:
    # Only the shape-not-found variant should match; anything else
    # starting with [aiter] (e.g. init logs) must be ignored.
    assert parse_line("[aiter] initializing rocm runtime\n") is None


def test_returns_none_when_anchor_matches_but_fields_missing() -> None:
    # Truncated/corrupt aiter line: anchor present but the shape
    # fields are gone. The parser should refuse rather than emit
    # a partial shape with zero-defaults.
    bad = (
        "[aiter] shape is "
        "not found tuned config in /tmp/aiter_configs/bf16_gemm.csv, ..."
    )
    assert parse_line(bad) is None


def test_parses_true_booleans() -> None:
    line = CANONICAL.replace("bias=False", "bias=True").replace(
        "bpreshuffle=False", "bpreshuffle=True"
    )
    shape = parse_line(line)
    assert shape is not None
    assert shape.bias is True
    assert shape.bpreshuffle is True
    assert shape.scale_ab is False  # untouched


def test_parses_fp8_dtype() -> None:
    line = CANONICAL.replace("torch.bfloat16", "torch.float8_e4m3fn", 1)
    shape = parse_line(line)
    assert shape is not None
    assert shape.dtype == "torch.float8_e4m3fn"
    # otype was the second occurrence in the original line — still bfloat16
    assert shape.otype == "torch.bfloat16"


def test_parses_different_csv_target() -> None:
    line = CANONICAL.replace("bf16_gemm.csv", "fp8_blockscale_gemm.csv")
    shape = parse_line(line)
    assert shape is not None
    assert shape.target == "fp8_blockscale_gemm"


def test_parser_tolerates_field_reordering() -> None:
    # If a future AITER reorders fields, we should still recognize them.
    reordered = (
        "[aiter] shape is K:512, M:1024, N:7168 "
        "scaleAB=False bpreshuffle=False bias=False "
        "otype=torch.bfloat16 dtype=torch.bfloat16, "
        "not found tuned config in /tmp/aiter_configs/bf16_gemm.csv, "
        "will use default config!"
    )
    shape = parse_line(reordered)
    assert shape is not None
    assert (shape.m, shape.n, shape.k) == (1024, 7168, 512)


def test_parses_line_with_trailing_newline() -> None:
    # Stream-tee feeds us lines with trailing \n; the parser must
    # match anyway.
    shape = parse_line(CANONICAL + "\n")
    assert shape is not None
    assert shape.target == "bf16_gemm"
