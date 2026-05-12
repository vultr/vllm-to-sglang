"""Tests for the AITER shape store CSV writer."""

import csv
from pathlib import Path

import pytest
from vllm_shim.aiter.log_parser import AiterShape
from vllm_shim.aiter.shape_store import ShapeStore, padded_m_gl0


def _shape(**overrides: object) -> AiterShape:
    base: dict[str, object] = {
        "m": 1024,
        "n": 7168,
        "k": 512,
        "dtype": "torch.bfloat16",
        "outdtype": "torch.bfloat16",
        "bias": False,
        "scale_ab": False,
        "bpreshuffle": False,
        "target": "bf16_tuned_gemm",
    }
    base.update(overrides)
    return AiterShape(**base)  # type: ignore[arg-type]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_first_add_creates_csv_with_header(tmp_path: Path) -> None:
    store = ShapeStore(tmp_path)
    assert store.add(_shape()) is True
    csv_path = tmp_path / "bf16_tuned_gemm.csv"
    assert csv_path.exists()
    rows = _read_rows(csv_path)
    # Column order and naming match AITER's own ``save_shapes`` writer
    # so this CSV is a drop-in ``--untune_file`` for the tuner.
    assert rows == [
        {
            "M": "1024",
            "N": "7168",
            "K": "512",
            "bias": "False",
            "dtype": "torch.bfloat16",
            "outdtype": "torch.bfloat16",
            "scaleAB": "False",
            "bpreshuffle": "False",
        }
    ]


def test_duplicate_add_is_noop(tmp_path: Path) -> None:
    store = ShapeStore(tmp_path)
    assert store.add(_shape()) is True
    assert store.add(_shape()) is False
    assert len(_read_rows(tmp_path / "bf16_tuned_gemm.csv")) == 1


def test_distinct_shapes_both_persist(tmp_path: Path) -> None:
    store = ShapeStore(tmp_path)
    assert store.add(_shape(m=1024)) is True
    assert store.add(_shape(m=2048)) is True
    rows = _read_rows(tmp_path / "bf16_tuned_gemm.csv")
    assert {r["M"] for r in rows} == {"1024", "2048"}


def test_different_targets_go_to_different_files(tmp_path: Path) -> None:
    store = ShapeStore(tmp_path)
    store.add(_shape(target="bf16_tuned_gemm"))
    store.add(_shape(target="a8w8_blockscale_tuned_gemm"))
    assert (tmp_path / "bf16_tuned_gemm.csv").exists()
    assert (tmp_path / "a8w8_blockscale_tuned_gemm.csv").exists()


def test_dedup_seeded_from_existing_file(tmp_path: Path) -> None:
    # First process writes a shape, then exits.
    ShapeStore(tmp_path).add(_shape())
    # Fresh process: should see the existing row and refuse to duplicate.
    new_store = ShapeStore(tmp_path)
    assert new_store.add(_shape()) is False
    # But an unseen shape should still go through.
    assert new_store.add(_shape(k=1024)) is True
    rows = _read_rows(tmp_path / "bf16_tuned_gemm.csv")
    assert len(rows) == 2


def test_dedup_keys_include_every_shape_field(tmp_path: Path) -> None:
    # Each field flip should be treated as a different shape.
    store = ShapeStore(tmp_path)
    base = _shape()
    variants = [
        base,
        _shape(m=2048),
        _shape(n=14336),
        _shape(k=1024),
        _shape(dtype="torch.float16"),
        _shape(outdtype="torch.float16"),
        _shape(bias=True),
        _shape(scale_ab=True),
        _shape(bpreshuffle=True),
    ]
    for v in variants:
        assert store.add(v) is True
    assert len(_read_rows(tmp_path / "bf16_tuned_gemm.csv")) == len(variants)


def test_creates_intermediate_directories(tmp_path: Path) -> None:
    deep = tmp_path / "gfx942-304cu" / "moonshotai--Kimi-K2.6" / "tp8-ep8"
    store = ShapeStore(deep)
    store.add(_shape())
    assert (deep / "bf16_tuned_gemm.csv").exists()


def test_header_written_only_once(tmp_path: Path) -> None:
    store = ShapeStore(tmp_path)
    store.add(_shape(m=1))
    store.add(_shape(m=2))
    store.add(_shape(m=3))
    with (tmp_path / "bf16_tuned_gemm.csv").open() as f:
        header_count = sum(1 for line in f if line.startswith("M,"))
    assert header_count == 1


# ---------- padded_m_gl0 ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # M <= 256: ceil to multiple of 16
        (1, 16),
        (15, 16),
        (16, 16),       # already on grid
        (17, 32),
        (256, 256),     # boundary, stays on grid
        # 256 < M <= 1024: ceil to multiple of 32
        (257, 288),
        (1024, 1024),
        # 1024 < M <= 4096: ceil to multiple of 64
        (1025, 1088),
        (4096, 4096),
        # M > 4096: ceil to multiple of 128
        (4097, 4224),
        (16384, 16384),
        (16385, 16512),
    ],
)
def test_padded_m_gl0_grids(raw: int, expected: int) -> None:
    # Mirrors AITER's getPaddedM(M, N, K, gl=0) from
    # repos/aiter/csrc/py_itfs_cu/gemm_common.cu. Boundary values
    # (16, 256, 1024, 4096) must stay on the grid; off-grid values
    # round UP to the next bucket.
    assert padded_m_gl0(raw) == expected


def test_padded_m_gl0_handles_nonpositive() -> None:
    # AITER never logs M<=0 in practice, but the helper must not
    # blow up if a corrupt CSV row feeds in a stray value: the
    # canonicalize pass calls this on every existing row.
    assert padded_m_gl0(0) == 0
    assert padded_m_gl0(-5) == -5


def test_add_collapses_raw_m_onto_grid_bucket(tmp_path: Path) -> None:
    # Two captures of the same shape with different raw M values
    # that fall onto the same AITER lookup bucket should land as
    # one row. Without bucketing, continuous batching would write
    # thousands of essentially equivalent rows.
    store = ShapeStore(tmp_path)
    assert store.add(_shape(m=257)) is True
    # 258 also pads to 288. Dedup hits.
    assert store.add(_shape(m=258)) is False
    rows = _read_rows(tmp_path / "bf16_tuned_gemm.csv")
    assert len(rows) == 1
    # CSV stores the padded form, not the raw value.
    assert rows[0]["M"] == "288"


def test_add_writes_padded_m_when_raw_is_off_grid(tmp_path: Path) -> None:
    # Even a single off-grid raw value is persisted as its bucket
    # value so future dedup against this row uses the canonical key.
    store = ShapeStore(tmp_path)
    store.add(_shape(m=4097))
    rows = _read_rows(tmp_path / "bf16_tuned_gemm.csv")
    assert rows[0]["M"] == "4224"  # 4097 rounds up to next 128 = 4224


def test_dedup_against_pre_bucketing_raw_row(tmp_path: Path) -> None:
    # Existing CSV has a raw M=257 row from before bucketing was
    # added. A new capture of M=258 must dedup against it (both pad
    # to 288). Without this, we'd accumulate two rows for what AITER
    # sees as a single bucket.
    path = tmp_path / "bf16_tuned_gemm.csv"
    path.write_text(
        "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n"
        "257,7168,512,False,torch.bfloat16,torch.bfloat16,False,False\n"
    )
    store = ShapeStore(tmp_path)
    assert store.add(_shape(m=258)) is False


def test_corrupt_existing_row_does_not_block_writes(tmp_path: Path) -> None:
    # Pre-create the CSV with a header + one valid row + one garbage row.
    path = tmp_path / "bf16_tuned_gemm.csv"
    path.write_text(
        "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n"
        "1024,7168,512,False,torch.bfloat16,torch.bfloat16,False,False\n"
        "garbage,row,with,no,numbers,a,b,c\n"
    )
    store = ShapeStore(tmp_path)
    # The valid row should still be detected as a duplicate.
    assert store.add(_shape()) is False
    # And a fresh shape should still write.
    assert store.add(_shape(m=4096)) is True
