"""Tests for the AITER shape store CSV writer."""

import csv
from pathlib import Path

from vllm_shim.aiter.log_parser import AiterShape
from vllm_shim.aiter.shape_store import ShapeStore


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
