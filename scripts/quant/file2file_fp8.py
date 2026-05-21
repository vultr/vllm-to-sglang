"""Quark file2file W8A8 FP8 (per-channel) from the BF16 staging snapshot.

Stage 2 of the recipe, FP8 variant. Reads the bf16 snapshot produced by
`stage_dequant.py`, runs Quark's `direct_quantize_checkpoint` against it, and
emits a per-shard Quark FP8 snapshot under the HF cache. See
`docs/quark-mimo-v2.md` for the full pipeline and the rationale behind the
exclude list and spec flags.

Granularity note: SGLang's `QuarkConfig._is_fp8_w8a8` requires weight qscheme
in {per_tensor, per_channel}. Upstream FineGrainedFP8 uses 128x128 blocks,
which Quark categorizes as `per_group`; SGLang's Quark dispatch does not
accept that. Per-channel FP8 (one scale per output row) is coarser but is the
standard LLM weight-quant scheme and serves coherently without calibration.

Per-channel FP8 on a fused QKV with mixed-magnitude rows truncates the small
values of Q rows to zero. The exclude list keeps all of `self_attn` in BF16
to sidestep that; only MoE expert projections get quantized.
"""

from __future__ import annotations

import os
from pathlib import Path

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    FP8E4M3PerChannelSpec,
    FP8E4M3PerTensorSpec,
    QConfig,
    QLayerConfig,
)

BF16_SNAPSHOT_DIR = (
    "/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-BF16/snapshots/bf16"
)
MODEL_ID_OUT = "XiaomiMiMo/MiMo-V2.5-Pro-QuarkFP8"
SNAPSHOT_OUT = "fp8-perchannel-bf16qkv"

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def build_quant_config() -> QConfig:
    """W8A8 FP8 with per-channel weight scaling and per-tensor dynamic activations.

    Matches SGLang's `QuarkConfig._is_fp8_w8a8` acceptance:
      - weight: fp8_e4m3, qscheme=per_channel, is_dynamic=False
      - input:  fp8_e4m3, qscheme=per_tensor, is_dynamic=True

    `ch_axis=0` on the weight = one scale per output row. The output dim is the
    channel axis for Linear weights stored as (out_features, in_features).
    """
    weight_spec = FP8E4M3PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
    input_spec = FP8E4M3PerTensorSpec(is_dynamic=True).to_quantization_spec()

    return QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec, input_tensors=input_spec),
        exclude=[
            "lm_head",
            "model.embed_tokens",
            # All of self_attn (fused qkv_proj and o_proj) stays BF16. Per-channel
            # FP8 of fused qkv with mixed-magnitude rows truncates small Q values
            # to zero (step size = row_max/256). MoE expert weights have more
            # uniform row distributions and tolerate per-channel FP8.
            "*.self_attn.qkv_proj",
            "*.self_attn.o_proj",
            "*.mlp.gate",
            # `*.mlp.gate_proj` etc. match layer-0 dense MLP only; the MoE-layer
            # projections live at `mlp.experts.N.<...>` and have a longer suffix,
            # so they are not matched and do get quantized.
            "*.mlp.gate_proj",
            "*.mlp.up_proj",
            "*.mlp.down_proj",
            "*.mtp.*",
            "model.mtp.*",
        ],
    )


def output_dir() -> Path:
    sanitized = MODEL_ID_OUT.replace("/", "--")
    return HF_HOME / "hub" / f"models--{sanitized}" / "snapshots" / SNAPSHOT_OUT


def main() -> None:
    target = output_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"source: {BF16_SNAPSHOT_DIR}")
    print(f"output: {target}")

    quantizer = ModelQuantizer(build_quant_config())
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=BF16_SNAPSHOT_DIR,
        save_path=str(target),
        device="cuda",
    )
    print(f"done: {target}")


if __name__ == "__main__":
    main()
