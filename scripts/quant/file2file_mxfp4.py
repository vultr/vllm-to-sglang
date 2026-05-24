"""Quark file2file MXFP4 (W4A4) from the BF16 staging snapshot.

Stage 2 of the recipe, MXFP4 variant. Reads the bf16 snapshot produced by
`stage_dequant.py`, runs Quark's `direct_quantize_checkpoint` against it, and
emits a per-shard MXFP4 snapshot. See `docs/quark-mimo-v2.md` for the full
pipeline.

Spec flags are pinned to the only combination SGLang's `QuarkConfig._is_mx_fp4`
accepts: weight `is_dynamic=False`, input `is_dynamic=True`, both with
`scale_calculation_mode="even"` (matches AMD's published Kimi MXFP4 recipe).
Flipping either flag yields a checkpoint that silently fails to load.

Exclude list mirrors AMD's Kimi recipe with MiMo's namespace; net effect is
that only `mlp.experts.N.{gate,up,down}_proj` weights become MXFP4, with
attention, router, MTP head, and embeddings staying BF16.
"""

from __future__ import annotations

import os
from pathlib import Path

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
)

BF16_SNAPSHOT_DIR = (
    "/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-BF16/snapshots/bf16"
)
MODEL_ID_OUT = "XiaomiMiMo/MiMo-V2.5-Pro-MXFP4"
SNAPSHOT_OUT = "mxfp4-bf16qkv-mtp"

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def build_quant_config() -> QConfig:
    """W4A4 MXFP4 spec matching SGLang's `QuarkConfig._is_mx_fp4` acceptance.

    Weight `is_dynamic=False` is what causes Quark to compute and persist
    static E8M0 scales at quant time. With `is_dynamic=True` the resulting
    checkpoint has no `weight_scale` tensor and SGLang's `_is_mx_fp4` returns
    False, falling the layer through to `UnquantizedLinearMethod` (which
    itself fails on the uint8-packed weight).
    """
    weight_spec = OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=False, scale_calculation_mode="even"
    ).to_quantization_spec()
    input_spec = OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=True, scale_calculation_mode="even"
    ).to_quantization_spec()

    return QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec, input_tensors=input_spec),
        exclude=[
            "lm_head",
            "model.embed_tokens",
            # All of self_attn (fused qkv_proj and o_proj) stays BF16. The
            # fused qkv path is sensitive to per-channel-style truncation; the
            # safe choice for both FP8 and MXFP4 variants is to leave it
            # unquantized and rely on the corrected BF16 staging.
            "*.self_attn.*",
            "*.mlp.gate",
            # `*.mlp.gate_proj` etc. match layer-0 dense MLP only; the MoE-layer
            # projections live at `mlp.experts.N.<...>` and have a longer suffix,
            # so they are not matched and do get quantized.
            "*.mlp.gate_proj",
            "*.mlp.up_proj",
            "*.mlp.down_proj",
            # Embedding inside the MTP block must stay BF16; embeddings can't
            # be MXFP4 (they're lookup tables, not Linear matmuls).
            "model.mtp.embed_tokens",
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
