"""Quark file2file MXFP4 with FP8 attention (the Kimi-K2.5-AttnFP8 template).

Stage 2 of the recipe, the "AttnFP8" variant. Mirrors AMD's official recipe
for `amd/Kimi-K2.5-MXFP4-AttnFP8`:

    cd Quark/examples/torch/language_modeling/llm_ptq/
    python3 quantize_quark.py \\
      --model_dir <bf16> \\
      --quant_scheme mxfp4 \\
      --layer_quant_scheme '*self_attn*' ptpc_fp8 \\
      --exclude_layers <list> \\
      --file2file_quantization

Why this instead of `file2file_mxfp4.py` (the BF16-attn baseline):

  The non-overridden recipe keeps `*.self_attn.*` BF16 because static
  per-group MXFP4 truncates v_proj's small rows to zero (v_proj collapse).
  Kimi K2.5 hits the same problem and solves it by quantizing attention to
  Per-Token-Per-Channel FP8 (PTPC FP8) instead of BF16 or MXFP4. PTPC FP8
  has per-channel weight scales and per-token activation scales, both
  finer-grained than MXFP4's per-32-element groups, so small rows survive.

Spec equivalences (matches `quark.torch.quantization.config.template.PTPCFP8Scheme`):

  weight:  FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0)
  input:   FP8E4M3PerChannelSpec(is_dynamic=True,  ch_axis=1)

This is RTN, not AWQ. No calibration data. No model load. No 8-GPU
dispatch. `direct_quantize_checkpoint` walks the BF16 safetensors
shard-by-shard, peak RAM ~5-10 GB. The previous AWQ attempts at this
problem were wrong: AWQ + MXFP4 isn't a tested Quark path for huge MoE.
"""

from __future__ import annotations

import os
from pathlib import Path

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    FP8E4M3PerChannelSpec,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
)

BF16_SNAPSHOT_DIR = (
    "/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-BF16/snapshots/bf16"
)
MODEL_ID_OUT = "XiaomiMiMo/MiMo-V2.5-Pro-MXFP4"
SNAPSHOT_OUT = "mxfp4-attnfp8-mtp"

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def build_quant_config() -> QConfig:
    """MXFP4 globally + PTPC FP8 override for self_attn."""
    # Global MXFP4 spec (matches `file2file_mxfp4.py`, accepted by SGLang's
    # `QuarkConfig._is_mx_fp4`: weight static, input dynamic, even scales).
    mxfp4_weight = OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=False, scale_calculation_mode="even"
    ).to_quantization_spec()
    mxfp4_input = OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=True, scale_calculation_mode="even"
    ).to_quantization_spec()

    # PTPC FP8 spec for attention (per-channel weights, per-token activations).
    # SGLang's Quark loader accepts FP8E4M3 with per-channel weight scales and
    # dynamic per-token activation scales via the same `Quark` quantization
    # config dispatch as `amd/Kimi-K2.5-MXFP4-AttnFP8` ships.
    fp8_weight = FP8E4M3PerChannelSpec(
        is_dynamic=False, ch_axis=0
    ).to_quantization_spec()
    fp8_input = FP8E4M3PerChannelSpec(
        is_dynamic=True, ch_axis=1
    ).to_quantization_spec()

    return QConfig(
        global_quant_config=QLayerConfig(weight=mxfp4_weight, input_tensors=mxfp4_input),
        # `layer_quant_config` keys are glob patterns matched against module
        # FQNs at quant time. Patterns ending in `*` match the whole subtree;
        # the FQN naming convention in MiMo is `model.layers.N.self_attn.X`
        # for `X` in {qkv_proj, o_proj}.
        layer_quant_config={
            "*self_attn*": QLayerConfig(weight=fp8_weight, input_tensors=fp8_input),
        },
        exclude=[
            "lm_head",
            "model.embed_tokens",
            # Router gate stays BF16 across all Quark recipes (small, not a
            # Linear matmul that benefits from quant).
            "*.mlp.gate",
            # Dense MLP for layer 0 (first_k_dense_replace=1). MiMo's MoE-layer
            # projections live at `mlp.experts.N.<...>` and have a longer
            # suffix so they are not matched and DO get quantized. Excluding
            # these matches AMD's Kimi MXFP4 recipe — without this the dense
            # MLP of layer 0 gets MXFP4 too but accuracy regressions in early
            # layers are more visible.
            "*.mlp.gate_proj",
            "*.mlp.up_proj",
            "*.mlp.down_proj",
            # Embedding inside the MTP block must stay BF16 (lookup table, not
            # a Linear matmul).
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
    print("scheme: MXFP4 (MoE experts) + PTPC FP8 (self_attn) + BF16 (excludes)")

    quantizer = ModelQuantizer(build_quant_config())
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=BF16_SNAPSHOT_DIR,
        save_path=str(target),
        device="cuda",
    )
    print(f"done: {target}")


if __name__ == "__main__":
    main()
