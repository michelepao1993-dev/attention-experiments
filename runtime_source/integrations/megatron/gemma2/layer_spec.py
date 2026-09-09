from __future__ import annotations

from typing import TYPE_CHECKING

from .core_attention import MoSARGemma2DotProductAttention

try:  # pragma: no cover - Megatron-only adapter.
    from megatron.bridge.models.gemma.gemma2_provider import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinearLayerNorm,
    )
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSubmodules,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if TYPE_CHECKING:
    from megatron.bridge.models.gpt_provider import GPTModelProvider


def mosar_gemma2_layer_spec(config: "GPTModelProvider") -> "ModuleSpec":
    """Gemma 2 layer spec differing only in the core-attention class."""

    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise RuntimeError("The pinned Megatron environment is required.") from _IMPORT_ERROR

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=MoSARGemma2DotProductAttention,
                    linear_proj=TERowParallelLinearLayerNorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
