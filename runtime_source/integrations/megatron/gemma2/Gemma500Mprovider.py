from dataclasses import dataclass

from megatron.bridge.models.gemma.gemma2_provider import Gemma2ModelProvider


@dataclass
class Gemma2ModelProvider500M(Gemma2ModelProvider):
    """~500M parameter Gemma2-style model trained from scratch.

    num_query_groups=1 means only TP=1 is supported (gemma2_layer_spec divides
    num_query_groups by tensor_model_parallel_size). For TP>1, bump it to a
    value divisible by your TP size (e.g. 2 or 4).
    """

    num_layers: int = 18
    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_query_groups: int = 1
    ffn_hidden_size: int = 8192
    query_pre_attn_scalar: int = 256
