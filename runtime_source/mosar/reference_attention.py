from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compatibility import gemma2_logit_softcap
from .config import MoSARConfig
from .outputs import RoutingState, WorthFieldOutput
from .routers import RoleSpecificPostPositionRouters
from .worth_field import WorthField


@dataclass(slots=True)
class ReferenceAttentionOutput:
    output: Tensor
    probabilities: Tensor
    logits: Tensor
    routing: RoutingState | None
    worth: WorthFieldOutput | None


def _expand_gqa_heads(tensor: Tensor, num_query_heads: int) -> Tensor:
    """Repeat KV heads into their query-head groups for a transparent reference."""

    num_kv_heads = tensor.shape[-2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads.")
    repeats = num_query_heads // num_kv_heads
    return tensor.repeat_interleave(repeats, dim=-2)


class MoSARReferenceAttention(nn.Module):
    """Readable dense attention for correctness tests.

    Inputs are Q/K after the backbone positional transform, and V in the same
    canonical `[batch, sequence, heads, head_dim]` layout. This module never
    computes RoPE itself; the adapter owns the backbone-specific positional
    transformation.
    """

    def __init__(
        self,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        config: MoSARConfig,
        softmax_scale: float | None = None,
        logit_softcap: float | None = None,
    ) -> None:
        super().__init__()
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.config = config
        self.softmax_scale = (1.0 / sqrt(head_dim)) if softmax_scale is None else float(softmax_scale)
        if self.softmax_scale <= 0:
            raise ValueError("softmax_scale must be positive.")
        if logit_softcap is not None and logit_softcap < 0:
            raise ValueError("logit_softcap must be non-negative or None.")
        self.logit_softcap = logit_softcap
        self.routers = RoleSpecificPostPositionRouters(
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            config=config,
        )
        self.worth_field = WorthField(config)

    def forward(
        self,
        query_post_position: Tensor,
        key_post_position: Tensor,
        value: Tensor,
        *,
        use_mosar: bool = True,
        force_global: bool = False,
        attention_mask: Tensor | None = None,
    ) -> ReferenceAttentionOutput:
        self._validate_shapes(query_post_position, key_post_position, value)
        batch, query_length, _, _ = query_post_position.shape
        key_length = key_post_position.shape[1]

        expanded_key = _expand_gqa_heads(key_post_position, self.num_query_heads)
        expanded_value = _expand_gqa_heads(value, self.num_query_heads)
        logits = torch.einsum(
            "bihd,bjhd->bhij",
            query_post_position.float(),
            expanded_key.float(),
        ) * self.softmax_scale
        logits = gemma2_logit_softcap(logits, self.logit_softcap)

        routing: RoutingState | None = None
        worth_output: WorthFieldOutput | None = None
        if use_mosar:
            routing = self.routers(query_post_position, key_post_position)
            if force_global:
                q_probs = torch.zeros_like(routing.query.probabilities)
                k_probs = torch.zeros_like(routing.key.probabilities)
                q_probs[..., self.config.global_regime_index] = 1.0
                k_probs[..., self.config.global_regime_index] = 1.0
            else:
                q_probs = routing.query.probabilities
                k_probs = routing.key.probabilities
            worth_output = self.worth_field(q_probs, k_probs)
            logits = logits + worth_output.log_bias.float().unsqueeze(1)

        causal = torch.ones(
            query_length,
            key_length,
            device=logits.device,
            dtype=torch.bool,
        ).tril(diagonal=key_length - query_length)
        allowed = causal[None, None, :, :]

        if attention_mask is not None:
            if attention_mask.shape not in {
                (batch, query_length, key_length),
                (batch, 1, query_length, key_length),
            }:
                raise ValueError("attention_mask has an unsupported shape.")
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :]
            allowed = allowed & attention_mask.bool()

        logits = logits.masked_fill(~allowed, float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        output = torch.einsum(
            "bhij,bjhd->bihd",
            probabilities.to(dtype=expanded_value.dtype),
            expanded_value,
        )
        return ReferenceAttentionOutput(
            output=output,
            probabilities=probabilities,
            logits=logits,
            routing=routing,
            worth=worth_output,
        )

    def _validate_shapes(self, q: Tensor, k: Tensor, v: Tensor) -> None:
        expected_q = (self.num_query_heads, self.head_dim)
        expected_kv = (self.num_kv_heads, self.head_dim)
        if q.ndim != 4 or q.shape[-2:] != expected_q:
            raise ValueError(f"query must end in {expected_q}; received {tuple(q.shape)}")
        if k.ndim != 4 or k.shape[-2:] != expected_kv:
            raise ValueError(f"key must end in {expected_kv}; received {tuple(k.shape)}")
        if v.ndim != 4 or v.shape[-2:] != expected_kv:
            raise ValueError(f"value must end in {expected_kv}; received {tuple(v.shape)}")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("query, key, and value batch sizes must match.")
        if k.shape[:2] != v.shape[:2]:
            raise ValueError("key and value sequence shapes must match.")
