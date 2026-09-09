from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
import math
from typing import Any, Literal, Optional

import torch
from torch import Tensor

from mosar.config import MoSARConfig
from mosar.losses import normalized_reach_cost
from mosar.outputs import RouterOutput, RoutingState
from mosar.routers import RoleSpecificPostPositionRouters
from mosar.worth_field import WorthField

from .auxiliary import MoSARLayerAuxiliary
from .diagnostics import MoSARLayerDiagnostics, build_layer_diagnostics

MoSARMode = Literal["vanilla_copy", "forced_short", "forced_medium", "forced_global", "uniform", "learned"]

_MEGATRON_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - exercised only in the pinned Megatron environment.
    _gemma = import_module("megatron.bridge.models.gemma.gemma2_provider")
    Gemma2DotProductAttention = _gemma.Gemma2DotProductAttention
    logit_softcapping = _gemma.logit_softcapping
    get_swa = _gemma.get_swa

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType
except Exception as exc:  # pragma: no cover - keeps the framework-free package importable.
    _MEGATRON_IMPORT_ERROR = exc

    class Gemma2DotProductAttention(torch.nn.Module):  # type: ignore[no-redef]
        pass

    PackedSeqParams = Any  # type: ignore[assignment,misc]
    AttnMaskType = Any  # type: ignore[assignment,misc]
    parallel_state = None  # type: ignore[assignment]
    tensor_parallel = None  # type: ignore[assignment]
    logit_softcapping = None  # type: ignore[assignment]
    get_swa = None  # type: ignore[assignment]


def _apply_gemma_sliding_window_mask(
    attention_mask: Tensor,
    window_size: Any,
    *,
    query_length: int,
    key_length: int,
) -> Tensor:
    """Reproduce the pinned Gemma 2 sliding-window mask composition.

    The pinned Bridge helper exposes ``get_swa(seq_q, seq_kv, window_size)``
    and returns the additional mask for positions outside the local window.
    Gemma combines it with the already-present causal/padding mask via logical
    OR.  A small compatibility branch is retained for older two-argument
    helpers that directly accept the existing mask.
    """

    signature = inspect.signature(get_swa)
    names = list(signature.parameters)

    if names == ["seq_q", "seq_kv", "window_size"]:
        swa_mask = get_swa(query_length, key_length, window_size)
        swa_mask = swa_mask.to(device=attention_mask.device, dtype=torch.bool)
        return torch.logical_or(attention_mask.to(dtype=torch.bool), swa_mask)

    if len(names) == 2:
        return get_swa(attention_mask, window_size)

    values = {
        "attention_mask": attention_mask,
        "mask": attention_mask,
        "window_size": window_size,
        "query_length": query_length,
        "q_len": query_length,
        "sq": query_length,
        "seq_q": query_length,
        "key_length": key_length,
        "k_len": key_length,
        "sk": key_length,
        "seq_kv": key_length,
    }
    kwargs = {name: values[name] for name in names if name in values}
    if len(kwargs) == len(names):
        generated_mask = get_swa(**kwargs)
        if "attention_mask" in names or "mask" in names:
            return generated_mask
        generated_mask = generated_mask.to(
            device=attention_mask.device,
            dtype=torch.bool,
        )
        return torch.logical_or(attention_mask.to(dtype=torch.bool), generated_mask)

    raise RuntimeError(
        f"Unsupported get_swa signature {signature}. Inspect the pinned "
        "gemma2_provider.py before enabling even-layer tests."
    )


def _inverse_rope_suffix(
    tensor: Tensor,
    *,
    keep_rope_percentage: float,
    rotary_base: float = 10000.0,
) -> Tensor:
    """Undo RoPE on the suffix of split-half rotary pairs.

    The input arrives post-RoPE from Megatron SelfAttention. With
    keep_rope_percentage=0.0, this returns NoPE. With 0<p<1, it keeps the first
    p fraction of RoPE pairs and removes RoPE from the remaining pairs.
    """

    if keep_rope_percentage >= 1.0:
        return tensor

    head_dim = tensor.size(-1)
    if head_dim % 2 != 0:
        raise RuntimeError(f"RoPE inverse requires even head_dim, got {head_dim}.")

    half_dim = head_dim // 2
    keep_pairs = int(keep_rope_percentage * half_dim)

    if keep_pairs >= half_dim:
        return tensor
    if keep_pairs < 0:
        raise RuntimeError(f"Invalid keep_pairs={keep_pairs}.")

    seq_len = tensor.size(0)
    device = tensor.device

    # Megatron split-half RoPE uses inv_freq = 1/base^(arange(0, D, 2)/D).
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        float(rotary_base)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)[:, None, None, :]  # [S, 1, 1, D/2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    first, second = torch.chunk(tensor, 2, dim=-1)

    # Work in fp32 for the trigonometric inverse, then cast back.
    first_f = first.float()
    second_f = second.float()

    out_first = first_f.clone()
    out_second = second_f.clone()

    # Forward split-half RoPE:
    # y1 = x1*cos - x2*sin
    # y2 = x2*cos + x1*sin
    #
    # Inverse:
    # x1 = y1*cos + y2*sin
    # x2 = y2*cos - y1*sin
    c = cos[..., keep_pairs:]
    s = sin[..., keep_pairs:]

    y1 = first_f[..., keep_pairs:]
    y2 = second_f[..., keep_pairs:]

    out_first[..., keep_pairs:] = y1 * c + y2 * s
    out_second[..., keep_pairs:] = y2 * c - y1 * s

    return torch.cat((out_first, out_second), dim=-1).to(dtype=tensor.dtype)


def _local_causal_hard_mask_bias(
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return additive -inf bias outside a causal local window.

    Keeps keys j such that 0 <= i-j < window_size.
    Assumes standard left-to-right causal semantics.
    """

    q_pos = torch.arange(key_length - query_length, key_length, device=device)
    k_pos = torch.arange(0, key_length, device=device)

    distance = q_pos[:, None] - k_pos[None, :]
    outside = (distance < 0) | (distance >= int(window_size))

    bias = torch.zeros((query_length, key_length), device=device, dtype=dtype)
    bias = bias.masked_fill(outside, torch.finfo(dtype).min)

    return bias[None, None, :, :].expand(batch_size, num_heads, -1, -1)


def _alibi_slopes(num_heads: int, *, device: torch.device) -> Tensor:
    """Press et al. ALiBi slopes for power-of-two heads.

    For this 500M pilot num_heads=8, so the simple closed form is sufficient:
    slope_h = 2^(-8(h+1)/n).
    """

    if num_heads <= 0:
        raise RuntimeError(f"num_heads must be positive, got {num_heads}.")
    heads = torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    return torch.pow(torch.tensor(2.0, device=device), -8.0 * heads / float(num_heads))


def _alibi_bias(
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return ALiBi bias with shape [B, H, Sq, Sk]."""

    slopes = _alibi_slopes(num_heads, device=device)

    # General enough for Sq <= Sk. In ordinary training Sq == Sk.
    q_pos = torch.arange(key_length - query_length, key_length, device=device, dtype=torch.float32)
    k_pos = torch.arange(0, key_length, device=device, dtype=torch.float32)

    distance = (q_pos[:, None] - k_pos[None, :]).clamp_min(0.0)  # [Sq, Sk]
    bias = -slopes[None, :, None, None] * distance[None, None, :, :]
    return bias.expand(batch_size, -1, -1, -1).to(dtype=dtype)


@dataclass(frozen=True, slots=True)
class _ResolvedMoSARSettings:
    mode: MoSARMode
    config: MoSARConfig


def _config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _resolve_settings(config: Any) -> _ResolvedMoSARSettings:
    mode = _config_value(config, "mosar_mode", "learned")
    valid_modes = {"vanilla_copy", "forced_short", "forced_medium", "forced_global", "uniform", "learned"}
    if mode not in valid_modes:
        raise ValueError(f"unsupported mosar_mode={mode!r}; expected one of {sorted(valid_modes)}")

    sequence_length = int(
        _config_value(config, "mosar_sequence_length", _config_value(config, "seq_length", 2048))
    )
    reaches = tuple(_config_value(config, "mosar_reaches", (128, 512, sequence_length)))
    alphas = tuple(_config_value(config, "mosar_alphas", (0.75, 0.5, 1.0)))
    core_config = MoSARConfig(
        sequence_length=sequence_length,
        regime_names=tuple(_config_value(config, "mosar_regime_names", ("S", "M", "G"))),
        reaches=reaches,
        alphas=alphas,
        router_hidden_size=int(_config_value(config, "mosar_router_hidden_size", 64)),
        temperature=float(_config_value(config, "mosar_temperature", 1.0)),
        bias_floor=float(_config_value(config, "mosar_bias_floor", 6.0)),
        transition_power=float(_config_value(config, "mosar_transition_power", 2.0)),
        epsilon=float(_config_value(config, "mosar_epsilon", 1e-6)),
        input_weight_std=float(_config_value(config, "mosar_input_weight_std", 0.02)),
        output_weight_std=float(_config_value(config, "mosar_output_weight_std", 1e-3)),
    )
    return _ResolvedMoSARSettings(mode=mode, config=core_config)


def _constant_router_output(reference: Tensor, probabilities: Tensor) -> RouterOutput:
    """Build a shape-compatible RouterOutput for deterministic test policies."""

    logits = torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))
    # Deterministic modes do not need hidden features; a zero-width tensor keeps
    # the output contract without allocating a fake router representation.
    features = reference.new_empty((*probabilities.shape[:-1], 0))
    return RouterOutput(features=features, logits=logits, probabilities=probabilities)


class MoSARGemma2DotProductAttention(Gemma2DotProductAttention):
    """Gemma 2 core attention with a backbone-agnostic MoSAR log-worth bias.

    Q and K arrive here after RoPE and before GQA key/value repetition.  The
    Gemma compatibility score remains exactly

        c * tanh((Q K^T / sqrt(query_pre_attn_scalar)) / c).

    MoSAR then adds ``log W`` before Gemma's existing mask + vanilla softmax.
    Version 0 deliberately supports tensor parallel size 1 only; this prevents
    accidental local-head routing from being mistaken for the global router
    specified by the theory.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _MEGATRON_IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(
                "The pinned Megatron Bridge/Core environment is required for "
                "MoSARGemma2DotProductAttention."
            ) from _MEGATRON_IMPORT_ERROR

        super().__init__(*args, **kwargs)
        settings = _resolve_settings(self.config)
        self.mosar_mode: MoSARMode = settings.mode
        self.mosar_routing_inference = str(
            _config_value(self.config, "mosar_routing_inference", "soft")
        )
        if self.mosar_routing_inference not in {"soft", "hard_top1"}:
            raise ValueError(
                f"unsupported mosar_routing_inference={self.mosar_routing_inference!r}; "
                "expected 'soft' or 'hard_top1'."
            )
        self.mosar_position_variant = _config_value(self.config, "mosar_position_variant", "rope")
        valid_position_variants = {"rope", "alibi", "prope", "rope_s_mask", "rope_m_mask"}
        if self.mosar_position_variant not in valid_position_variants:
            raise ValueError(
                f"unsupported mosar_position_variant={self.mosar_position_variant!r}; "
                f"expected one of {sorted(valid_position_variants)}"
            )
        self.mosar_prope_percentage = float(_config_value(self.config, "mosar_prope_percentage", 0.5))
        if not 0.0 <= self.mosar_prope_percentage <= 1.0:
            raise ValueError("mosar_prope_percentage must be in [0, 1].")
        self.mosar_rotary_base = float(_config_value(self.config, "mosar_rotary_base", 10000.0))
        self.mosar_hard_mask_window = int(_config_value(self.config, "mosar_hard_mask_window", 0))
        self.mosar_config = settings.config

        tp_size = int(self.config.tensor_model_parallel_size)
        if tp_size != 1:
            raise NotImplementedError(
                "MoSAR v0 requires tensor-model-parallel size 1.  TP>1 needs an "
                "explicit global-router communication design."
            )

        self.mosar_routers = RoleSpecificPostPositionRouters(
            num_query_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_query_groups_per_partition,
            head_dim=self.hidden_size_per_attention_head,
            config=self.mosar_config,
        )
        # Vanilla uses the identical MoSAR model shell so construction order and
        # backbone initialization remain matched, but router parameters are
        # frozen and excluded from the optimizer.
        if self.mosar_mode == "vanilla_copy":
            self.mosar_routers.requires_grad_(False)
        self.mosar_worth_field = WorthField(self.mosar_config)
        self._pending_mosar_auxiliary: MoSARLayerAuxiliary | None = None
        self.mosar_capture_diagnostics = bool(
            _config_value(self.config, "mosar_capture_diagnostics", False)
        )
        self.mosar_diagnostic_layers = tuple(
            int(layer)
            for layer in _config_value(
                self.config,
                "mosar_diagnostic_layers",
                (1, 2, 9, 18),
            )
        )
        self.mosar_diagnostic_max_tokens = int(
            _config_value(self.config, "mosar_diagnostic_max_tokens", 128)
        )
        if self.mosar_diagnostic_max_tokens <= 0:
            raise ValueError("mosar_diagnostic_max_tokens must be positive.")
        self._latest_mosar_diagnostics: MoSARLayerDiagnostics | None = None

    def peek_mosar_diagnostics(self) -> MoSARLayerDiagnostics | None:
        """Return the latest detached diagnostic snapshot without consuming it."""

        return getattr(self, "_latest_mosar_diagnostics", None)

    def consume_mosar_auxiliary(self) -> MoSARLayerAuxiliary | None:
        state = self._pending_mosar_auxiliary
        self._pending_mosar_auxiliary = None
        return state

    def _route(self, query: Tensor, key: Tensor) -> RoutingState:
        # Megatron layout [S, B, H, D] -> canonical MoSAR [B, S, H, D].
        query_canonical = query.permute(1, 0, 2, 3)
        key_canonical = key.permute(1, 0, 2, 3)

        learned = self.mosar_routers(query_canonical, key_canonical)
        if self.mosar_mode == "learned" and self.mosar_routing_inference == "hard_top1":
            q_probs = torch.zeros_like(learned.query.probabilities)
            k_probs = torch.zeros_like(learned.key.probabilities)

            q_top1 = learned.query.probabilities.argmax(dim=-1, keepdim=True)
            k_top1 = learned.key.probabilities.argmax(dim=-1, keepdim=True)

            q_probs.scatter_(-1, q_top1, 1.0)
            k_probs.scatter_(-1, k_top1, 1.0)

            return RoutingState(
                query=_constant_router_output(query_canonical, q_probs),
                key=_constant_router_output(key_canonical, k_probs),
            )

        if self.mosar_mode == "learned" or self.mosar_mode == "vanilla_copy":
            return learned

        q_shape = learned.query.probabilities.shape
        k_shape = learned.key.probabilities.shape
        if self.mosar_mode == "forced_short":
            q_probs = torch.zeros_like(learned.query.probabilities)
            k_probs = torch.zeros_like(learned.key.probabilities)
            q_probs[..., 0] = 1.0
            k_probs[..., 0] = 1.0
        elif self.mosar_mode == "forced_medium":
            q_probs = torch.zeros_like(learned.query.probabilities)
            k_probs = torch.zeros_like(learned.key.probabilities)
            q_probs[..., 1] = 1.0
            k_probs[..., 1] = 1.0
        elif self.mosar_mode == "forced_global":
            q_probs = torch.zeros_like(learned.query.probabilities)
            k_probs = torch.zeros_like(learned.key.probabilities)
            q_probs[..., self.mosar_config.global_regime_index] = 1.0
            k_probs[..., self.mosar_config.global_regime_index] = 1.0
        elif self.mosar_mode == "uniform":
            q_probs = torch.full_like(
                learned.query.probabilities, 1.0 / self.mosar_config.num_regimes
            )
            k_probs = torch.full_like(
                learned.key.probabilities, 1.0 / self.mosar_config.num_regimes
            )
        else:  # defensive: _resolve_settings already validates this.
            raise AssertionError(f"unreachable mode {self.mosar_mode}")

        assert q_probs.shape == q_shape and k_probs.shape == k_shape
        return RoutingState(
            query=_constant_router_output(query_canonical, q_probs),
            key=_constant_router_output(key_canonical, k_probs),
        )

    def _mosar_log_bias(self, query: Tensor, key: Tensor) -> Tensor | None:
        if self.mosar_mode == "vanilla_copy":
            self._pending_mosar_auxiliary = None
            return None

        routing = self._route(query, key)
        worth = self.mosar_worth_field(
            routing.query.probabilities,
            routing.key.probabilities,
        )
        cost = normalized_reach_cost(
            routing.query.probabilities,
            routing.key.probabilities,
            self.mosar_config,
        )
        self._pending_mosar_auxiliary = MoSARLayerAuxiliary(
            routing=routing,
            worth=worth,
            cost=cost,
        )
        return worth.log_bias

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        assert packed_seq_params is None, "Packed sequence is not supported by Gemma/MoSAR v0."
        assert attention_bias is None, "External attention_bias is not supported by Gemma/MoSAR v0."

        # Positional-variant patch. Q/K arrive here after full RoPE.
        if self.mosar_position_variant == "alibi":
            query = _inverse_rope_suffix(
                query,
                keep_rope_percentage=0.0,
                rotary_base=self.mosar_rotary_base,
            )
            key = _inverse_rope_suffix(
                key,
                keep_rope_percentage=0.0,
                rotary_base=self.mosar_rotary_base,
            )
        elif self.mosar_position_variant == "prope":
            query = _inverse_rope_suffix(
                query,
                keep_rope_percentage=self.mosar_prope_percentage,
                rotary_base=self.mosar_rotary_base,
            )
            key = _inverse_rope_suffix(
                key,
                keep_rope_percentage=self.mosar_prope_percentage,
                rotary_base=self.mosar_rotary_base,
            )

        # PATCH 1: role-specific routing after the final positional geometry.
        mosar_log_bias = self._mosar_log_bias(query, key)

        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            repeats = (
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            )
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)

        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query.dtype,
            "mpu",
        )
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),
            key.transpose(0, 1).transpose(1, 2),
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # Preserve Gemma 2's exact compatibility function.
        matmul_result = logit_softcapping(
            matmul_result,
            self.config.attn_logit_softcapping,
        )
        attention_scores = matmul_result.view(*output_size)

        if self.mosar_position_variant == "alibi":
            attention_scores = attention_scores + _alibi_bias(
                batch_size=output_size[0],
                num_heads=output_size[1],
                query_length=output_size[2],
                key_length=output_size[3],
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            )
        elif self.mosar_position_variant in {"rope_s_mask", "rope_m_mask"}:
            if self.mosar_hard_mask_window <= 0:
                raise RuntimeError(
                    f"Hard-mask variant {self.mosar_position_variant} requires "
                    f"mosar_hard_mask_window > 0."
                )
            attention_scores = attention_scores + _local_causal_hard_mask_bias(
                batch_size=output_size[0],
                num_heads=output_size[1],
                query_length=output_size[2],
                key_length=output_size[3],
                window_size=self.mosar_hard_mask_window,
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            )

        vanilla_attention_scores = attention_scores

        # PATCH 2: S_ij <- S_ij + log W_ij, shared across query heads.
        if mosar_log_bias is not None:
            expected = (output_size[0], output_size[2], output_size[3])
            if tuple(mosar_log_bias.shape) != expected:
                raise RuntimeError(
                    f"MoSAR log-bias shape {tuple(mosar_log_bias.shape)} != expected {expected}."
                )
            attention_scores = attention_scores + mosar_log_bias.to(
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            ).unsqueeze(1)

        # Preserve Gemma's local/global layer mask behavior.
        if attention_mask is not None and self.window_size is not None:
            attention_mask = _apply_gemma_sliding_window_mask(
                attention_mask,
                self.window_size,
                query_length=output_size[2],
                key_length=output_size[3],
            )

        should_capture = (
            getattr(self, "mosar_capture_diagnostics", False)
            and self.layer_number in getattr(self, "mosar_diagnostic_layers", ())
            and self._pending_mosar_auxiliary is not None
        )
        vanilla_attention_probs: Tensor | None = None
        if should_capture:
            vanilla_attention_probs = self.scale_mask_softmax(
                vanilla_attention_scores,
                attention_mask,
            )

        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)
        if should_capture:
            assert vanilla_attention_probs is not None
            assert self._pending_mosar_auxiliary is not None
            state = self._pending_mosar_auxiliary
            self._latest_mosar_diagnostics = build_layer_diagnostics(
                layer_number=self.layer_number,
                window_size=self.window_size,
                regime_names=self.mosar_config.regime_names,
                normalized_costs=self.mosar_config.normalized_costs,
                max_tokens=self.mosar_diagnostic_max_tokens,
                query_probabilities=state.routing.query.probabilities,
                key_probabilities=state.routing.key.probabilities,
                worth=state.worth.worth,
                log_bias=state.worth.log_bias,
                vanilla_attention_probabilities=vanilla_attention_probs,
                mosar_attention_probabilities=attention_probs,
                attention_mask=attention_mask,
                kernel=self.mosar_worth_field.kernel,
            )

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], output_size[3]
        )
        context = torch.bmm(attention_probs, value.transpose(0, 1))
        context = context.view(
            output_size[0],
            output_size[1],
            output_size[2],
            self.hidden_size_per_attention_head,
        )
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(
            output_size[2], output_size[0], self.hidden_size_per_partition
        )
        return context
