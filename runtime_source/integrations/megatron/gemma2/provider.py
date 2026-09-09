from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mosar.config import MoSARConfig

from .layer_spec import mosar_gemma2_layer_spec

try:
    # Gemma2 500M provider used for the reported experiments.
    from .Gemma500Mprovider import Gemma2ModelProvider500M
except ImportError as exc:  # pragma: no cover - HPC-only dependency
    Gemma2ModelProvider500M = object  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class MoSARGemma2ModelProvider500M(Gemma2ModelProvider500M):  # type: ignore[misc]
    """500M Gemma provider wired to the MoSAR core-attention adapter."""

    mosar_mode: Literal["vanilla_copy", "forced_short", "forced_medium", "forced_global", "uniform", "learned"] = "learned"
    mosar_routing_inference: Literal["soft", "hard_top1"] = "soft"
    mosar_cost_weight: float = 0.0

    mosar_position_variant: Literal["rope", "alibi", "prope", "rope_s_mask", "rope_m_mask"] = "rope"
    mosar_prope_percentage: float = 0.5
    mosar_rotary_base: float = 10000.0
    mosar_hard_mask_window: int = 0

    mosar_sequence_length: int = 2048
    mosar_regime_names: tuple[str, ...] = ("S", "M", "G")
    mosar_router_hidden_size: int = 64
    mosar_temperature: float = 1.0
    mosar_reaches: tuple[int, ...] = (128, 512, 2048)
    mosar_alphas: tuple[float, ...] = (0.75, 0.5, 1.0)
    mosar_bias_floor: float = 6.0
    mosar_transition_power: float = 2.0
    mosar_epsilon: float = 1e-6
    mosar_input_weight_std: float = 0.02
    mosar_output_weight_std: float = 1e-3

    mosar_capture_diagnostics: bool = False
    mosar_diagnostic_layers: tuple[int, ...] = (1, 2, 9, 18)
    mosar_diagnostic_max_tokens: int = 128

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "The pinned Megatron Bridge stack is required to instantiate the Gemma2 500M provider."
            ) from _IMPORT_ERROR
        self.transformer_layer_spec = mosar_gemma2_layer_spec
        if self.mosar_cost_weight < 0.0:
            raise ValueError("mosar_cost_weight must be non-negative.")
        if self.mosar_position_variant not in {"rope", "alibi", "prope", "rope_s_mask", "rope_m_mask"}:
            raise ValueError(f"unsupported mosar_position_variant={self.mosar_position_variant!r}.")
        if not 0.0 <= self.mosar_prope_percentage <= 1.0:
            raise ValueError("mosar_prope_percentage must be in [0, 1].")
        if self.mosar_mode != "learned" and self.mosar_cost_weight != 0.0:
            raise ValueError("A cost loss is meaningful only for learned routing.")
        if self.tensor_model_parallel_size != 1:
            raise NotImplementedError("MoSAR v0 requires tensor_model_parallel_size=1.")
        if self.context_parallel_size != 1:
            raise NotImplementedError("Gemma/MoSAR v0 requires context_parallel_size=1.")
        if self.mosar_diagnostic_max_tokens <= 0:
            raise ValueError("mosar_diagnostic_max_tokens must be positive.")
        invalid_layers = [
            layer
            for layer in self.mosar_diagnostic_layers
            if layer < 1 or layer > self.num_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"mosar_diagnostic_layers contains invalid layers {invalid_layers}; "
                f"valid range is [1, {self.num_layers}]."
            )
        self.build_mosar_config()

    def build_mosar_config(self) -> MoSARConfig:
        return MoSARConfig(
            sequence_length=self.mosar_sequence_length,
            regime_names=self.mosar_regime_names,
            reaches=self.mosar_reaches,
            alphas=self.mosar_alphas,
            router_hidden_size=self.mosar_router_hidden_size,
            temperature=self.mosar_temperature,
            bias_floor=self.mosar_bias_floor,
            transition_power=self.mosar_transition_power,
            epsilon=self.mosar_epsilon,
            input_weight_std=self.mosar_input_weight_std,
            output_weight_std=self.mosar_output_weight_std,
        )
