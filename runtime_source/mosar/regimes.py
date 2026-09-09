from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import MoSARConfig


@dataclass(frozen=True, slots=True)
class PairGeometry:
    reach: float
    threshold: float
    transition_length: float


def pair_geometry(config: MoSARConfig) -> tuple[tuple[PairGeometry, ...], ...]:
    """Return the exact pair geometry implied by the regime parameters."""

    rows: list[tuple[PairGeometry, ...]] = []
    for delta_m, alpha_m in zip(config.reaches, config.alphas, strict=True):
        row: list[PairGeometry] = []
        for delta_n, alpha_n in zip(config.reaches, config.alphas, strict=True):
            delta = 0.5 * (delta_m + delta_n)
            alpha = 0.5 * (alpha_m + alpha_n)
            threshold = alpha * delta
            transition_length = (1.0 - alpha) * delta
            row.append(PairGeometry(delta, threshold, transition_length))
        rows.append(tuple(row))
    return tuple(rows)


def causal_distances(
    query_length: int,
    key_length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    query_position_offset: int = 0,
    key_position_offset: int = 0,
) -> Tensor:
    """Return `i-j` distances with shape `[query_length, key_length]`."""

    if query_length <= 0 or key_length <= 0:
        raise ValueError("query_length and key_length must be positive.")
    q = torch.arange(
        query_position_offset,
        query_position_offset + query_length,
        device=device,
        dtype=dtype,
    )
    k = torch.arange(
        key_position_offset,
        key_position_offset + key_length,
        device=device,
        dtype=dtype,
    )
    return q[:, None] - k[None, :]


class RegimeKernel(nn.Module):
    """Frozen plateau-transition geometry for every regime pair.

    The reaches, plateau fractions, transition power, and bias floor are
    configuration-level inductive biases. They are registered as buffers and
    are deliberately not trainable parameters. The learned part of MoSAR is
    the role-specific routing distribution that selects or mixes these fixed
    geometries.

    Input distances may have any shape. The returned tensor appends
    `[num_regimes, num_regimes]` to that shape.
    """

    def __init__(self, config: MoSARConfig) -> None:
        super().__init__()
        self.config = config
        geometry = pair_geometry(config)
        reaches = torch.tensor(
            [[entry.reach for entry in row] for row in geometry],
            dtype=torch.float32,
        )
        thresholds = torch.tensor(
            [[entry.threshold for entry in row] for row in geometry],
            dtype=torch.float32,
        )
        transition_lengths = torch.tensor(
            [[entry.transition_length for entry in row] for row in geometry],
            dtype=torch.float32,
        )
        self.register_buffer("pair_reaches", reaches, persistent=True)
        self.register_buffer("thresholds", thresholds, persistent=True)
        self.register_buffer("transition_lengths", transition_lengths, persistent=True)

    @property
    def geometry_parameter_count(self) -> int:
        """Return zero for the frozen v0 formulation by construction."""

        return sum(parameter.numel() for parameter in self.parameters())

    def geometry_signature(self) -> dict[str, object]:
        """Return a serializable description of the frozen geometry."""

        return {
            "regime_names": tuple(self.config.regime_names),
            "reaches": tuple(int(value) for value in self.config.reaches),
            "alphas": tuple(float(value) for value in self.config.alphas),
            "bias_floor": float(self.config.bias_floor),
            "transition_power": float(self.config.transition_power),
            "global_regime_index": int(self.config.global_regime_index),
            "trainable_geometry_parameters": int(self.geometry_parameter_count),
        }

    def forward(self, distances: Tensor) -> Tensor:
        if not torch.is_floating_point(distances):
            distances = distances.to(dtype=torch.float32)

        # Future positions are masked separately. Clamping here ensures the
        # kernel remains finite everywhere and does not encode anti-causal use.
        d = distances.clamp_min(0).to(dtype=torch.float32)
        d = d[..., None, None]

        thresholds = self.thresholds
        lengths = self.transition_lengths
        safe_lengths = lengths.clamp_min(torch.finfo(lengths.dtype).eps)
        progress = ((d - thresholds) / safe_lengths).clamp(0.0, 1.0)
        bias = -self.config.bias_floor * progress.pow(self.config.transition_power)

        # Preserve the exact plateau and floor branches.
        bias = torch.where(d <= thresholds, torch.zeros_like(bias), bias)
        bias = torch.where(
            d >= thresholds + lengths,
            torch.full_like(bias, -self.config.bias_floor),
            bias,
        )

        kernel = torch.exp(bias)
        global_index = self.config.global_regime_index
        kernel[..., global_index, global_index] = 1.0
        return kernel.to(dtype=distances.dtype)
