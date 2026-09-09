from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class MoSARConfig:
    """Configuration for the model-agnostic MoSAR core.

    The last regime is treated as the global regime. Its self-pair kernel is
    exactly one at every valid causal distance.
    """

    sequence_length: int = 2048
    regime_names: Tuple[str, ...] = ("S", "M", "G")
    reaches: Tuple[int, ...] = (128, 512, 2048)
    alphas: Tuple[float, ...] = (0.75, 0.5, 1.0)

    router_hidden_size: int = 64
    temperature: float = 1.0
    bias_floor: float = 6.0
    transition_power: float = 2.0
    epsilon: float = 1e-6

    input_weight_std: float = 0.02
    output_weight_std: float = 1e-3

    def __post_init__(self) -> None:
        n = len(self.regime_names)
        if n < 2:
            raise ValueError("MoSAR requires at least two regimes.")
        if len(self.reaches) != n or len(self.alphas) != n:
            raise ValueError("regime_names, reaches, and alphas must have equal length.")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if any(reach <= 0 for reach in self.reaches):
            raise ValueError("all reaches must be positive.")
        if any(reach > self.sequence_length for reach in self.reaches):
            raise ValueError("reaches cannot exceed sequence_length in the normalized-cost design.")
        if tuple(sorted(self.reaches)) != self.reaches:
            raise ValueError("reaches must be non-decreasing.")
        if any(not 0.0 <= alpha <= 1.0 for alpha in self.alphas):
            raise ValueError("alphas must lie in [0, 1].")
        if self.router_hidden_size <= 0:
            raise ValueError("router_hidden_size must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.bias_floor <= 0:
            raise ValueError("bias_floor must be positive.")
        if self.transition_power <= 0:
            raise ValueError("transition_power must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.input_weight_std <= 0 or self.output_weight_std <= 0:
            raise ValueError("router initialization standard deviations must be positive.")

    @property
    def num_regimes(self) -> int:
        return len(self.regime_names)

    @property
    def global_regime_index(self) -> int:
        return self.num_regimes - 1

    @property
    def normalized_costs(self) -> Tuple[float, ...]:
        return tuple(reach / self.sequence_length for reach in self.reaches)
