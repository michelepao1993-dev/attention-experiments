from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import MoSARConfig


@dataclass(slots=True)
class CostLossOutput:
    loss: Tensor
    query_cost: Tensor
    key_cost: Tensor


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    if mask.shape != values.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}"
        )
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("cost mask selects zero tokens.")
    return (values * weights).sum() / denominator


def normalized_reach_cost(
    query_probabilities: Tensor,
    key_probabilities: Tensor,
    config: MoSARConfig,
    *,
    query_valid_mask: Tensor | None = None,
    key_valid_mask: Tensor | None = None,
) -> CostLossOutput:
    """Compute the normalized role-averaged reach cost."""

    if query_probabilities.ndim != 3 or key_probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [batch, sequence, regimes].")
    if query_probabilities.shape[-1] != config.num_regimes:
        raise ValueError("query probability regime count does not match config.")
    if key_probabilities.shape[-1] != config.num_regimes:
        raise ValueError("key probability regime count does not match config.")

    costs = torch.tensor(
        config.normalized_costs,
        device=query_probabilities.device,
        dtype=torch.float32,
    )
    q_values = torch.einsum("bir,r->bi", query_probabilities.float(), costs)
    k_values = torch.einsum("bjr,r->bj", key_probabilities.float(), costs)
    query_cost = _masked_mean(q_values, query_valid_mask)
    key_cost = _masked_mean(k_values, key_valid_mask)
    loss = 0.5 * (query_cost + key_cost)
    return CostLossOutput(loss=loss, query_cost=query_cost, key_cost=key_cost)
