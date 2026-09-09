from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from mosar.losses import CostLossOutput
from mosar.outputs import RoutingState, WorthFieldOutput


@dataclass(slots=True)
class MoSARLayerAuxiliary:
    """Differentiable auxiliary state produced by one MoSAR attention layer."""

    routing: RoutingState
    worth: WorthFieldOutput
    cost: CostLossOutput


@dataclass(slots=True)
class MoSARAuxiliarySummary:
    # Aggregated differentiable cost plus global and per-layer diagnostics.
    cost_loss: Tensor
    query_cost: Tensor
    key_cost: Tensor
    mean_query_entropy: Tensor
    mean_key_entropy: Tensor
    mean_query_probabilities: Tensor
    mean_key_probabilities: Tensor
    num_layers: int
    layer_numbers: tuple[int, ...]
    layer_query_costs: Tensor
    layer_key_costs: Tensor
    layer_query_entropies: Tensor
    layer_key_entropies: Tensor
    layer_query_probabilities: Tensor
    layer_key_probabilities: Tensor


def _iter_mosar_attention_modules(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if callable(getattr(module, "consume_mosar_auxiliary", None)):
            yield module


def collect_mosar_auxiliary(model: nn.Module) -> MoSARAuxiliarySummary:
    # Consume one pending state from each MoSAR layer in layer-number order.
    numbered_states: list[tuple[int, MoSARLayerAuxiliary]] = []
    for fallback_number, module in enumerate(_iter_mosar_attention_modules(model), start=1):
        state = module.consume_mosar_auxiliary()
        if state is None:
            continue
        layer_number = int(getattr(module, "layer_number", fallback_number))
        numbered_states.append((layer_number, state))

    if not numbered_states:
        raise RuntimeError("No pending MoSAR auxiliary states were found.")

    numbered_states.sort(key=lambda item: item[0])
    layer_numbers = tuple(number for number, _ in numbered_states)
    states = [state for _, state in numbered_states]

    layer_query_costs = torch.stack([state.cost.query_cost for state in states])
    layer_key_costs = torch.stack([state.cost.key_cost for state in states])
    layer_query_entropies = torch.stack(
        [state.routing.query.entropy.float().mean() for state in states]
    )
    layer_key_entropies = torch.stack(
        [state.routing.key.entropy.float().mean() for state in states]
    )
    layer_query_probabilities = torch.stack(
        [state.routing.query.probabilities.float().mean(dim=(0, 1)) for state in states]
    )
    layer_key_probabilities = torch.stack(
        [state.routing.key.probabilities.float().mean(dim=(0, 1)) for state in states]
    )

    return MoSARAuxiliarySummary(
        cost_loss=torch.stack([state.cost.loss for state in states]).mean(),
        query_cost=layer_query_costs.mean(),
        key_cost=layer_key_costs.mean(),
        mean_query_entropy=layer_query_entropies.mean(),
        mean_key_entropy=layer_key_entropies.mean(),
        mean_query_probabilities=layer_query_probabilities.mean(dim=0),
        mean_key_probabilities=layer_key_probabilities.mean(dim=0),
        num_layers=len(states),
        layer_numbers=layer_numbers,
        layer_query_costs=layer_query_costs,
        layer_key_costs=layer_key_costs,
        layer_query_entropies=layer_query_entropies,
        layer_key_entropies=layer_key_entropies,
        layer_query_probabilities=layer_query_probabilities,
        layer_key_probabilities=layer_key_probabilities,
    )
