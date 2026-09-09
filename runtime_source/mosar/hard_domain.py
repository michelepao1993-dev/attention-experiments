from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import MoSARConfig
from .regimes import causal_distances


@dataclass(slots=True)
class HardDomainOutput:
    mask: Tensor
    density: Tensor
    allowed_pairs: Tensor
    causal_pairs: Tensor
    query_regimes: Tensor
    key_regimes: Tensor


def build_hard_domain(
    query_probabilities: Tensor,
    key_probabilities: Tensor,
    config: MoSARConfig,
    *,
    query_valid_mask: Tensor | None = None,
    key_valid_mask: Tensor | None = None,
    distances: Tensor | None = None,
) -> HardDomainOutput:
    """Construct the top-1 hard domain before evaluating any QK products.

    Non-global regime pairs retain causal positions up to their averaged reach.
    The global-global pair is an explicit identity route and therefore remains
    valid at every causal distance, independently of the nominal global reach.
    This explicit override is what allows disconnected domains: a global query
    may keep isolated remote global keys while dropping intervening non-global
    keys outside their mixed-pair reach.
    """

    if query_probabilities.ndim != 3 or key_probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [batch, sequence, regimes].")
    if query_probabilities.shape[0] != key_probabilities.shape[0]:
        raise ValueError("query and key probabilities must share the batch size.")
    if query_probabilities.shape[-1] != config.num_regimes:
        raise ValueError("query probability regime count does not match config.")
    if key_probabilities.shape[-1] != config.num_regimes:
        raise ValueError("key probability regime count does not match config.")

    batch, query_length, _ = query_probabilities.shape
    _, key_length, _ = key_probabilities.shape
    device = query_probabilities.device

    if distances is None:
        distances = causal_distances(
            query_length,
            key_length,
            device=device,
            dtype=torch.float32,
        )
    if distances.shape != (query_length, key_length):
        raise ValueError("distance shape does not match query/key lengths.")

    query_regimes = query_probabilities.argmax(dim=-1)
    key_regimes = key_probabilities.argmax(dim=-1)

    reaches = torch.tensor(config.reaches, device=device, dtype=torch.float32)
    q_reach = reaches[query_regimes]
    k_reach = reaches[key_regimes]
    pair_reach = 0.5 * (q_reach[:, :, None] + k_reach[:, None, :])

    causal = distances >= 0
    within_pair_reach = distances[None, :, :] <= pair_reach

    global_index = config.global_regime_index
    global_global = (
        (query_regimes[:, :, None] == global_index)
        & (key_regimes[:, None, :] == global_index)
    )

    hard = causal[None, :, :] & (within_pair_reach | global_global)

    valid_pairs = causal[None, :, :].expand(batch, -1, -1).clone()
    if query_valid_mask is not None:
        if query_valid_mask.shape != (batch, query_length):
            raise ValueError("query_valid_mask has the wrong shape.")
        valid_pairs &= query_valid_mask[:, :, None].bool()
    if key_valid_mask is not None:
        if key_valid_mask.shape != (batch, key_length):
            raise ValueError("key_valid_mask has the wrong shape.")
        valid_pairs &= key_valid_mask[:, None, :].bool()

    hard &= valid_pairs
    allowed_pairs = hard.sum()
    causal_pairs = valid_pairs.sum()
    if causal_pairs.item() == 0:
        raise ValueError("no valid causal pairs are available.")
    density = allowed_pairs.to(torch.float32) / causal_pairs.to(torch.float32)

    return HardDomainOutput(
        mask=hard,
        density=density,
        allowed_pairs=allowed_pairs,
        causal_pairs=causal_pairs,
        query_regimes=query_regimes,
        key_regimes=key_regimes,
    )
