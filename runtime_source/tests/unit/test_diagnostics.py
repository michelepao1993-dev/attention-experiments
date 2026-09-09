from __future__ import annotations

import torch
from torch import nn

from integrations.megatron.gemma2.diagnostics import build_layer_diagnostics


class _Kernel(nn.Module):
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        regimes = 3
        base = torch.exp(-distances.float() / 8.0)
        values = base[:, None, None].expand(-1, regimes, regimes).clone()
        values[:, -1, -1] = 1.0
        return values


def test_layer_diagnostics_shapes_and_normalization() -> None:
    torch.manual_seed(4)
    batch, heads, sequence, regimes = 1, 2, 8, 3
    q = torch.softmax(torch.randn(batch, sequence, regimes), dim=-1)
    k = torch.softmax(torch.randn(batch, sequence, regimes), dim=-1)
    worth = torch.rand(batch, sequence, sequence).clamp_min(0.05)
    log_bias = worth.log()
    mask = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)[None, None]
    scores = torch.randn(batch, heads, sequence, sequence).masked_fill(mask, float("-inf"))
    vanilla = torch.softmax(scores, dim=-1)
    mosar = torch.softmax(scores + log_bias[:, None], dim=-1)

    snapshot = build_layer_diagnostics(
        layer_number=2,
        window_size=(4, 0),
        regime_names=("S", "M", "G"),
        normalized_costs=(0.125, 0.5, 1.0),
        max_tokens=6,
        query_probabilities=q,
        key_probabilities=k,
        worth=worth,
        log_bias=log_bias,
        vanilla_attention_probabilities=vanilla,
        mosar_attention_probabilities=mosar,
        attention_mask=mask,
        kernel=_Kernel(),
    )

    assert snapshot.attention_scope == "local"
    assert snapshot.window_size == (4, 0)
    assert snapshot.query_probabilities.shape == (6, 3)
    assert snapshot.key_probabilities.shape == (6, 3)
    assert snapshot.worth.shape == (6, 6)
    assert snapshot.vanilla_attention.shape == (6, 6)
    assert snapshot.kernel_values.shape == (sequence, regimes, regimes)
    torch.testing.assert_close(snapshot.expected_regime_pair_mass.sum(), torch.tensor(1.0))
    torch.testing.assert_close(
        snapshot.attention_weighted_regime_pair_mass.sum(), torch.tensor(1.0)
    )
    assert torch.isfinite(snapshot.mean_query_entropy)
    assert torch.isfinite(snapshot.mean_key_entropy)
    assert 0.0 <= snapshot.normalized_cost.item() <= 1.0
    assert 0.0 <= snapshot.mean_attention_total_variation.item() <= 1.0


def test_local_mask_removes_unreachable_distances_from_profiles() -> None:
    batch, heads, sequence, regimes = 1, 2, 8, 3
    q = torch.full((batch, sequence, regimes), 1.0 / regimes)
    k = torch.full((batch, sequence, regimes), 1.0 / regimes)
    worth = torch.full((batch, sequence, sequence), 0.5)
    log_bias = worth.log()
    row = torch.arange(sequence)[:, None]
    col = torch.arange(sequence)[None, :]
    mask_2d = (col > row) | ((row - col) >= 4)
    mask = mask_2d[None, None]
    scores = torch.zeros(batch, heads, sequence, sequence).masked_fill(mask, float("-inf"))
    vanilla = torch.softmax(scores, dim=-1)
    mosar = torch.softmax(scores + log_bias[:, None], dim=-1)

    snapshot = build_layer_diagnostics(
        layer_number=2,
        window_size=(4, 0),
        regime_names=("S", "M", "G"),
        normalized_costs=(0.125, 0.5, 1.0),
        max_tokens=sequence,
        query_probabilities=q,
        key_probabilities=k,
        worth=worth,
        log_bias=log_bias,
        vanilla_attention_probabilities=vanilla,
        mosar_attention_probabilities=mosar,
        attention_mask=mask,
        kernel=_Kernel(),
    )

    assert torch.all(snapshot.distance_counts[:4] > 0)
    assert torch.all(snapshot.distance_counts[4:] == 0)
    assert torch.all(torch.isnan(snapshot.worth_by_distance[4:]))
