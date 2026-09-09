import torch

from mosar.config import MoSARConfig
from mosar.hard_domain import build_hard_domain


def test_global_domain_is_full_causal_domain() -> None:
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8))
    q = torch.zeros(1, 8, 3)
    k = torch.zeros(1, 8, 3)
    q[..., 2] = 1
    k[..., 2] = 1
    output = build_hard_domain(q, k, config)
    assert output.allowed_pairs.item() == 8 * 9 // 2
    assert output.causal_pairs.item() == 8 * 9 // 2
    torch.testing.assert_close(output.density, torch.tensor(1.0))


def test_short_short_domain_has_known_density() -> None:
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8))
    q = torch.zeros(1, 8, 3)
    k = torch.zeros(1, 8, 3)
    q[..., 0] = 1
    k[..., 0] = 1
    output = build_hard_domain(q, k, config)
    # Distances 0, 1, 2 are retained: 1 + 2 + 3 + 5*3 = 21 pairs.
    assert output.allowed_pairs.item() == 21
    assert output.causal_pairs.item() == 36
    torch.testing.assert_close(output.density, torch.tensor(21 / 36))


def test_global_global_override_is_full_even_beyond_nominal_global_reach() -> None:
    config = MoSARConfig(sequence_length=12, reaches=(2, 4, 6))
    q = torch.zeros(1, 12, 3)
    k = torch.zeros(1, 12, 3)
    q[..., 2] = 1
    k[..., 2] = 1
    output = build_hard_domain(q, k, config)
    assert output.allowed_pairs.item() == 12 * 13 // 2
    assert output.causal_pairs.item() == 12 * 13 // 2
    torch.testing.assert_close(output.density, torch.tensor(1.0))


def test_global_query_can_create_disconnected_remote_island() -> None:
    config = MoSARConfig(sequence_length=16, reaches=(2, 6, 16))
    q = torch.zeros(1, 16, 3)
    k = torch.zeros(1, 16, 3)
    q[..., 0] = 1
    k[..., 0] = 1

    q[:, 15, :] = 0
    q[:, 15, 2] = 1
    k[:, 1, :] = 0
    k[:, 1, 2] = 1

    output = build_hard_domain(q, k, config)
    row = output.mask[0, 15]
    # G-S reach is 9. Far G at position 1 is retained, positions 2..5 are
    # excluded, and the local S block begins at position 6.
    assert row[1]
    assert not row[2]
    assert not row[5]
    assert row[6]
    assert row[15]

    starts = row & ~torch.cat([torch.tensor([False]), row[:-1]])
    assert starts.sum().item() == 2
