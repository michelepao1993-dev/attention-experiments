import torch

from mosar.config import MoSARConfig
from mosar.worth_field import WorthField


def test_worth_field_matches_explicit_separable_sum() -> None:
    torch.manual_seed(3)
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8))
    q = torch.softmax(torch.randn(2, 5, 3), dim=-1)
    k = torch.softmax(torch.randn(2, 5, 3), dim=-1)
    module = WorthField(config)
    result = module(q, k)
    kernel = module.kernel(
        torch.arange(5, dtype=torch.float32)[:, None]
        - torch.arange(5, dtype=torch.float32)[None, :]
    )

    explicit = torch.empty_like(result.worth)
    for b in range(2):
        for i in range(5):
            for j in range(5):
                explicit[b, i, j] = q[b, i] @ kernel[i, j] @ k[b, j]
    torch.testing.assert_close(result.worth, explicit)
    torch.testing.assert_close(result.log_bias, torch.log(explicit.clamp_min(config.epsilon)))


def test_pure_global_field_has_exactly_zero_bias() -> None:
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8))
    q = torch.zeros(1, 8, 3)
    k = torch.zeros(1, 8, 3)
    q[..., 2] = 1
    k[..., 2] = 1
    result = WorthField(config)(q, k)
    torch.testing.assert_close(result.worth, torch.ones_like(result.worth), rtol=0, atol=0)
    torch.testing.assert_close(result.log_bias, torch.zeros_like(result.log_bias), rtol=0, atol=0)
