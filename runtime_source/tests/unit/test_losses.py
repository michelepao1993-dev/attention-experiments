import torch

from mosar.config import MoSARConfig
from mosar.losses import normalized_reach_cost


def _pure(index: int) -> torch.Tensor:
    probabilities = torch.zeros(2, 4, 3)
    probabilities[..., index] = 1
    return probabilities


def test_pure_regime_costs() -> None:
    config = MoSARConfig()
    for index, expected in enumerate((1 / 16, 1 / 4, 1.0)):
        output = normalized_reach_cost(_pure(index), _pure(index), config)
        torch.testing.assert_close(output.loss, torch.tensor(expected))


def test_uniform_cost_is_analytical_mean() -> None:
    config = MoSARConfig()
    probabilities = torch.full((2, 4, 3), 1 / 3)
    output = normalized_reach_cost(probabilities, probabilities, config)
    expected = torch.tensor(sum(config.normalized_costs) / 3)
    torch.testing.assert_close(output.loss, expected)
