import pytest

from mosar.config import MoSARConfig


def test_default_normalized_costs() -> None:
    config = MoSARConfig()
    assert config.normalized_costs == (1 / 16, 1 / 4, 1.0)
    assert config.global_regime_index == 2


def test_invalid_reach_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        MoSARConfig(sequence_length=128, reaches=(32, 64, 256))
