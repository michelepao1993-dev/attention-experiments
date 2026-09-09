import math

import torch

from mosar.config import MoSARConfig
from mosar.regimes import RegimeKernel, pair_geometry


def test_protocol_pair_geometry() -> None:
    geometry = pair_geometry(MoSARConfig())
    expected_reaches = (
        (128, 320, 1088),
        (320, 512, 1280),
        (1088, 1280, 2048),
    )
    expected_thresholds = (
        (96, 200, 952),
        (200, 256, 960),
        (952, 960, 2048),
    )
    expected_lengths = (
        (32, 120, 136),
        (120, 256, 320),
        (136, 320, 0),
    )
    for m in range(3):
        for n in range(3):
            assert geometry[m][n].reach == expected_reaches[m][n]
            assert geometry[m][n].threshold == expected_thresholds[m][n]
            assert geometry[m][n].transition_length == expected_lengths[m][n]


def test_global_global_is_exact_identity() -> None:
    kernel = RegimeKernel(MoSARConfig())
    distances = torch.arange(0, 2048, dtype=torch.float32)
    values = kernel(distances)
    torch.testing.assert_close(values[..., 2, 2], torch.ones_like(distances), rtol=0, atol=0)


def test_non_global_kernels_are_monotone_and_bounded() -> None:
    config = MoSARConfig()
    values = RegimeKernel(config)(torch.arange(0, 2048, dtype=torch.float32))
    floor = math.exp(-config.bias_floor)
    for m in range(config.num_regimes):
        for n in range(config.num_regimes):
            if (m, n) == (config.global_regime_index, config.global_regime_index):
                continue
            curve = values[:, m, n]
            assert torch.all(curve[1:] <= curve[:-1] + 1e-7)
            assert curve.min() >= floor - 1e-7
            assert curve.max() <= 1.0


def test_v0_geometry_is_frozen_and_parameter_free() -> None:
    kernel = RegimeKernel(MoSARConfig())
    assert kernel.geometry_parameter_count == 0
    assert list(kernel.named_parameters()) == []
    signature = kernel.geometry_signature()
    assert signature["trainable_geometry_parameters"] == 0
    assert signature["regime_names"] == ("S", "M", "G")


def test_each_non_global_pair_has_exact_plateau_and_floor() -> None:
    config = MoSARConfig(sequence_length=32, reaches=(4, 16, 32))
    kernel = RegimeKernel(config)
    distances = torch.arange(0, 32, dtype=torch.float32)
    values = kernel(distances)
    geometry = pair_geometry(config)
    floor = math.exp(-config.bias_floor)
    g = config.global_regime_index

    for m in range(config.num_regimes):
        for n in range(config.num_regimes):
            if (m, n) == (g, g):
                continue
            entry = geometry[m][n]
            plateau = distances <= entry.threshold
            floor_region = distances >= entry.reach
            if plateau.any():
                torch.testing.assert_close(
                    values[plateau, m, n],
                    torch.ones_like(values[plateau, m, n]),
                    rtol=0,
                    atol=0,
                )
            if floor_region.any():
                torch.testing.assert_close(
                    values[floor_region, m, n],
                    torch.full_like(values[floor_region, m, n], floor),
                    rtol=1e-6,
                    atol=1e-7,
                )
