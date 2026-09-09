import math

import torch

from mosar.config import MoSARConfig
from mosar.routers import RoleSpecificPostPositionRouters


def test_role_specific_gqa_shapes_and_uniform_initialization() -> None:
    torch.manual_seed(7)
    config = MoSARConfig(router_hidden_size=16)
    routers = RoleSpecificPostPositionRouters(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=4,
        config=config,
    )
    q = torch.randn(3, 11, 8, 4)
    k = torch.randn(3, 11, 2, 4)
    state = routers(q, k)

    assert state.query.probabilities.shape == (3, 11, 3)
    assert state.key.probabilities.shape == (3, 11, 3)
    q_mean = state.query.probabilities.mean(dim=(0, 1))
    k_mean = state.key.probabilities.mean(dim=(0, 1))
    target = torch.full((3,), 1 / 3)
    torch.testing.assert_close(q_mean, target, atol=0.02, rtol=0)
    torch.testing.assert_close(k_mean, target, atol=0.02, rtol=0)
    assert state.query.entropy.mean() > 0.98 * math.log(3)
    assert state.key.entropy.mean() > 0.98 * math.log(3)


def test_query_and_key_router_parameters_are_distinct() -> None:
    config = MoSARConfig(router_hidden_size=8)
    routers = RoleSpecificPostPositionRouters(
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=4,
        config=config,
    )
    assert routers.query_router.input_projection.weight.data_ptr() != (
        routers.key_router.input_projection.weight.data_ptr()
    )


def test_router_gradients_are_finite_and_nonzero() -> None:
    torch.manual_seed(11)
    config = MoSARConfig(router_hidden_size=8)
    routers = RoleSpecificPostPositionRouters(
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=4,
        config=config,
    )
    q = torch.randn(2, 5, 4, 4, requires_grad=True)
    k = torch.randn(2, 5, 2, 4, requires_grad=True)
    state = routers(q, k)
    loss = state.query.probabilities[..., 0].mean() + state.key.probabilities[..., 1].mean()
    loss.backward()

    for parameter in routers.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
