import torch

from mosar.config import MoSARConfig
from mosar.routers import PostPositionRouter


def _rotate_pairs(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Tiny test-only rotary transform over the last dimension."""

    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def test_router_can_observe_post_position_geometry() -> None:
    torch.manual_seed(23)
    config = MoSARConfig(sequence_length=4, reaches=(1, 2, 4), router_hidden_size=8)
    router = PostPositionRouter(num_heads=2, head_dim=4, config=config)

    # Same pre-RoPE content at every position.
    pre = torch.randn(1, 1, 2, 4).expand(1, 4, 2, 4).clone()
    positions = torch.arange(4, dtype=torch.float32)
    frequencies = torch.tensor([0.2, 0.7])
    angles = positions[:, None] * frequencies[None, :]
    post = _rotate_pairs(pre, angles)
    output = router(post)

    assert not torch.allclose(output.features[:, 0], output.features[:, 3])
