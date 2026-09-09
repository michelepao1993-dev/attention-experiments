import torch

from mosar.config import MoSARConfig
from mosar.reference_attention import MoSARReferenceAttention


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(19)
    return (
        torch.randn(2, 7, 4, 8),
        torch.randn(2, 7, 2, 8),
        torch.randn(2, 7, 2, 8),
    )


def test_forced_global_matches_vanilla_reference() -> None:
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8), router_hidden_size=8)
    attention = MoSARReferenceAttention(
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        config=config,
    )
    q, k, v = _inputs()
    vanilla = attention(q, k, v, use_mosar=False)
    forced_global = attention(q, k, v, use_mosar=True, force_global=True)

    torch.testing.assert_close(vanilla.probabilities, forced_global.probabilities, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vanilla.output, forced_global.output, atol=1e-6, rtol=1e-6)


def test_future_attention_probability_is_zero() -> None:
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8), router_hidden_size=8)
    attention = MoSARReferenceAttention(
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        config=config,
    )
    q, k, v = _inputs()
    result = attention(q, k, v)
    future = torch.ones(7, 7, dtype=torch.bool).triu(diagonal=1)
    assert torch.count_nonzero(result.probabilities[..., future]) == 0


def test_lm_path_reaches_both_role_routers() -> None:
    config = MoSARConfig(sequence_length=8, reaches=(2, 4, 8), router_hidden_size=8)
    attention = MoSARReferenceAttention(
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        config=config,
    )
    q, k, v = (tensor.requires_grad_() for tensor in _inputs())
    result = attention(q, k, v)
    loss = result.output.square().mean()
    loss.backward()

    for router in (attention.routers.query_router, attention.routers.key_router):
        assert router.input_projection.weight.grad is not None
        assert router.output_projection.weight.grad is not None
        assert torch.isfinite(router.input_projection.weight.grad).all()
        assert torch.isfinite(router.output_projection.weight.grad).all()
        assert router.input_projection.weight.grad.abs().sum() > 0
        assert router.output_projection.weight.grad.abs().sum() > 0
