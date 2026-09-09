from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

import integrations.megatron.gemma2.core_attention as adapter
from mosar.config import MoSARConfig
from mosar.routers import RoleSpecificPostPositionRouters
from mosar.worth_field import WorthField


class _Buffer:
    @staticmethod
    def get_tensor(shape, dtype, name):
        del name
        return torch.empty(shape, dtype=dtype)


class _ParallelState:
    @staticmethod
    def get_global_memory_buffer():
        return _Buffer()


class _TensorParallel:
    class _Tracker:
        @staticmethod
        def fork():
            return nullcontext()

    @staticmethod
    def get_cuda_rng_tracker():
        return _TensorParallel._Tracker()


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(dim=-1)


def _build(mode: str) -> adapter.MoSARGemma2DotProductAttention:
    module = adapter.MoSARGemma2DotProductAttention.__new__(
        adapter.MoSARGemma2DotProductAttention
    )
    nn.Module.__init__(module)
    module.config = SimpleNamespace(
        attn_logit_softcapping=50.0,
        sequence_parallel=True,
    )
    module.num_attention_heads_per_partition = 4
    module.num_query_groups_per_partition = 1
    module.hidden_size_per_attention_head = 8
    module.hidden_size_per_partition = 32
    module.norm_factor = 8.0**0.5
    module.window_size = None
    module.scale_mask_softmax = _masked_softmax
    module.attention_dropout = nn.Identity()
    module.mosar_mode = mode
    module.mosar_position_variant = "rope"
    module.mosar_config = MoSARConfig(
        sequence_length=6,
        reaches=(2, 4, 6),
        alphas=(0.75, 0.5, 1.0),
        router_hidden_size=8,
        bias_floor=4.0,
    )
    module.mosar_routers = RoleSpecificPostPositionRouters(
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=8,
        config=module.mosar_config,
    )
    module.mosar_worth_field = WorthField(module.mosar_config)
    module._pending_mosar_auxiliary = None
    return module


def _inputs():
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(6, 2, 4, 8, generator=generator)
    k = torch.randn(6, 2, 1, 8, generator=generator)
    v = torch.randn(6, 2, 1, 8, generator=generator)
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)[None, None]
    return q, k, v, mask


def _direct_gemma(q, k, v, mask):
    k = k.repeat_interleave(4, dim=2)
    v = v.repeat_interleave(4, dim=2)
    scores = torch.einsum("sbhd,tbhd->bhst", q, k) / (8.0**0.5)
    scores = 50.0 * torch.tanh(scores / 50.0)
    probs = scores.masked_fill(mask, float("-inf")).softmax(-1)
    context = torch.einsum("bhst,tbhd->sbhd", probs, v)
    return context.reshape(6, 2, 32)


def test_isolated_vanilla_copy_matches_direct_gemma(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "parallel_state", _ParallelState())
    monkeypatch.setattr(adapter, "tensor_parallel", _TensorParallel())
    monkeypatch.setattr(
        adapter,
        "logit_softcapping",
        lambda x, c: c * torch.tanh(x / c),
    )
    q, k, v, mask = _inputs()
    module = _build("vanilla_copy")
    actual = module(q, k, v, mask)
    expected = _direct_gemma(q, k, v, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_isolated_forced_global_is_exact_vanilla(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "parallel_state", _ParallelState())
    monkeypatch.setattr(adapter, "tensor_parallel", _TensorParallel())
    monkeypatch.setattr(
        adapter,
        "logit_softcapping",
        lambda x, c: c * torch.tanh(x / c),
    )
    q, k, v, mask = _inputs()
    vanilla = _build("vanilla_copy")(q, k, v, mask)
    forced_module = _build("forced_global")
    forced = forced_module(q, k, v, mask)
    torch.testing.assert_close(forced, vanilla, rtol=0.0, atol=0.0)
    assert forced_module.consume_mosar_auxiliary() is not None


def test_isolated_uniform_is_finite_and_nontrivial(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "parallel_state", _ParallelState())
    monkeypatch.setattr(adapter, "tensor_parallel", _TensorParallel())
    monkeypatch.setattr(
        adapter,
        "logit_softcapping",
        lambda x, c: c * torch.tanh(x / c),
    )
    q, k, v, mask = _inputs()
    vanilla = _build("vanilla_copy")(q, k, v, mask)
    uniform_module = _build("uniform")
    uniform = uniform_module(q, k, v, mask)
    assert torch.isfinite(uniform).all()
    assert not torch.allclose(uniform, vanilla)
    state = uniform_module.consume_mosar_auxiliary()
    assert state is not None
    torch.testing.assert_close(
        state.routing.query.probabilities,
        torch.full_like(state.routing.query.probabilities, 1.0 / 3.0),
    )
