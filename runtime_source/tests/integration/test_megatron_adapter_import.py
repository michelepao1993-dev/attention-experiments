from __future__ import annotations

from integrations.megatron.gemma2.core_attention import (
    _MEGATRON_IMPORT_ERROR,
    _resolve_settings,
)


class _Config:
    seq_length = 16
    mosar_mode = "uniform"
    mosar_reaches = (4, 8, 16)
    mosar_alphas = (0.75, 0.5, 1.0)


def test_adapter_module_is_importable_without_megatron() -> None:
    # Framework-free development must not fail merely by importing the adapter.
    assert _MEGATRON_IMPORT_ERROR is None or isinstance(_MEGATRON_IMPORT_ERROR, Exception)


def test_settings_resolve_without_megatron_runtime() -> None:
    settings = _resolve_settings(_Config())
    assert settings.mode == "uniform"
    assert settings.config.sequence_length == 16
    assert settings.config.reaches == (4, 8, 16)


def test_gemma_adapter_does_not_require_core_pg_collection_attribute() -> None:
    """Gemma2DotProductAttention is a custom core, not Core's standard DPA."""
    import inspect

    from integrations.megatron.gemma2.core_attention import MoSARGemma2DotProductAttention

    source = inspect.getsource(MoSARGemma2DotProductAttention.__init__)
    assert "self.pg_collection" not in source
    assert "self.config.tensor_model_parallel_size" in source
