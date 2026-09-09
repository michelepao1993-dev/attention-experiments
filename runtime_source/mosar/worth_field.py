from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import MoSARConfig
from .outputs import WorthFieldOutput
from .regimes import RegimeKernel, causal_distances


class WorthField(nn.Module):
    r"""Bilinear separable MoSAR worth field.

    .. math::
        W_{bij} = \pi^Q_{bi}{}^T G(i-j) \pi^K_{bj}.
    """

    def __init__(self, config: MoSARConfig) -> None:
        super().__init__()
        self.config = config
        self.kernel = RegimeKernel(config)

    def forward(
        self,
        query_probabilities: Tensor,
        key_probabilities: Tensor,
        *,
        distances: Tensor | None = None,
    ) -> WorthFieldOutput:
        self._validate_probabilities(query_probabilities, key_probabilities)

        batch_q, query_length, _ = query_probabilities.shape
        batch_k, key_length, _ = key_probabilities.shape
        if batch_q != batch_k:
            raise ValueError("query and key probabilities must have the same batch size.")

        if distances is None:
            distances = causal_distances(
                query_length,
                key_length,
                device=query_probabilities.device,
                dtype=query_probabilities.dtype,
            )
        if distances.shape != (query_length, key_length):
            raise ValueError(
                f"distances must have shape {(query_length, key_length)}, "
                f"received {tuple(distances.shape)}"
            )

        # Accumulate the small regime contraction in float32 for BF16 safety.
        q = query_probabilities.float()
        k = key_probabilities.float()
        regime_values = self.kernel(distances.float())
        worth = torch.einsum("bim,ijmn,bjn->bij", q, regime_values, k)
        log_bias = torch.log(worth.clamp_min(self.config.epsilon))

        output_dtype = query_probabilities.dtype
        return WorthFieldOutput(
            worth=worth.to(dtype=output_dtype),
            log_bias=log_bias.to(dtype=output_dtype),
        )

    def _validate_probabilities(self, q: Tensor, k: Tensor) -> None:
        expected = self.config.num_regimes
        for name, tensor in (("query", q), ("key", k)):
            if tensor.ndim != 3:
                raise ValueError(
                    f"{name} probabilities must have shape [batch, sequence, regimes]."
                )
            if tensor.shape[-1] != expected:
                raise ValueError(
                    f"{name} probabilities use {tensor.shape[-1]} regimes; expected {expected}."
                )
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name} probabilities must be floating point.")
