from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class RouterOutput:
    """Token-wise router output.

    All tensors use `[batch, sequence, ...]` leading dimensions.
    """

    features: Tensor
    logits: Tensor
    probabilities: Tensor

    @property
    def entropy(self) -> Tensor:
        probs = self.probabilities.clamp_min(torch.finfo(self.probabilities.dtype).tiny)
        return -(probs * probs.log()).sum(dim=-1)

    @property
    def top1(self) -> Tensor:
        return self.probabilities.argmax(dim=-1)


@dataclass(slots=True)
class RoutingState:
    query: RouterOutput
    key: RouterOutput


@dataclass(slots=True)
class WorthFieldOutput:
    worth: Tensor
    log_bias: Tensor
