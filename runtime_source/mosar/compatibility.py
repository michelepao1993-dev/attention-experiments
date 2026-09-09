from __future__ import annotations

import torch
from torch import Tensor


def gemma2_logit_softcap(scores: Tensor, cap: float | None) -> Tensor:
    """Apply Gemma 2's smooth logit cap to pre-softmax compatibility scores.

    ``cap=None`` or ``cap=0`` leaves the scores unchanged.  For a positive cap
    ``c`` this computes ``c * tanh(scores / c)`` elementwise.
    """

    if cap is None or cap == 0:
        return scores
    if cap < 0:
        raise ValueError("softcap must be non-negative or None.")
    return float(cap) * torch.tanh(scores / float(cap))
