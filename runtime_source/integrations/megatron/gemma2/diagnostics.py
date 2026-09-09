from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class MoSARLayerDiagnostics:
    """Detached diagnostics captured from one MoSAR attention layer.

    Heatmap tensors are cropped to a contiguous suffix of the sequence, while
    distance profiles and layer-level summaries use the complete sequence.
    All tensors are stored on CPU in float32 (masks remain bool).
    """

    layer_number: int
    attention_scope: str
    window_size: tuple[int, int] | None
    regime_names: tuple[str, ...]
    token_start: int
    token_end: int
    query_probabilities: Tensor
    key_probabilities: Tensor
    query_entropy: Tensor
    key_entropy: Tensor
    query_token_cost: Tensor
    key_token_cost: Tensor
    worth: Tensor
    log_bias: Tensor
    valid_attention_mask: Tensor
    vanilla_attention: Tensor
    mosar_attention: Tensor
    attention_delta: Tensor
    expected_regime_pair_mass: Tensor
    attention_weighted_regime_pair_mass: Tensor
    kernel_distances: Tensor
    kernel_values: Tensor
    distance_counts: Tensor
    worth_by_distance: Tensor
    vanilla_attention_by_distance: Tensor
    mosar_attention_by_distance: Tensor
    mean_query_probabilities: Tensor
    mean_key_probabilities: Tensor
    mean_query_entropy: Tensor
    mean_key_entropy: Tensor
    normalized_cost: Tensor
    mean_attention_total_variation: Tensor


def _cpu_float(tensor: Tensor) -> Tensor:
    return tensor.detach().float().cpu().contiguous()


def _distance_profile(
    matrix: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Mean matrix value by causal distance over positions that can be attended."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got {tuple(matrix.shape)}")
    sequence = matrix.shape[0]
    row = torch.arange(sequence, device=matrix.device)[:, None]
    col = torch.arange(sequence, device=matrix.device)[None, :]
    distances = row - col
    valid = distances >= 0
    if valid_mask is not None:
        if valid_mask.shape != matrix.shape:
            raise ValueError(
                f"valid mask shape {tuple(valid_mask.shape)} != matrix shape {tuple(matrix.shape)}"
            )
        valid = valid & valid_mask.to(device=matrix.device, dtype=torch.bool)
    valid = valid & torch.isfinite(matrix)
    flat_distances = distances[valid].long()
    flat_values = matrix[valid].float()
    sums = torch.zeros(sequence, device=matrix.device, dtype=torch.float32)
    counts = torch.zeros(sequence, device=matrix.device, dtype=torch.float32)
    if flat_distances.numel() > 0:
        sums.scatter_add_(0, flat_distances, flat_values)
        counts.scatter_add_(0, flat_distances, torch.ones_like(flat_values))
    profile = sums / counts.clamp_min(1.0)
    profile = profile.masked_fill(counts == 0, float("nan"))
    return profile, counts


def build_layer_diagnostics(
    *,
    layer_number: int,
    window_size: tuple[int, int] | None,
    regime_names: tuple[str, ...],
    normalized_costs: tuple[float, ...],
    max_tokens: int,
    query_probabilities: Tensor,
    key_probabilities: Tensor,
    worth: Tensor,
    log_bias: Tensor,
    vanilla_attention_probabilities: Tensor,
    mosar_attention_probabilities: Tensor,
    attention_mask: Tensor | None,
    kernel: nn.Module,
) -> MoSARLayerDiagnostics:
    """Create a compact, detached diagnostic snapshot.

    The first batch element is used for token-level heatmaps. Attention is
    averaged over query heads, because the current MoSAR worth field is shared
    across heads. Layer means still include the full batch.
    """

    if query_probabilities.ndim != 3 or key_probabilities.ndim != 3:
        raise ValueError("router probabilities must have shape [B, S, R]")
    if worth.ndim != 3 or log_bias.ndim != 3:
        raise ValueError("worth and log_bias must have shape [B, Sq, Sk]")
    if vanilla_attention_probabilities.ndim != 4 or mosar_attention_probabilities.ndim != 4:
        raise ValueError("attention probabilities must have shape [B, H, Sq, Sk]")

    sequence_q = query_probabilities.shape[1]
    sequence_k = key_probabilities.shape[1]
    if sequence_q != sequence_k:
        raise NotImplementedError("diagnostic v0 expects self-attention with Sq == Sk")
    sequence = sequence_q
    crop = min(int(max_tokens), sequence)
    start = sequence - crop

    q0 = query_probabilities[0].float()
    k0 = key_probabilities[0].float()
    worth0 = worth[0].float()
    log_bias0 = log_bias[0].float()
    vanilla0 = vanilla_attention_probabilities[0].float().mean(dim=0)
    mosar0 = mosar_attention_probabilities[0].float().mean(dim=0)

    if attention_mask is None:
        valid = torch.tril(
            torch.ones(sequence, sequence, device=worth.device, dtype=torch.bool)
        )
    else:
        mask = attention_mask
        while mask.ndim > 2:
            mask = mask[0]
        valid = ~mask.to(device=worth.device, dtype=torch.bool)
        if valid.shape != (sequence, sequence):
            valid = valid.expand(sequence, sequence)

    valid_float = valid.float()
    pair_denominator = valid_float.sum().clamp_min(1.0)
    expected_pair_mass = torch.einsum("im,jn,ij->mn", q0, k0, valid_float)
    expected_pair_mass = expected_pair_mass / pair_denominator

    attention_denominator = mosar0.sum().clamp_min(torch.finfo(torch.float32).eps)
    attention_pair_mass = torch.einsum("im,jn,ij->mn", q0, k0, mosar0)
    attention_pair_mass = attention_pair_mass / attention_denominator

    costs = torch.tensor(normalized_costs, device=q0.device, dtype=torch.float32)
    q_token_cost = torch.einsum("ir,r->i", q0, costs)
    k_token_cost = torch.einsum("jr,r->j", k0, costs)

    worth_profile, distance_counts = _distance_profile(worth0, valid)
    vanilla_profile, _ = _distance_profile(vanilla0, valid)
    mosar_profile, _ = _distance_profile(mosar0, valid)

    kernel_distances = torch.arange(sequence, device=worth.device, dtype=torch.float32)
    kernel_values = kernel(kernel_distances)

    q_entropy = -(q0.clamp_min(torch.finfo(q0.dtype).tiny) * q0.clamp_min(
        torch.finfo(q0.dtype).tiny
    ).log()).sum(dim=-1)
    k_entropy = -(k0.clamp_min(torch.finfo(k0.dtype).tiny) * k0.clamp_min(
        torch.finfo(k0.dtype).tiny
    ).log()).sum(dim=-1)

    mean_q = query_probabilities.float().mean(dim=(0, 1))
    mean_k = key_probabilities.float().mean(dim=(0, 1))
    mean_q_entropy = -(
        query_probabilities.float().clamp_min(torch.finfo(torch.float32).tiny)
        * query_probabilities.float().clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1).mean()
    mean_k_entropy = -(
        key_probabilities.float().clamp_min(torch.finfo(torch.float32).tiny)
        * key_probabilities.float().clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1).mean()
    normalized_cost = 0.5 * (
        torch.einsum("bir,r->bi", query_probabilities.float(), costs).mean()
        + torch.einsum("bjr,r->bj", key_probabilities.float(), costs).mean()
    )
    mean_attention_total_variation = 0.5 * (mosar0 - vanilla0).abs().sum(dim=-1).mean()

    crop_slice = slice(start, sequence)
    valid_crop = valid[crop_slice, crop_slice]
    vanilla_crop = vanilla0[crop_slice, crop_slice].masked_fill(~valid_crop, float("nan"))
    mosar_crop = mosar0[crop_slice, crop_slice].masked_fill(~valid_crop, float("nan"))
    worth_crop = worth0[crop_slice, crop_slice].masked_fill(~valid_crop, float("nan"))
    log_bias_crop = log_bias0[crop_slice, crop_slice].masked_fill(~valid_crop, float("nan"))

    return MoSARLayerDiagnostics(
        layer_number=int(layer_number),
        attention_scope="local" if window_size is not None else "global",
        window_size=tuple(window_size) if window_size is not None else None,
        regime_names=tuple(regime_names),
        token_start=start,
        token_end=sequence,
        query_probabilities=_cpu_float(q0[crop_slice]),
        key_probabilities=_cpu_float(k0[crop_slice]),
        query_entropy=_cpu_float(q_entropy[crop_slice]),
        key_entropy=_cpu_float(k_entropy[crop_slice]),
        query_token_cost=_cpu_float(q_token_cost[crop_slice]),
        key_token_cost=_cpu_float(k_token_cost[crop_slice]),
        worth=_cpu_float(worth_crop),
        log_bias=_cpu_float(log_bias_crop),
        valid_attention_mask=valid_crop.detach().cpu().contiguous(),
        vanilla_attention=_cpu_float(vanilla_crop),
        mosar_attention=_cpu_float(mosar_crop),
        attention_delta=_cpu_float(mosar_crop - vanilla_crop),
        expected_regime_pair_mass=_cpu_float(expected_pair_mass),
        attention_weighted_regime_pair_mass=_cpu_float(attention_pair_mass),
        kernel_distances=_cpu_float(kernel_distances),
        kernel_values=_cpu_float(kernel_values),
        distance_counts=_cpu_float(distance_counts),
        worth_by_distance=_cpu_float(worth_profile),
        vanilla_attention_by_distance=_cpu_float(vanilla_profile),
        mosar_attention_by_distance=_cpu_float(mosar_profile),
        mean_query_probabilities=_cpu_float(mean_q),
        mean_key_probabilities=_cpu_float(mean_k),
        mean_query_entropy=_cpu_float(mean_q_entropy),
        mean_key_entropy=_cpu_float(mean_k_entropy),
        normalized_cost=_cpu_float(normalized_cost),
        mean_attention_total_variation=_cpu_float(mean_attention_total_variation),
    )



def layer_diagnostics_to_dict(snapshot: MoSARLayerDiagnostics) -> dict[str, Any]:
    """Convert a snapshot to a pickle-portable dictionary of scalars/tensors."""

    return {field.name: getattr(snapshot, field.name) for field in fields(snapshot)}


def _iter_diagnostic_modules(model: nn.Module) -> Iterable[nn.Module]:
    for module in model.modules():
        if callable(getattr(module, "peek_mosar_diagnostics", None)):
            yield module


def collect_mosar_diagnostics(model: nn.Module) -> list[MoSARLayerDiagnostics]:
    snapshots: list[MoSARLayerDiagnostics] = []
    for module in _iter_diagnostic_modules(model):
        snapshot = module.peek_mosar_diagnostics()
        if snapshot is not None:
            snapshots.append(snapshot)
    snapshots.sort(key=lambda item: item.layer_number)
    if not snapshots:
        raise RuntimeError("No MoSAR diagnostics were captured.")
    return snapshots


def save_mosar_diagnostics(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshots = collect_mosar_diagnostics(model)
    torch.save(
        {
            "format_version": 1,
            "metadata": dict(metadata or {}),
            "layers": [layer_diagnostics_to_dict(snapshot) for snapshot in snapshots],
        },
        destination,
    )
    return destination
