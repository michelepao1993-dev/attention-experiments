from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import MutableMapping
from typing import Any, Callable

import torch
from torch import Tensor

from .auxiliary import collect_mosar_auxiliary


VALID_TRAJECTORIES = {"vanilla", "mosar0", "cost_fixed", "cost_warmup", "s_fixed", "m_fixed", "alibi", "prope", "prope075", "rope_s_mask", "rope_m_mask"}


def _float(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


def _merge_metrics(metrics: MutableMapping[str, Any], additions: dict[str, Any]) -> None:
    for key, value in additions.items():
        metrics[key] = value


def _unpack_loss_result(result: Any) -> tuple[Tensor, Tensor | None, MutableMapping[str, Any], int]:
    """Normalize the two Megatron loss-function return conventions.

    Supported forms are ``(loss, metrics)`` and
    ``(loss, local_num_tokens, metrics)``.  The original arity is returned so
    the wrapper can preserve the pinned runtime contract exactly.
    """

    if not isinstance(result, tuple):
        raise TypeError(f"Megatron loss function returned {type(result)!r}, expected tuple")
    if len(result) == 2:
        loss, metrics = result
        local_num_tokens = None
    elif len(result) == 3:
        loss, local_num_tokens, metrics = result
    else:
        raise RuntimeError(
            "Unsupported Megatron loss result arity "
            f"{len(result)}; expected 2 or 3."
        )
    if not isinstance(loss, Tensor):
        raise TypeError("Megatron LM loss is not a tensor")
    if not isinstance(metrics, MutableMapping):
        raise TypeError("Megatron loss metrics are not a mutable mapping")
    return loss, local_num_tokens, metrics, len(result)


@dataclass(slots=True)
class MoSARObjectiveState:
    trajectory: str
    phase_start_step: int
    microbatches_per_step: int
    warmup_steps: int
    metrics_path: Path
    max_cost_weight: float = 1.0
    num_regimes: int = 3
    _microbatch_count: int = 0
    _pending: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.trajectory not in VALID_TRAJECTORIES:
            raise ValueError(f"unknown MoSAR trajectory {self.trajectory!r}")
        if self.phase_start_step < 0:
            raise ValueError("phase_start_step must be non-negative")
        if self.microbatches_per_step <= 0:
            raise ValueError("microbatches_per_step must be positive")
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if not 0.0 <= self.max_cost_weight <= 1.0:
            raise ValueError("max_cost_weight must be in [0, 1]")
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_vanilla(self) -> bool:
        return self.trajectory in {"vanilla", "alibi", "prope", "prope075", "rope_s_mask", "rope_m_mask"}

    @property
    def completed_steps_before_current_microbatch(self) -> int:
        return self.phase_start_step + self._microbatch_count // self.microbatches_per_step

    def cost_weight(self) -> float:
        if self.trajectory in {"vanilla", "mosar0", "s_fixed", "m_fixed", "alibi", "prope", "prope075", "rope_s_mask", "rope_m_mask"}:
            return 0.0
        if self.trajectory == "cost_fixed":
            return self.max_cost_weight
        completed = self.completed_steps_before_current_microbatch
        warmup_fraction = min(
            float(completed) / float(self.warmup_steps),
            1.0,
        )
        return self.max_cost_weight * warmup_fraction

    def record(self, values: dict[str, float]) -> None:
        self._pending.append(values)
        self._microbatch_count += 1
        if len(self._pending) < self.microbatches_per_step:
            return

        keys = sorted({key for item in self._pending for key in item})
        averaged: dict[str, float] = {}
        for key in keys:
            observed = [item[key] for item in self._pending if key in item]
            averaged[key] = sum(observed) / len(observed)
        averaged.update(
            {
                "trajectory": self.trajectory,
                "optimizer_step": self.phase_start_step
                + self._microbatch_count // self.microbatches_per_step,
                "microbatches": self.microbatches_per_step,
            }
        )
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(averaged, sort_keys=True) + "\n")
        self._pending.clear()


def make_mosar_forward_step(objective_state: MoSARObjectiveState) -> Callable[..., Any]:
    """Wrap the pinned Bridge GPT forward step with the MoSAR objective.

    The base data path, masks, shifted labels, and token-normalized LM loss stay
    untouched.  MoSAR only augments the returned scalar objective and metrics.
    """

    def mosar_forward_step(state: Any, data_iterator: Any, model: Any):
        from megatron.bridge.training.gpt_step import forward_step as bridge_gpt_forward_step

        base_result = bridge_gpt_forward_step(state, data_iterator, model)
        if not isinstance(base_result, tuple) or len(base_result) != 2:
            raise RuntimeError(
                "Pinned Bridge gpt_step.forward_step changed contract; expected "
                "(output_tensor, loss_func)."
            )
        output_tensor, base_loss_func = base_result

        vanilla_like = objective_state.trajectory in {"vanilla", "alibi", "prope", "prope075", "rope_s_mask", "rope_m_mask"}
        auxiliary = None if vanilla_like else collect_mosar_auxiliary(model)
        weight = objective_state.cost_weight()
        track_training = torch.is_grad_enabled()

        def mosar_loss_func(output: Any):
            base_loss, local_num_tokens, metrics, arity = _unpack_loss_result(
                base_loss_func(output)
            )

            # Bridge returns the summed LM loss together with the number of
            # valid tokens. Megatron Core performs the actual token and
            # microbatch normalization after this function returns.
            if local_num_tokens is None:
                token_count = None
                lm_loss = base_loss
            else:
                token_count = (
                    local_num_tokens.detach()
                    .to(device=base_loss.device, dtype=base_loss.dtype)
                    .reshape(())
                )
                safe_token_count = torch.clamp(token_count, min=1.0)
                lm_loss = base_loss / safe_token_count

            def reporting_metric(mean_value: Tensor) -> Tensor:
                """Encode a token-weighted mean using Bridge's native format."""
                detached = mean_value.detach().float().reshape(())
                if token_count is None:
                    return detached

                count = token_count.detach().float().reshape(())
                return torch.stack((detached * count, count))

            weight_tensor = torch.tensor(
                weight,
                device=base_loss.device,
                dtype=base_loss.dtype,
            )

            additions: dict[str, Any] = {
                "mosar/lm_loss": reporting_metric(lm_loss),
                "mosar/cost_weight": reporting_metric(weight_tensor),
            }
            record: dict[str, float] = {
                "lm_loss": _float(lm_loss),
                "cost_weight": weight,
            }

            if auxiliary is None:
                # Exact Vanilla path: preserve the native Bridge loss tensor.
                objective_mean = lm_loss
                backward_loss = base_loss
            else:
                cost_loss = auxiliary.cost_loss

                # Scientific objective:
                #   mean_LM + lambda * mean_cost
                objective_mean = lm_loss + weight * cost_loss

                # Bridge/Megatron expects a token-summed numerator here.
                if token_count is None:
                    backward_loss = objective_mean
                else:
                    backward_loss = (
                        base_loss
                        + token_count * weight * cost_loss
                    )

                additions.update(
                    {
                        "mosar/cost_loss": reporting_metric(cost_loss),
                        "mosar/query_cost": reporting_metric(
                            auxiliary.query_cost
                        ),
                        "mosar/key_cost": reporting_metric(
                            auxiliary.key_cost
                        ),
                        "mosar/query_entropy": reporting_metric(
                            auxiliary.mean_query_entropy
                        ),
                        "mosar/key_entropy": reporting_metric(
                            auxiliary.mean_key_entropy
                        ),
                    }
                )
                record.update(
                    {
                        "cost_loss": _float(cost_loss),
                        "query_cost": _float(auxiliary.query_cost),
                        "key_cost": _float(auxiliary.key_cost),
                        "query_entropy": _float(
                            auxiliary.mean_query_entropy
                        ),
                        "key_entropy": _float(
                            auxiliary.mean_key_entropy
                        ),
                    }
                )

                for index in range(
                    auxiliary.mean_query_probabilities.numel()
                ):
                    query_probability = (
                        auxiliary.mean_query_probabilities[index]
                    )
                    key_probability = (
                        auxiliary.mean_key_probabilities[index]
                    )

                    additions[f"mosar/query_regime_{index}"] = (
                        reporting_metric(query_probability)
                    )
                    additions[f"mosar/key_regime_{index}"] = (
                        reporting_metric(key_probability)
                    )
                    record[f"query_regime_{index}"] = _float(
                        query_probability
                    )
                    record[f"key_regime_{index}"] = _float(
                        key_probability
                    )

                global_index = auxiliary.layer_query_probabilities.shape[-1] - 1
                for layer_position, layer_number in enumerate(auxiliary.layer_numbers):
                    prefix = f"layer_{layer_number:02d}"
                    q_probs = auxiliary.layer_query_probabilities[layer_position]
                    k_probs = auxiliary.layer_key_probabilities[layer_position]
                    record[f"{prefix}_query_cost"] = _float(auxiliary.layer_query_costs[layer_position])
                    record[f"{prefix}_key_cost"] = _float(auxiliary.layer_key_costs[layer_position])
                    record[f"{prefix}_query_entropy"] = _float(auxiliary.layer_query_entropies[layer_position])
                    record[f"{prefix}_key_entropy"] = _float(auxiliary.layer_key_entropies[layer_position])
                    for regime_index in range(q_probs.numel()):
                        record[f"{prefix}_query_regime_{regime_index}"] = _float(q_probs[regime_index])
                        record[f"{prefix}_key_regime_{regime_index}"] = _float(k_probs[regime_index])
                    record[f"{prefix}_gg_mass"] = _float(q_probs[global_index] * k_probs[global_index])

            if not torch.isfinite(backward_loss).all():
                raise RuntimeError("MoSAR backward loss is non-finite")
            if not torch.isfinite(objective_mean).all():
                raise RuntimeError("MoSAR mean objective is non-finite")

            additions["mosar/total_loss"] = reporting_metric(
                objective_mean
            )
            record["total_loss"] = _float(objective_mean)
            _merge_metrics(metrics, additions)

            if track_training:
                objective_state.record(record)

            if arity == 2:
                return backward_loss, metrics
            return backward_loss, local_num_tokens, metrics

        return output_tensor, mosar_loss_func

    return mosar_forward_step
