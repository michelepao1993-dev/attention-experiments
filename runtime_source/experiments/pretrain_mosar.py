#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch

from integrations.megatron.gemma2 import MoSARGemma2ModelProvider500M
from integrations.megatron.gemma2.mosar_forward_step import (
    MoSARObjectiveState,
    make_mosar_forward_step,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.pretrain import pretrain


TRAJECTORY_SETTINGS = {
    "vanilla": {"mode": "vanilla_copy", "cost_weight": 0.0},
    "mosar0": {"mode": "learned", "cost_weight": 0.0},
    "cost_fixed": {"mode": "learned", "cost_weight": 1.0},
    "cost_warmup": {"mode": "learned", "cost_weight": 1.0},
    "s_fixed": {"mode": "forced_short", "cost_weight": 0.0},
    "m_fixed": {"mode": "forced_medium", "cost_weight": 0.0},
    "alibi": {"mode": "vanilla_copy", "cost_weight": 0.0, "position_variant": "alibi"},
    "prope": {"mode": "vanilla_copy", "cost_weight": 0.0, "position_variant": "prope", "prope_percentage": 0.5},
    "prope075": {"mode": "vanilla_copy", "cost_weight": 0.0, "position_variant": "prope", "prope_percentage": 0.75},
    "rope_s_mask": {"mode": "vanilla_copy", "cost_weight": 0.0, "position_variant": "rope_s_mask", "hard_mask_window": 128},
    "rope_m_mask": {"mode": "vanilla_copy", "cost_weight": 0.0, "position_variant": "rope_m_mask", "hard_mask_window": 512},
}


def _validate_user_path(path: Path, label: str) -> None:
    text = str(path.resolve())
    forbidden = tuple(os.environ.get("MOSAR_FORBIDDEN_PATH_TOKENS", "").split())
    if forbidden and any(token in text for token in forbidden):
        raise RuntimeError(f"{label} contains a forbidden path token: {text}")


def _validate_data_prefix(prefix: Path) -> None:
    _validate_user_path(prefix, "data prefix")
    missing = [str(prefix) + suffix for suffix in (".bin", ".idx") if not Path(str(prefix) + suffix).is_file()]
    if missing:
        raise FileNotFoundError(
            "Megatron indexed dataset files are missing: " + ", ".join(missing)
        )


def _validate_tokenizer(path: Path) -> None:
    _validate_user_path(path, "tokenizer")
    if not path.is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {path}")
    candidates = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    if not any((path / name).is_file() for name in candidates):
        raise FileNotFoundError(
            f"tokenizer directory {path} contains none of {candidates}"
        )


def _set_path(root: Any, dotted: str, value: Any, *, required: bool = True) -> bool:
    parts = dotted.split(".")
    current = root
    for part in parts[:-1]:
        if not hasattr(current, part):
            if required:
                raise AttributeError(f"configuration has no path {dotted!r}")
            return False
        current = getattr(current, part)
    leaf = parts[-1]
    if not hasattr(current, leaf):
        if required:
            raise AttributeError(f"configuration has no path {dotted!r}")
        return False
    setattr(current, leaf, value)
    return True


def _set_first(root: Any, paths: tuple[str, ...], value: Any) -> str | None:
    for path in paths:
        if _set_path(root, path, value, required=False):
            return path
    return None



def _try_set_path(root: Any, dotted: str, value: Any) -> bool:
    """Best-effort config setter used only for eval-only posttraining jobs."""
    parts = dotted.split(".")
    current = root
    for part in parts[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    leaf = parts[-1]
    if not hasattr(current, leaf):
        return False
    setattr(current, leaf, value)
    return True


def _reset_eval_consumed_samples(cfg: Any) -> list[str]:
    """Reset Megatron/Bridge consumed-sample counters for eval-only extrapolation.

    This avoids train-sampler exhaustion when loading a high-iteration checkpoint
    and evaluating with a different sequence length / batch size.
    """
    candidate_paths = (
        "train.consumed_train_samples",
        "train.consumed_valid_samples",
        "train.consumed_samples",
        "train.consumed_train_tokens",
        "train.consumed_valid_tokens",
        "data.consumed_train_samples",
        "data.consumed_valid_samples",
        "dataset.consumed_train_samples",
        "dataset.consumed_valid_samples",
    )
    changed = []
    for dotted in candidate_paths:
        if _try_set_path(cfg, dotted, 0):
            changed.append(dotted)
    return changed


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    return repr(value)


def _sampled_file_digest(path: Path, sample_bytes: int = 4 * 1024 * 1024) -> str:
    """Fast identity probe for potentially huge indexed-data files.

    The digest includes file size plus first and last samples.  It is not a
    replacement for the immutable dataset manifest used by final training.
    """

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def build_config(args: argparse.Namespace):
    cfg = _pretrain_common()
    settings = TRAJECTORY_SETTINGS[args.trajectory]
    resolved_mode = settings["mode"]
    if args.routing_override != "native":
        if args.trajectory == "vanilla":
            raise ValueError("routing overrides are valid only for MoSAR checkpoints")
        resolved_mode = args.routing_override

    cfg.model = MoSARGemma2ModelProvider500M(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=args.sequence_length,
        vocab_size=args.vocab_size,
        should_pad_vocab=False,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        transformer_impl="transformer_engine",
        cuda_graph_impl="none",
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        gradient_accumulation_fusion=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        mosar_mode=resolved_mode,
        mosar_routing_inference=args.routing_inference,
        mosar_cost_weight=(0.0 if args.eval_only or args.trajectory in {"vanilla", "mosar0", "s_fixed", "m_fixed", "alibi", "prope", "prope075", "rope_s_mask", "rope_m_mask"} else args.max_cost_weight),
        mosar_position_variant=TRAJECTORY_SETTINGS[args.trajectory].get("position_variant", "rope"),
        mosar_prope_percentage=float(TRAJECTORY_SETTINGS[args.trajectory].get("prope_percentage", 0.5)),
        mosar_rotary_base=10000.0,
        mosar_hard_mask_window=int(TRAJECTORY_SETTINGS[args.trajectory].get("hard_mask_window", 0)),
        mosar_sequence_length=args.sequence_length,
        mosar_regime_names=("S", "M", "G"),
        mosar_reaches=(128, 512, args.sequence_length),
        mosar_alphas=(0.75, 0.5, 1.0),
        mosar_router_hidden_size=64,
        mosar_temperature=1.0,
        mosar_bias_floor=6.0,
        mosar_transition_power=2.0,
        mosar_epsilon=1e-6,
        mosar_input_weight_std=0.02,
        mosar_output_weight_std=1e-3,
        mosar_capture_diagnostics=False,
    )

    # Gemma/DDP settings follow the known-good from-scratch Bridge recipe.
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.attention_backend = None
    cfg.model.cuda_graph_scope = "full"

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    _set_path(cfg, "optimizer.lr", args.learning_rate)
    if _set_first(cfg, ("optimizer.min_lr", "scheduler.min_lr"), args.min_learning_rate) is None:
        raise AttributeError("could not resolve min_lr in optimizer or scheduler config")
    _set_path(cfg, "optimizer.weight_decay", args.weight_decay)
    _set_path(cfg, "optimizer.adam_beta1", 0.9)
    _set_path(cfg, "optimizer.adam_beta2", 0.95)
    _set_first(cfg, ("optimizer.adam_eps", "optimizer.adam_epsilon"), 1e-8)
    _set_path(cfg, "optimizer.clip_grad", 1.0)

    if hasattr(cfg, "scheduler"):
        _set_path(cfg, "scheduler.lr_decay_style", "cosine")
        _set_path(cfg, "scheduler.lr_decay_iters", args.total_steps)
        _set_path(cfg, "scheduler.lr_warmup_iters", args.lr_warmup_steps)
        _set_first(cfg, ("scheduler.min_lr",), args.min_learning_rate)

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = str(args.tokenizer_path)

    cfg.dataset.data_path = [str(args.data_prefix)]
    cfg.dataset.split = "990,10,0"
    cfg.dataset.sequence_length = args.sequence_length
    cfg.dataset.num_workers = 1
    dataset_cache_dir = args.output_dir.parent / "dataset_cache"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset.path_to_cache = str(dataset_cache_dir)
    _set_first(cfg, ("dataset.random_seed", "dataset.seed"), args.data_seed)

    cfg.train.train_iters = args.phase_end_step
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size

    cfg.validation.skip_train = args.eval_only
    cfg.validation.eval_iters = args.eval_iters

    cfg.checkpoint.save = str(args.checkpoint_dir)
    cfg.checkpoint.load = str(args.checkpoint_dir)
    cfg.checkpoint.save_interval = args.checkpoint_interval
    _set_first(cfg, ("checkpoint.async_save",), False)

    if hasattr(cfg, "logger"):
        _set_first(cfg, ("logger.log_interval",), 1)
        _set_first(cfg, ("logger.tensorboard_dir",), str(args.output_dir / "tensorboard"))
        _set_first(cfg, ("logger.wandb_project",), None)

    resolved_seed_path = _set_first(
        cfg,
        ("rng.seed", "train.seed", "model.seed"),
        args.model_seed,
    )
    if resolved_seed_path is None:
        raise AttributeError(
            "could not resolve the Bridge RNG seed field; matched run cannot proceed"
        )

    return cfg, resolved_seed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", choices=tuple(TRAJECTORY_SETTINGS), required=True)
    parser.add_argument("--data-prefix", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-start-step", type=int, required=True)
    parser.add_argument("--phase-end-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=256000)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=4)
    parser.add_argument("--cost-warmup-steps", type=int, default=16)
    parser.add_argument("--max-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--eval-reset-consumed-samples",
        action="store_true",
        default=os.environ.get("EVAL_RESET_CONSUMED_SAMPLES", "0") in {"1", "true", "True", "yes", "YES"},
        help="For eval-only posttraining/extrapolation: reset consumed-sample counters after loading config.",
    )
    parser.add_argument(
        "--routing-inference",
        choices=("soft", "hard_top1"),
        default=os.environ.get("MOSAR_ROUTING_INFERENCE", "soft"),
        help="Post-training routing inference mode: soft router probabilities or hard top-1 one-hot projection.",
    )
    parser.add_argument(
        "--routing-override",
        choices=("native", "forced_short", "forced_medium", "forced_global"),
        default="native",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-iters", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--data-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.global_batch_size % args.micro_batch_size != 0:
        raise ValueError("global batch size must be divisible by micro batch size")
    if not 0.0 <= args.max_cost_weight <= 1.0:
        raise ValueError("max cost weight must be in [0, 1]")
    if args.eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    if args.eval_only:
        if not 0 <= args.phase_start_step == args.phase_end_step <= args.total_steps:
            raise ValueError("eval-only requires phase_start_step == phase_end_step")
    elif not 0 <= args.phase_start_step < args.phase_end_step <= args.total_steps:
        raise ValueError("invalid training phase interval")
    _validate_data_prefix(args.data_prefix)
    _validate_tokenizer(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHONHASHSEED", str(args.model_seed))
    random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)

    cfg, resolved_seed_path = build_config(args)

    if args.eval_only and args.eval_reset_consumed_samples:
        changed_consumed_paths = _reset_eval_consumed_samples(cfg)
        print(
            "[posttraining] reset consumed-sample counters: "
            + (", ".join(changed_consumed_paths) if changed_consumed_paths else "<none found>")
        )
    config_path = args.output_dir / f"resolved_config_phase_{args.phase_start_step}_{args.phase_end_step}.json"
    config_path.write_text(
        json.dumps(_jsonable(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "trajectory": args.trajectory,
        "phase_start_step": args.phase_start_step,
        "phase_end_step": args.phase_end_step,
        "total_steps": args.total_steps,
        "max_cost_weight": args.max_cost_weight,
        "cost_warmup_steps": args.cost_warmup_steps,
        "routing_override": args.routing_override,
        "eval_only": args.eval_only,
        "eval_iters": args.eval_iters,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "resolved_seed_config_path": resolved_seed_path,
        "data_prefix": str(args.data_prefix),
        "dataset_bin_sampled_sha256": _sampled_file_digest(Path(str(args.data_prefix) + ".bin")),
        "dataset_idx_sha256": _sampled_file_digest(Path(str(args.data_prefix) + ".idx")),
        "tokenizer_path": str(args.tokenizer_path),
        "metrics_path": str(args.output_dir / "metrics.jsonl"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "activation_recomputation": "disabled_for_differentiable_cost",
    }
    (args.output_dir / f"manifest_phase_{args.phase_start_step}_{args.phase_end_step}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # In eval-only, learned cost variants should be evaluated as learned routing
    # without an active cost term. Positional vanilla-like variants must keep
    # their identity, otherwise the forward step incorrectly expects MoSAR
    # auxiliary states.
    state_trajectory = (
        "mosar0" if args.eval_only and args.trajectory in {"cost_fixed", "cost_warmup"}
        else args.trajectory
    )
    state = MoSARObjectiveState(
        trajectory=state_trajectory,
        phase_start_step=args.phase_start_step,
        microbatches_per_step=args.global_batch_size // args.micro_batch_size,
        warmup_steps=args.cost_warmup_steps,
        max_cost_weight=args.max_cost_weight,
        metrics_path=args.output_dir / "metrics.jsonl",
    )
    forward_step = make_mosar_forward_step(state)

    print("===== MoSAR training/evaluation phase =====")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    pretrain(config=cfg, forward_step_func=forward_step)
    print(
        "MOSAR PHASE PASSED "
        f"trajectory={args.trajectory} start={args.phase_start_step} end={args.phase_end_step}"
    )


if __name__ == "__main__":
    main()
