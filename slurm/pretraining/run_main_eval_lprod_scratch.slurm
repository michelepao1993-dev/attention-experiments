#!/bin/bash
#SBATCH --job-name=mosar_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=100GB
#SBATCH --output=slurm-%x-%j.out

# Evaluation at the training context length. For length extrapolation and hard routing,
# use slurm/posttraining/run_posttraining_eval_lprod_scratch.slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -f "$REPO_ROOT/configs/main_constants.env" ]]; then
  set -a
  source "$REPO_ROOT/configs/main_constants.env"
  set +a
fi

RUNTIME_SOURCE="${RUNTIME_SOURCE:-$REPO_ROOT/runtime_source}"
SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_ROOT/mosar_main_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/evals}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
DATA_PREFIX="${DATA_PREFIX:?set DATA_PREFIX}"
TOKENIZER_PATH="${TOKENIZER_PATH:?set TOKENIZER_PATH}"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:?set MEGATRON_BRIDGE_PATH}"
CONTAINER_PATH="${CONTAINER_PATH:?set CONTAINER_PATH}"

RUN_ID="${RUN_ID:?missing RUN_ID}"
GROUP="${GROUP:?missing GROUP}"
STEP="${STEP:?missing STEP}"
TOTAL_STEPS="${TOTAL_STEPS:-610000}"
MODEL_SEED="${MODEL_SEED:?missing MODEL_SEED}"
DATA_SEED="${DATA_SEED:?missing DATA_SEED}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
EVAL_ITERS="${EVAL_ITERS:-64}"

case "$RUN_ID" in
  vanilla) TRAJECTORY=vanilla; MAX_WEIGHT=0.0;;
  alibi) TRAJECTORY=alibi; MAX_WEIGHT=0.0;;
  prope075) TRAJECTORY=prope075; MAX_WEIGHT=0.0;;
  s_fixed) TRAJECTORY=s_fixed; MAX_WEIGHT=0.0;;
  m_fixed) TRAJECTORY=m_fixed; MAX_WEIGHT=0.0;;
  mosar0) TRAJECTORY=mosar0; MAX_WEIGHT=0.0;;
  cost001) TRAJECTORY=cost_warmup; MAX_WEIGHT=0.01;;
  rope_m_mask) TRAJECTORY=rope_m_mask; MAX_WEIGHT=0.0;;
  *) echo "invalid RUN_ID=$RUN_ID" >&2; exit 2;;
esac

CKPT="$CHECKPOINT_ROOT/$GROUP/$RUN_ID"
OUT="$EVAL_ROOT/$GROUP/step_${STEP}/$RUN_ID"
MARKERS="$RUN_ROOT/housekeeping/$GROUP/eval_markers/step_${STEP}"
TMP="$RUN_ROOT/tmp/eval_${SLURM_JOB_ID:-manual}_${RUN_ID}_${STEP}"
CACHE="$RUN_ROOT/cache/singularity"
TORCH_CACHE="$RUN_ROOT/cache/torch_eval_${SLURM_JOB_ID:-manual}_${RUN_ID}_${STEP}"
TRITON_CACHE="$RUN_ROOT/cache/triton_eval_${SLURM_JOB_ID:-manual}_${RUN_ID}_${STEP}"
mkdir -p "$OUT" "$MARKERS" "$TMP" "$CACHE" "$TORCH_CACHE" "$TRITON_CACHE" "$LOG_ROOT"
trap 'rm -rf "$TMP" "$TORCH_CACHE" "$TRITON_CACHE"' EXIT

if [[ ! -f "$CKPT/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing checkpoint marker: $CKPT/latest_checkpointed_iteration.txt" >&2
  exit 3
fi
latest=$(tr -d '[:space:]' < "$CKPT/latest_checkpointed_iteration.txt")
if [[ "$latest" != "$STEP" ]]; then
  echo "Expected checkpoint $STEP, found latest=$latest in $CKPT" >&2
  exit 4
fi

export SINGULARITY_TMPDIR="$TMP"
export SINGULARITY_CACHEDIR="$CACHE"
export APPTAINER_TMPDIR="$TMP"
export APPTAINER_CACHEDIR="$CACHE"
export XDG_CACHE_HOME="$TORCH_CACHE"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_CACHE/inductor"
export TRITON_CACHE_DIR="$TRITON_CACHE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONHASHSEED="$MODEL_SEED"

MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 40000))
MASTER_ADDR="$(hostname -s)"
PYTHONPATH_IN="$REPO_ROOT:$RUNTIME_SOURCE:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM"

echo "===== MOSAR MAIN EVAL ====="
echo "run_id=$RUN_ID"
echo "trajectory=$TRAJECTORY"
echo "group=$GROUP"
echo "step=$STEP"
echo "sequence_length=$SEQUENCE_LENGTH"
echo "checkpoint=$CKPT"
echo "out=$OUT"
echo "eval_iters=$EVAL_ITERS"

cd "$RUNTIME_SOURCE"
srun singularity exec --nv \
  --bind "$REPO_ROOT:$REPO_ROOT" \
  --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  --bind "$MEGATRON_BRIDGE_PATH:/opt/Megatron-Bridge" \
  --pwd "$RUNTIME_SOURCE" \
  --env PYTHONPATH="$PYTHONPATH_IN" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  "$CONTAINER_PATH" \
  torchrun --nproc_per_node=1 \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    experiments/pretrain_mosar.py \
      --trajectory "$TRAJECTORY" \
      --eval-only \
      --output-dir "$OUT" \
      --checkpoint-dir "$CKPT" \
      --tokenizer-path "$TOKENIZER_PATH" \
      --data-prefix "$DATA_PREFIX" \
      --phase-start-step "$STEP" \
      --phase-end-step "$STEP" \
      --total-steps "$TOTAL_STEPS" \
      --checkpoint-interval "$STEP" \
      --sequence-length "$SEQUENCE_LENGTH" \
      --vocab-size 256000 \
      --micro-batch-size "$MICRO_BATCH_SIZE" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --learning-rate 3e-4 \
      --min-learning-rate 3e-5 \
      --lr-warmup-steps 100 \
      --cost-warmup-steps 500 \
      --eval-iters "$EVAL_ITERS" \
      --max-cost-weight "$MAX_WEIGHT" \
      --model-seed "$MODEL_SEED" \
      --data-seed "$DATA_SEED"

printf 'job_id=%s\nrun_id=%s\ngroup=%s\nstep=%s\nsequence_length=%s\nhost=%s\nout=%s\nckpt=%s\n' \
  "${SLURM_JOB_ID:-manual}" "$RUN_ID" "$GROUP" "$STEP" "$SEQUENCE_LENGTH" "$(hostname)" "$OUT" "$CKPT" \
  > "$MARKERS/${RUN_ID}.passed"

echo "MOSAR MAIN EVAL PASSED run=$RUN_ID step=$STEP seq=$SEQUENCE_LENGTH eval_iters=$EVAL_ITERS"
