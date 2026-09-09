#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -f "$REPO_ROOT/configs/main_constants.env" ]]; then
  set -a; source "$REPO_ROOT/configs/main_constants.env"; set +a
fi

SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"
POST_RUN_ROOT="${POST_RUN_ROOT:-$SCRATCH_ROOT/mosar_posttraining_v1}"
POST_LOG_ROOT="${POST_LOG_ROOT:-$POST_RUN_ROOT/logs}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/slurm/posttraining/run_posttraining_eval_lprod_scratch.slurm}"
GROUP="${GROUP:-anonymous_main_v1}"
STEP="${STEP:?missing STEP}"
TOTAL_STEPS="${TOTAL_STEPS:-610000}"
MODEL_SEED="${MODEL_SEED:-1239}"
DATA_SEED="${DATA_SEED:-2031}"
EVAL_ITERS="${EVAL_ITERS:-64}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
ROUTING_INFERENCE="${ROUTING_INFERENCE:-soft}"
EVAL_RESET_CONSUMED_SAMPLES="${EVAL_RESET_CONSUMED_SAMPLES:-0}"
DEFAULT_RUNS="vanilla alibi prope075 s_fixed m_fixed mosar0 cost001 rope_m_mask"
RUNS_STRING="${RUNS:-$DEFAULT_RUNS}"
read -r -a RUN_ARRAY <<< "$RUNS_STRING"

JOBID_DIR="$POST_RUN_ROOT/housekeeping/$GROUP/eval_jobids/step_${STEP}/seq_${SEQUENCE_LENGTH}/${ROUTING_INFERENCE}"
mkdir -p "$JOBID_DIR" "$POST_LOG_ROOT"
bash -n "$SLURM_SCRIPT"
make_sbatch_args() {
  SBATCH_ARGS=()
  [[ -n "${SBATCH_ACCOUNT:-}" ]] && SBATCH_ARGS+=(--account="$SBATCH_ACCOUNT")
  [[ -n "${SBATCH_PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="$SBATCH_PARTITION")
  [[ -n "${SBATCH_QOS:-}" ]] && SBATCH_ARGS+=(--qos="$SBATCH_QOS")
  if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( $SBATCH_EXTRA_ARGS )
    SBATCH_ARGS+=("${EXTRA_ARGS[@]}")
  fi
}
make_sbatch_args

echo "===== SUBMIT MOSAR POSTTRAINING EVAL ====="
echo "GROUP=$GROUP STEP=$STEP SEQUENCE_LENGTH=$SEQUENCE_LENGTH ROUTING_INFERENCE=$ROUTING_INFERENCE"
echo "RUNS=${RUN_ARRAY[*]}"

for RUN_ID in "${RUN_ARRAY[@]}"; do
  JOBID=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    --job-name="posteval_${RUN_ID}_${STEP}_s${SEQUENCE_LENGTH}_${ROUTING_INFERENCE}" \
    --output="$POST_LOG_ROOT/${GROUP}_posteval_step_${STEP}_seq_${SEQUENCE_LENGTH}_${ROUTING_INFERENCE}_${RUN_ID}_%j.out" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,RUN_ID=$RUN_ID,GROUP=$GROUP,STEP=$STEP,TOTAL_STEPS=$TOTAL_STEPS,MODEL_SEED=$MODEL_SEED,DATA_SEED=$DATA_SEED,EVAL_ITERS=$EVAL_ITERS,SEQUENCE_LENGTH=$SEQUENCE_LENGTH,ROUTING_INFERENCE=$ROUTING_INFERENCE,EVAL_RESET_CONSUMED_SAMPLES=$EVAL_RESET_CONSUMED_SAMPLES" \
    "$SLURM_SCRIPT")
  JOBID="${JOBID%%;*}"
  echo "$JOBID" > "$JOBID_DIR/${RUN_ID}.jobid"
  echo "$RUN_ID -> $JOBID"
done
