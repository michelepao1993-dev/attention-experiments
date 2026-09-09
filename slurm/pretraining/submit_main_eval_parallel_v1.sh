#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -f "$REPO_ROOT/configs/main_constants.env" ]]; then
  set -a; source "$REPO_ROOT/configs/main_constants.env"; set +a
fi

SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_ROOT/mosar_main_v1}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/slurm/pretraining/run_main_eval_lprod_scratch.slurm}"
GROUP="${GROUP:-anonymous_main_v1}"
STEP="${STEP:?missing STEP}"
TOTAL_STEPS="${TOTAL_STEPS:-610000}"
MODEL_SEED="${MODEL_SEED:-1239}"
DATA_SEED="${DATA_SEED:-2031}"
EVAL_ITERS="${EVAL_ITERS:-64}"

RUNS=(vanilla alibi prope075 s_fixed m_fixed mosar0 cost001 rope_m_mask)
JOBID_DIR="$RUN_ROOT/housekeeping/$GROUP/eval_jobids/step_${STEP}"
mkdir -p "$JOBID_DIR" "$LOG_ROOT"
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

for RUN_ID in "${RUNS[@]}"; do
  JOBID=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    --job-name="eval_${RUN_ID}_${STEP}" \
    --output="$LOG_ROOT/${GROUP}_eval_step_${STEP}_${RUN_ID}_%j.out" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,RUN_ID=$RUN_ID,GROUP=$GROUP,STEP=$STEP,TOTAL_STEPS=$TOTAL_STEPS,MODEL_SEED=$MODEL_SEED,DATA_SEED=$DATA_SEED,EVAL_ITERS=$EVAL_ITERS" \
    "$SLURM_SCRIPT")
  JOBID="${JOBID%%;*}"
  echo "$JOBID" > "$JOBID_DIR/${RUN_ID}.jobid"
  echo "$RUN_ID -> $JOBID"
done
