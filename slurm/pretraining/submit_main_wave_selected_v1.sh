#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [[ -f "$REPO_ROOT/configs/main_constants.env" ]]; then
  set -a; source "$REPO_ROOT/configs/main_constants.env"; set +a
fi

SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_ROOT/mosar_main_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/slurm/pretraining/run_main_segment_lprod_scratch.slurm}"

GROUP="${GROUP:-anonymous_main_v1}"
MODEL_SEED="${MODEL_SEED:-1239}"
DATA_SEED="${DATA_SEED:-2031}"
START="${START:?missing START}"
END="${END:?missing END}"
TOTAL_STEPS="${TOTAL_STEPS:-610000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
RUNS="${RUNS:?missing RUNS, e.g. RUNS='s_fixed cost001'}"

JOBID_DIR="$RUN_ROOT/housekeeping/$GROUP/jobids"
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

for RUN_ID in $RUNS; do
  case "$RUN_ID" in
    vanilla|alibi|prope075|s_fixed|m_fixed|mosar0|cost001|rope_m_mask) ;;
    *) echo "Invalid RUN_ID=$RUN_ID" >&2; exit 2;;
  esac

  CKPT="$CHECKPOINT_ROOT/$GROUP/$RUN_ID/latest_checkpointed_iteration.txt"
  if [[ "$START" != "0" ]]; then
    [[ -f "$CKPT" ]] || { echo "Missing checkpoint marker for $RUN_ID: $CKPT" >&2; exit 3; }
    latest=$(tr -d '[:space:]' < "$CKPT")
    [[ "$latest" == "$START" ]] || { echo "Refusing $RUN_ID: latest=$latest START=$START" >&2; exit 4; }
  fi

  JOBID=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    --job-name="main_${RUN_ID}_${START}_${END}" \
    --output="$LOG_ROOT/${GROUP}_${RUN_ID}_${START}_${END}_%j.out" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,RUN_ID=$RUN_ID,GROUP=$GROUP,MODEL_SEED=$MODEL_SEED,DATA_SEED=$DATA_SEED,START=$START,END=$END,TOTAL_STEPS=$TOTAL_STEPS,SAVE_INTERVAL=$SAVE_INTERVAL" \
    "$SLURM_SCRIPT")
  JOBID="${JOBID%%;*}"
  echo "$JOBID" > "$JOBID_DIR/${RUN_ID}_${START}_${END}.jobid"
  echo "$RUN_ID -> $JOBID"
done
