#!/usr/bin/env bash
# Dedicated single-node multi-GPU SAQRec tuning entry point.
#
# Usage:
#   ./SAQRec/run_tune_saqrec_ddp.sh <gpu_ids> [extra run.py args]
#
# Examples:
#   ./SAQRec/run_tune_saqrec_ddp.sh 3,4,5,6,7
#   LR_VALUES='1e-3' WD_VALUES='1e-5' ./SAQRec/run_tune_saqrec_ddp.sh 3,4,5,6,7
#
# Results, teacher checkpoints, per-trial logs, and latest.csv intentionally
# use the same locations as run_tune_baseline.sh.
set -euo pipefail

GPU_IDS="${1:?Usage: $0 <gpu_ids> [extra run.py args]}"
shift

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_LIST[@]}"
[[ "$NUM_GPUS" -gt 1 ]] || { echo "SAQRec DDP requires at least two comma-separated GPU IDs" >&2; exit 2; }
for GPU in "${GPU_LIST[@]}"; do
    [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "invalid GPU list: $GPU_IDS" >&2; exit 2; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── DDP defaults ───────────────────────────────────────────────────────────
# GLOBAL_BATCH_SIZE is preserved as the original single-GPU effective batch.
# BATCH_SIZE is derived per rank unless explicitly overridden.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-$(( (GLOBAL_BATCH_SIZE + NUM_GPUS - 1) / NUM_GPUS ))}"
NUM_WORKERS="${NUM_WORKERS:-4}"              # Per DDP rank; avoid CPU oversubscription.
EPOCHS="${EPOCHS:-100}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-$EPOCHS}"
PATIENCE="${PATIENCE:-10}"
MASTER_PORT="${MASTER_PORT:-}"

# ── Shared output and tuning configuration ──────────────────────────────────
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/klm3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/tune}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/log_tune}"
LR_VALUES="${LR_VALUES:-1e-3 5e-4 1e-4}"
WD_VALUES="${WD_VALUES:-0 1e-6 1e-5}"
KS_VALUES="${KS_VALUES:-[1,5,10,20]}"
SELECTION_METRIC="${SELECTION_METRIC:-ndcg@10}"

echo "SAQRec DDP: physical GPUs=$GPU_IDS, ranks=$NUM_GPUS"
echo "Batch: per-rank=$BATCH_SIZE, effective_global=$((BATCH_SIZE * NUM_GPUS))"
echo "Outputs: $OUTPUT_ROOT/SAQRec"
echo "Logs and summary: $LOG_ROOT/SAQRec"

export DATA_DIR OUTPUT_ROOT LOG_ROOT BATCH_SIZE NUM_WORKERS EPOCHS PRETRAIN_EPOCHS PATIENCE
export MASTER_PORT LR_VALUES WD_VALUES KS_VALUES SELECTION_METRIC
exec bash "$SCRIPT_DIR/run_tune_baseline.sh" SAQRec "$GPU_IDS" "$@"
