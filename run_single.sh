#!/usr/bin/env bash
# Run a single-domain RecBole baseline (BPR/NeuMF/NGCF/LightGCN/SASRec) on KLM3 live domain.
#
# Usage:
#   bash run_single.sh <MODEL> <GPU_ID> [extra_args...]
#
# Examples:
#   bash run_single.sh BPR 0
#   bash run_single.sh LightGCN 3 --n_layers=2
#   bash run_single.sh NGCF 5 --learning_rate=1e-3 --weight_decay=1e-5
#
# Notes:
#   - Reuses RecBole-CDR/dataset/klm3_live/klm3_live.inter (produced by preprocess_klm3.py).
#     No separate preprocessing needed — run preprocess_klm3.py once for CDR, then this.
#   - RecBole CLI requires --key=value (equals-connected); space-separated args are ignored.
#   - Uses base RecBole's run_recbole (not the CDR runner) since these are single-domain.

set -uo pipefail

MODEL="${1:?Usage: $0 <MODEL> <GPU_ID> [extra_args...]}"
GPU_ID="${2:?Usage: $0 <MODEL> <GPU_ID> [extra_args...]}"
shift 2 || true   # remaining args forwarded to recbole

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/RecBole"

ARGS=(
    --model="$MODEL"
    --dataset=klm3_live
    --config_files="$SCRIPT_DIR/config/klm3_live_single.yaml"
    --gpu_id=0
    --train_batch_size=4096
    --show_progress=False
)

echo "================================================================"
echo " Single-domain  : $MODEL on KLM3 live domain"
echo " GPU            : $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID, process-internal cuda:0)"
echo "================================================================"

# CUDA_VISIBLE_DEVICES masks to only the requested physical GPU; inside the process
# that GPU appears as index 0 (hence --gpu_id 0 above). Thread limits prevent
# multi-process CPU contention.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
    OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
    NUMEXPR_NUM_THREADS=12 \
    python -u run_recbole.py "${ARGS[@]}" "$@"
