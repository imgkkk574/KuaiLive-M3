#!/usr/bin/env bash
# Run KLM3 CDR pipeline: preprocess raw data then train + evaluate with RecBole-CDR.
#
# Usage:
#   bash run_klm3_cdr.sh <klm3_data_dir> [model] [extra_args...]
#
# Examples:
#   bash run_klm3_cdr.sh /data/klm3                        # CMF, full dataset
#   bash run_klm3_cdr.sh /data/klm3 EMCDR                  # EMCDR model
#   bash run_klm3_cdr.sh /data/klm3 CMF --use_gpu False    # force CPU
#   bash run_klm3_cdr.sh /data/klm3 CMF --sample_ratio 0.01  # quick dev run
#
# Notes:
#   - Preprocessing is skipped if output .inter files already exist.
#     Delete RecBole-CDR/dataset/klm3_*/  to force re-preprocessing.
#   - Pass --sample_ratio <0..1> and/or --min_interactions <N> to forward to
#     preprocess_klm3.py (these are consumed before the remaining args are
#     passed to run_recbole_cdr.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDR_DIR="$SCRIPT_DIR/RecBole-CDR"

KLM3_DATA_DIR="${1:?Usage: $0 <klm3_data_dir> [model] [extra_args...]}"
MODEL="${2:-CMF}"
shift 2 || shift 1 2>/dev/null || true   # remaining args forwarded to recbole

# Split remaining args: preprocess args vs recbole args
PREPROCESS_ARGS=()
RECBOLE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample_ratio|--min_interactions|--seed)
            PREPROCESS_ARGS+=("$1" "$2"); shift 2 ;;
        *)
            RECBOLE_ARGS+=("$1"); shift ;;
    esac
done

LIVE_INTER="$CDR_DIR/dataset/klm3_live/klm3_live.inter"
PHOTO_INTER="$CDR_DIR/dataset/klm3_photo/klm3_photo.inter"

# Step 1: Preprocess (skip if outputs already exist)
if [[ -f "$LIVE_INTER" && -f "$PHOTO_INTER" ]]; then
    echo "[preprocess] Output files already exist — skipping. Delete dataset/klm3_*/ to re-run."
else
    echo "[preprocess] Generating RecBole-CDR .inter files from $KLM3_DATA_DIR ..."
    python "$SCRIPT_DIR/preprocess_klm3.py" \
        --data_dir "$KLM3_DATA_DIR" \
        "${PREPROCESS_ARGS[@]}"
fi

# Step 2: Train + evaluate
echo ""
echo "[train] Model=$MODEL  Config=$SCRIPT_DIR/config/klm3.yaml"
cd "$CDR_DIR"
python run_recbole_cdr.py \
    --model "$MODEL" \
    --config_files "$SCRIPT_DIR/config/klm3.yaml" \
    "${RECBOLE_ARGS[@]}"
