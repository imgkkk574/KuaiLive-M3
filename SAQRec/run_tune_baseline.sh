#!/usr/bin/env bash
# Serial grid search for one KLM3-SA QRec sequential baseline on one GPU.
#
# Usage: bash SAQRec/run_tune_baseline.sh <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <gpu_ids> [extra run.py args]
# Examples: ... 3       (single GPU)   /   ... 3,4,5,6,7 (five GPUs)
# Example: bash SAQRec/run_tune_baseline.sh SASRec 0 --epochs 30
#
# Configuration lives below. Every value can still be overridden with an
# environment variable; extra run.py arguments passed after GPU_IDS win last.
set -uo pipefail

MODEL="${1:?Usage: $0 <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <gpu_ids> [extra run.py args]}"
GPU_IDS="${2:?Usage: $0 <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <gpu_ids> [extra run.py args]}"
shift 2

case "$MODEL" in Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec) ;; *) echo "unknown model: $MODEL" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_LIST[@]}"
[[ "$NUM_GPUS" -gt 0 ]] || { echo "at least one GPU id is required" >&2; exit 2; }
for GPU in "${GPU_LIST[@]}"; do
    [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "invalid GPU list: $GPU_IDS" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# ── Data, output, and runtime ───────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/klm3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/tune}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/log_tune}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-}"
if [[ -z "$MASTER_PORT" ]]; then
    MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi
SEED="${SEED:-1}"
USE_CPU="${USE_CPU:-0}"
TQDM="${TQDM:-1}"

# ── Shared model and sampling settings ──────────────────────────────────────
DIM="${DIM:-64}"
DROPOUT="${DROPOUT:-0.2}"
NUM_NEGS="${NUM_NEGS:-2}"
REC_LEN="${REC_LEN:-50}"
FEEDBACK_LEN="${FEEDBACK_LEN:-100}"
SATIS_LEN="${SATIS_LEN:-20}"
DISSATIS_LEN="${DISSATIS_LEN:-10}"
NUM_INTEREST="${NUM_INTEREST:-8}"
NUM_HEADS="${NUM_HEADS:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
DISENTANGLE_WEIGHT="${DISENTANGLE_WEIGHT:-0.1}"
PROPENSITY_CLAMP="${PROPENSITY_CLAMP:-0.05}"
SATISFACTION_WEIGHT="${SATISFACTION_WEIGHT:-0.01}"
CORRECTION_AFTER="${CORRECTION_AFTER:-3}"

# ── Validation, early stopping, and search space ────────────────────────────
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-$EPOCHS}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-1e-5}"
KS_VALUES="${KS_VALUES:-[1,5,10,20]}"
SELECTION_METRIC="${SELECTION_METRIC:-ndcg@10}"
PATIENCE="${PATIENCE:-10}"

KS_CSV="$(python -c 'import sys; print(",".join(str(x) for x in sorted({int(x) for x in sys.argv[1].strip().strip("[]").replace(" ", "").split(",") if x})))' "$KS_VALUES")"
IFS=',' read -r -a K_LIST <<< "$KS_CSV"
METRIC_KEYS=(mrr)
for K in "${K_LIST[@]}"; do
    METRIC_KEYS+=("hr@$K" "ndcg@$K")
done

COMMON_ARGS=(
    --eval_batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" --seed "$SEED"
    --dim "$DIM" --dropout "$DROPOUT" --num_negs "$NUM_NEGS"
    --rec_len "$REC_LEN" --satis_len "$SATIS_LEN" --dissatis_len "$DISSATIS_LEN"
    --feedback_len "$FEEDBACK_LEN"
    --num_interest "$NUM_INTEREST" --num_heads "$NUM_HEADS"
    --num_experts "$NUM_EXPERTS"
    --disentangle_weight "$DISENTANGLE_WEIGHT"
    --propensity_clamp "$PROPENSITY_CLAMP" --satisfaction_weight "$SATISFACTION_WEIGHT"
    --correction_after "$CORRECTION_AFTER"
)
[[ "$USE_CPU" == "1" ]] && COMMON_ARGS+=(--cpu)
[[ "$TQDM" == "0" ]] && COMMON_ARGS+=(--no_tqdm)

if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCH=(torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
else
    LAUNCH=(python -u)
fi

# Keep the default search compact: nine trials for single-block models and
# 27 trials for the configurable SASRec/FMLPRec encoders.
read -r -a LR_LIST <<< "${LR_VALUES:-1e-3 5e-4 1e-4}"
read -r -a WD_LIST <<< "${WD_VALUES:-0 1e-6 1e-5}"
if [[ "$MODEL" == "FeedRec" || "$MODEL" == "SASRec" || "$MODEL" == "FMLPRec" || \
      "$MODEL" == "SASRecM" || "$MODEL" == "FMLPRecM" || \
      "$MODEL" == "DFN" || "$MODEL" == "DMT" ]]; then
    read -r -a BLOCK_LIST <<< "${BLOCK_VALUES:-1 2 3}"
else
    BLOCK_LIST=(none)
fi

LOG_DIR="$LOG_ROOT/$MODEL"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/$MODEL"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/summary_${RUN_TS}.csv"
HEADER='model,lr,weight_decay,num_blocks,best_epoch'
for KEY in "${METRIC_KEYS[@]}"; do HEADER+=",val_${KEY}"; done
for KEY in "${METRIC_KEYS[@]}"; do HEADER+=",test_${KEY}"; done
HEADER+=',status,run_dir,log_file'
printf '%s\n' "$HEADER" > "$SUMMARY"
ln -sfn "summary_${RUN_TS}.csv" "$LOG_DIR/latest.csv"

TOTAL=$(( ${#LR_LIST[@]} * ${#WD_LIST[@]} * ${#BLOCK_LIST[@]} ))
COMBO=0
echo "Model=$MODEL visible_gpus=$GPU_IDS ddp_processes=$NUM_GPUS trials=$TOTAL data=$DATA_DIR"
echo "Validation selection: $SELECTION_METRIC; cutoffs: $KS_CSV"
echo "Summary: $SUMMARY"

case "$MODEL" in
    DFN|DMT|FeedRec|FMLPRecM|GRU4RecM|SASRecM)
        if [[ ! -f "$DATA_DIR/feedrec_events.parquet" ]]; then
            echo "Preparing independent multi-behavior data: $DATA_DIR/feedrec_events.parquet"
            python "$SCRIPT_DIR/prepare_multibehavior_data.py" --data_dir "$DATA_DIR" || exit 1
        fi
        ;;
esac

metric_row() {
    local metrics_file=$1
    python -c 'import json, sys; d=json.load(open(sys.argv[1])); print(",".join(str(d[k]) for k in sys.argv[2:]))' "$metrics_file" "${METRIC_KEYS[@]}"
}

metric_na_row() {
    local result='' _
    for _ in "${METRIC_KEYS[@]}"; do result+="N/A,"; done
    printf '%s' "${result%,}"
}

SELECT_COLUMN=0
for INDEX in "${!METRIC_KEYS[@]}"; do
    if [[ "${METRIC_KEYS[$INDEX]}" == "$SELECTION_METRIC" ]]; then
        SELECT_COLUMN=$((6 + INDEX))
        break
    fi
done
[[ $SELECT_COLUMN -gt 0 ]] || { echo "SELECTION_METRIC must be one of: ${METRIC_KEYS[*]}" >&2; exit 2; }

if [[ "$MODEL" == "SAQRec" ]]; then
    # The three upstream stages are independent of SAQRec's final lr/wd grid.
    # Train them once and share their frozen checkpoints across all trials.
    TEACHER_DIR="$OUTPUT_ROOT/SAQRec/teacher_${RUN_TS}"
    TEACHER_LOG="$LOG_DIR/teacher_${RUN_TS}.log"
    mkdir -p "$TEACHER_DIR"
    : > "$TEACHER_LOG"
    echo "Preparing shared SAQRec teacher chain: $TEACHER_DIR"
    for STAGE in base propensity satisfaction; do
        STAGE_DIR="$TEACHER_DIR/$STAGE"
        STAGE_ARGS=(--stage "$STAGE" --data_dir "$DATA_DIR" --work_dir "$STAGE_DIR"
                    --epochs "$PRETRAIN_EPOCHS" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
                    --lr "$PRETRAIN_LR" --weight_decay "$PRETRAIN_WEIGHT_DECAY")
        if [[ "$STAGE" == "propensity" || "$STAGE" == "satisfaction" ]]; then
            STAGE_ARGS+=(--base_ckpt "$TEACHER_DIR/base/best.pt")
        fi
        [[ "$STAGE" == "satisfaction" ]] && STAGE_ARGS+=(--propensity_ckpt "$TEACHER_DIR/propensity/best.pt")
        # Forward architecture overrides (for example --dim) to keep all
        # checkpoint shapes consistent with the final SAQRec trial.
        STAGE_ARGS+=("${COMMON_ARGS[@]}" "$@")
        STAGE_ARGS+=(--ks "$KS_VALUES" --selection_metric "$SELECTION_METRIC" --patience "$PATIENCE")
        echo "[teacher] $STAGE"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS" \
            OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 \
            PYTHONUNBUFFERED=1 "${LAUNCH[@]}" "$SCRIPT_DIR/run.py" "${STAGE_ARGS[@]}" 2>&1 | tee -a "$TEACHER_LOG"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "Teacher stage failed: $STAGE (see $TEACHER_LOG)" >&2
            exit 1
        fi
    done
fi

for LR in "${LR_LIST[@]}"; do
    for WD in "${WD_LIST[@]}"; do
        for BLOCKS in "${BLOCK_LIST[@]}"; do
            COMBO=$((COMBO + 1))
            TAG="lr${LR}_wd${WD}_blocks${BLOCKS}_${RUN_TS}"
            RUN_DIR="$OUTPUT_ROOT/$MODEL/$TAG"
            LOG_FILE="$LOG_DIR/$TAG.log"
            if [[ "$MODEL" == "SAQRec" ]]; then
                ARGS=(
                    --stage saqrec --data_dir "$DATA_DIR" --work_dir "$RUN_DIR"
                    --base_ckpt "$TEACHER_DIR/base/best.pt"
                    --satisfaction_ckpt "$TEACHER_DIR/satisfaction/best.pt"
                )
            else
                ARGS=(--stage baseline --model "$MODEL" --data_dir "$DATA_DIR" --work_dir "$RUN_DIR")
            fi
            ARGS+=(
                --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS"
                --lr "$LR" --weight_decay "$WD"
            )
            [[ "$BLOCKS" != "none" ]] && ARGS+=(--num_blocks "$BLOCKS")
            ARGS+=("${COMMON_ARGS[@]}" "$@")
            ARGS+=(--ks "$KS_VALUES" --selection_metric "$SELECTION_METRIC" --patience "$PATIENCE")

            echo "[$COMBO/$TOTAL] lr=$LR wd=$WD blocks=$BLOCKS"
            echo "  log: $LOG_FILE"
            set +e
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS" \
                OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 \
                PYTHONUNBUFFERED=1 "${LAUNCH[@]}" "$SCRIPT_DIR/run.py" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
            STATUS_CODE=${PIPESTATUS[0]}
            set -e

            if [[ $STATUS_CODE -eq 0 && -f "$RUN_DIR/metrics.json" && -f "$RUN_DIR/test_metrics.json" ]]; then
                if BEST_EPOCH="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["epoch"])' "$RUN_DIR/metrics.json")" && \
                   VAL_METRICS="$(metric_row "$RUN_DIR/metrics.json")" && \
                   TEST_METRICS="$(metric_row "$RUN_DIR/test_metrics.json")"; then
                    STATUS=OK
                else
                    echo "Unable to read metrics for trial $COMBO; marking it failed." >&2
                    STATUS=FAILED
                    BEST_EPOCH=N/A
                    VAL_METRICS="$(metric_na_row)"
                    TEST_METRICS="$(metric_na_row)"
                fi
            else
                STATUS=FAILED
                BEST_EPOCH=N/A
                VAL_METRICS="$(metric_na_row)"
                TEST_METRICS="$(metric_na_row)"
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$MODEL" "$LR" "$WD" "$BLOCKS" "$BEST_EPOCH" "$VAL_METRICS" "$TEST_METRICS" \
                "$STATUS" "$RUN_DIR" "$LOG_FILE" >> "$SUMMARY"
        done
    done
done

echo "Done. Results: $SUMMARY"
echo "Top 5 by validation ${SELECTION_METRIC}:"
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | grep ',OK,' | sort -t',' -k${SELECT_COLUMN},${SELECT_COLUMN}gr; } | head -6 | column -t -s','
