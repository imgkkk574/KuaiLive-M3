#!/usr/bin/env bash
# Hyperparameter tuning for one RecBole-CDR model on one GPU.
# Runs all lr × weight_decay × n_layers combinations serially.
#
# Usage:
#   bash run_tune.sh <MODEL> <GPU_ID>
#
# Examples:
#   bash run_tune.sh CMF 0
#   bash run_tune.sh EMCDR 3
#
# Outputs:
#   log_tune/<MODEL>/lr<X>_wd<Y>_layer<Z>.log   per-run log
#   log_tune/<MODEL>/summary.csv                  results table

set -uo pipefail

MODEL="${1:?Usage: $0 <MODEL> <GPU_ID>}"
GPU_ID="${2:?Usage: $0 <MODEL> <GPU_ID>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Single-domain baselines (base RecBole) vs CDR models (RecBole-CDR) ──────────
# Single-domain models run on the live domain only; CDR models cross live↔photo.
is_single_domain() {
    case "$1" in
        BPR|NeuMF|NGCF|LightGCN|SASRec) return 0 ;;
        *)                                return 1 ;;
    esac
}

if is_single_domain "$MODEL"; then
    RUNNER_DIR="$SCRIPT_DIR/RecBole"
    RUNNER="run_recbole.py"
    CONFIG_FILE="$SCRIPT_DIR/config/klm3_live_single.yaml"
else
    RUNNER_DIR="$SCRIPT_DIR/RecBole-CDR"
    RUNNER="run_recbole_cdr.py"
    CONFIG_FILE="$SCRIPT_DIR/config/klm3.yaml"
fi
cd "$RUNNER_DIR"

# ── Fixed hyperparameters (per paper) ────────────────────────────────────────
readonly EMB_SIZE=64
readonly BATCH_SIZE=2048

# ── Search space ─────────────────────────────────────────────────────────────
LR_LIST=(1e-3 5e-3 1e-4 5e-4 1e-5)
WD_LIST=(1e-5 1e-6 1e-7)
N_LAYER_LIST=(1 2 3)

# ── Model-specific layer parameter ───────────────────────────────────────────
# Returns the CLI arg name for the layer parameter, or "" if not applicable.
layer_param_name() {
    case "$1" in
        BiTGCF|MGCCDR|DisenCDR|NGCF|LightGCN) echo "n_layers" ;;
        DTCDR|CoNet|EMCDR|SSCDR|DCDCSR|NeuMF)  echo "mlp_hidden_size" ;;
        *)                                      echo "" ;;   # CMF,CLFM,DeepAPF,NATR,BPR,SASRec
    esac
}

# Build "[64,...,64]" with n repetitions for mlp_hidden_size
make_hidden_size() {
    local n=$1 i result="["
    for ((i = 0; i < n; i++)); do
        [[ $i -gt 0 ]] && result+=","
        result+="${EMB_SIZE}"
    done
    echo "${result}]"
}

# ── Setup ─────────────────────────────────────────────────────────────────────
LAYER_PARAM=$(layer_param_name "$MODEL")
LOG_DIR="../log_tune/${MODEL}"
mkdir -p "$LOG_DIR"
# Timestamped summary so reruns never overwrite previous results.
RUN_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/summary_${RUN_TS}.csv"
echo "model,lr,weight_decay,n_layers,recall@10,recall@20,recall@40,ndcg@10,ndcg@20,ndcg@40,mrr@10,test_ndcg@10,status" > "$SUMMARY"
ln -sf "summary_${RUN_TS}.csv" "${LOG_DIR}/latest.csv"   # convenience pointer

if [[ -z "$LAYER_PARAM" ]]; then
    TOTAL=$(( ${#LR_LIST[@]} * ${#WD_LIST[@]} ))
else
    TOTAL=$(( ${#LR_LIST[@]} * ${#WD_LIST[@]} * ${#N_LAYER_LIST[@]} ))
fi

COMBO=0
echo "================================================================"
echo " Model  : $MODEL"
echo " GPU    : $GPU_ID"
echo " Combos : $TOTAL"
echo " Log dir: $LOG_DIR"
echo " Summary: summary_${RUN_TS}.csv (also via latest.csv)"
echo "================================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
for LR in "${LR_LIST[@]}"; do
  for WD in "${WD_LIST[@]}"; do

    if [[ -z "$LAYER_PARAM" ]]; then
        LAYER_VARIANTS=("none")
    else
        LAYER_VARIANTS=("${N_LAYER_LIST[@]}")
    fi

    for LAYER in "${LAYER_VARIANTS[@]}"; do
        COMBO=$(( COMBO + 1 ))

        # Build argument array. Inside the process only 1 GPU is visible
        # (masked by CUDA_VISIBLE_DEVICES set below), so --gpu_id 0.
        # RecBole requires --key=value (equals-connected) form; space-separated
        # args are silently ignored (see "will not be used in RecBole" warning).
        ARGS=(
            --model="$MODEL"
            --config_files="$CONFIG_FILE"
            --gpu_id=0
            --learning_rate="$LR"
            --weight_decay="$WD"
            --train_batch_size="$BATCH_SIZE"
            --show_progress=False
        )
        # Base RecBole's run_recbole.py defaults --dataset to 'ml-100k' (non-None),
        # which would override the 'dataset' key in the config file. Pass it
        # explicitly for single-domain models so they use klm3_live.
        if is_single_domain "$MODEL"; then
            ARGS+=(--dataset=klm3_live)
        fi

        if [[ "$LAYER" != "none" ]]; then
            if [[ "$LAYER_PARAM" == "n_layers" ]]; then
                ARGS+=(--n_layers="$LAYER")
            else
                HIDDEN=$(make_hidden_size "$LAYER")
                ARGS+=(--"$LAYER_PARAM"="$HIDDEN")
            fi
        fi

        LOG_FILE="${LOG_DIR}/lr${LR}_wd${WD}_layer${LAYER}_$(date +%Y%m%d_%H%M%S).log"
        echo ""
        echo "[$(date '+%H:%M:%S')] [$COMBO/$TOTAL] lr=${LR}  wd=${WD}  layers=${LAYER}"
        echo "  log: $LOG_FILE"
        echo "  Physical GPU ${GPU_ID} (CUDA_VISIBLE_DEVICES=${GPU_ID}, process-internal cuda:0)"

        # Merge stdout+stderr, filter tqdm progress lines, tee to log & terminal.
        # CUDA_VISIBLE_DEVICES masks to only the requested physical GPU; inside
        # the process that GPU appears as index 0 (hence --gpu_id 0 in ARGS).
        # OMP/MKL thread limits prevent multi-process CPU thrashing.
        # 72-core box, 6 parallel jobs → 12 threads/proc = full utilization.
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
            OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
            NUMEXPR_NUM_THREADS=12 \
            python -u "$RUNNER" "${ARGS[@]}" 2>&1 \
            | grep --line-buffered -v "it/s" \
            | tee "$LOG_FILE"
        STATUS=$( [[ ${PIPESTATUS[0]} -eq 0 ]] && echo "OK" || echo "FAILED" )

        # Extract metrics from the "best valid" block in the log
        # Log format: OrderedDict([('recall@10', 0.0807), ('mrr@10', 0.1316), ...])
        extract_metric() {
            local key=$1
            grep "best valid" "$LOG_FILE" \
                | grep -oP "'${key}', \K[0-9.]+" | head -1
        }
        extract_test() {
            local key=$1
            grep "test result" "$LOG_FILE" \
                | grep -oP "'${key}', \K[0-9.]+" | head -1
        }
        R10=$(extract_metric "recall@10");  R10="${R10:-N/A}"
        R20=$(extract_metric "recall@20");  R20="${R20:-N/A}"
        R40=$(extract_metric "recall@40");  R40="${R40:-N/A}"
        N10=$(extract_metric "ndcg@10");    N10="${N10:-N/A}"
        N20=$(extract_metric "ndcg@20");    N20="${N20:-N/A}"
        N40=$(extract_metric "ndcg@40");    N40="${N40:-N/A}"
        MRR=$(extract_metric "mrr@10");     MRR="${MRR:-N/A}"
        TNDCG=$(extract_test  "ndcg@10");   TNDCG="${TNDCG:-N/A}"

        echo "    → ${STATUS}  Recall@10=${R10}  NDCG@10=${N10}  MRR@10=${MRR}  testNDCG@10=${TNDCG}"
        echo "${MODEL},${LR},${WD},${LAYER},${R10},${R20},${R40},${N10},${N20},${N40},${MRR},${TNDCG},${STATUS}" >> "$SUMMARY"
    done
  done
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Done: $MODEL  |  Results → ${SUMMARY}"
echo "================================================================"
echo "Top 5 by valid NDCG@10:"
{ head -1 "$SUMMARY"; \
  tail -n +2 "$SUMMARY" | grep -v "N/A" | sort -t',' -k7 -rn; } \
  | head -6 \
  | column -t -s','
