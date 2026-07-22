#!/usr/bin/env bash
# Usage: bash SAQRec/run_baseline.sh <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <data_dir> <gpu_ids> [extra args]
# Examples: ... 3       (single GPU)   /   ... 3,4,5,6,7 (five GPUs)
set -euo pipefail

MODEL="${1:?Usage: $0 <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <data_dir> <gpu_ids> [extra args]}"
DATA_DIR="${2:?Usage: $0 <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <data_dir> <gpu_ids> [extra args]}"
GPU_IDS="${3:?Usage: $0 <Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec> <data_dir> <gpu_ids> [extra args]}"
shift 3

case "$MODEL" in Caser|DFN|DMT|FeedRec|FMLPRec|FMLPRecM|GRU4Rec|GRU4RecM|HGN|NARM|SASRec|SASRecM|SAQRec) ;; *) echo "unknown model: $MODEL" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SAQRec requires its Base → Propensity → Satisfaction teacher chain. Keep the
# same convenient entry point, but hand it to the dedicated pipeline runner.
if [[ "$MODEL" == "SAQRec" ]]; then
  echo "SAQRec selected: dispatching to Base → Propensity → Satisfaction → SAQRec pipeline."
  DATA_DIR="$DATA_DIR" bash "$ROOT/SAQRec/run_tune_baseline.sh" SAQRec "$GPU_IDS" "$@"
  exit $?
fi
# Defaults for a normal single training run. Extra arguments at the end can
# override any of them, e.g. --epochs 50 --dim 128.
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
DIM="${DIM:-64}"
DROPOUT="${DROPOUT:-0.2}"
NUM_NEGS="${NUM_NEGS:-2}"
REC_LEN="${REC_LEN:-50}"
FEEDBACK_LEN="${FEEDBACK_LEN:-100}"
SATIS_LEN="${SATIS_LEN:-20}"
DISSATIS_LEN="${DISSATIS_LEN:-10}"
NUM_HEADS="${NUM_HEADS:-2}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
DISENTANGLE_WEIGHT="${DISENTANGLE_WEIGHT:-0.1}"
KS_VALUES="${KS_VALUES:-[1,5,10,20]}"
SELECTION_METRIC="${SELECTION_METRIC:-ndcg@10}"
PATIENCE="${PATIENCE:-10}"
TQDM="${TQDM:-1}"
CPU_FLAG=""
[[ "$USE_CPU" == "1" ]] && CPU_FLAG="--cpu"
TQDM_FLAG=""
[[ "$TQDM" == "0" ]] && TQDM_FLAG="--no_tqdm"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_LIST[@]}"
[[ "$NUM_GPUS" -gt 0 ]] || { echo "at least one GPU id is required" >&2; exit 2; }
for GPU in "${GPU_LIST[@]}"; do
  [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "invalid GPU list: $GPU_IDS" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
case "$MODEL" in
  DFN|DMT|FeedRec|FMLPRecM|GRU4RecM|SASRecM)
    if [[ ! -f "$DATA_DIR/feedrec_events.parquet" ]]; then
      echo "Preparing independent multi-behavior data: $DATA_DIR/feedrec_events.parquet"
      python "$ROOT/SAQRec/prepare_multibehavior_data.py" --data_dir "$DATA_DIR"
    fi
    ;;
esac
echo "Visible GPUs: $GPU_IDS (DDP processes: $NUM_GPUS)"
if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCH=(torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
else
  LAUNCH=(python -u)
fi
PYTHONUNBUFFERED=1 "${LAUNCH[@]}" "$ROOT/SAQRec/run.py" \
  --stage baseline --model "$MODEL" --data_dir "$DATA_DIR" \
  --work_dir "$ROOT/SAQRec/outputs/$MODEL" \
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" --seed "$SEED" --dim "$DIM" --dropout "$DROPOUT" --num_negs "$NUM_NEGS" \
  --rec_len "$REC_LEN" --feedback_len "$FEEDBACK_LEN" --satis_len "$SATIS_LEN" --dissatis_len "$DISSATIS_LEN" \
  --num_heads "$NUM_HEADS" --num_blocks "$NUM_BLOCKS" --num_experts "$NUM_EXPERTS" --ks "$KS_VALUES" \
  --disentangle_weight "$DISENTANGLE_WEIGHT" \
  --selection_metric "$SELECTION_METRIC" --patience "$PATIENCE" ${CPU_FLAG:+"$CPU_FLAG"} ${TQDM_FLAG:+"$TQDM_FLAG"} "$@"
