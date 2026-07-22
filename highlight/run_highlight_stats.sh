#!/usr/bin/env bash
# Stats-input baseline: same GRU backbone, but fed leakage-free causal
# interaction statistics instead of segment embeddings.
#
# Purpose: contrast with the embedding runs to show how much highlight
# signal comes from content (embeddings) vs behavior statistics alone.
# Expectation: stats mAP << embedding mAP.
#
# Usage:
#   bash run_highlight_stats.sh <klm3_data_dir> <gpu_list>
#
# Examples:
#   bash run_highlight_stats.sh /data/klm3 0,1,2,3        # 4 GPUs × 4 = 16 concurrent
#   JOBS_PER_GPU=2 bash run_highlight_stats.sh /data/klm3 0,1
#
# Env (overridable):
#   LR_LIST  (default "1e-3 5e-4 1e-4")   → 3
#   WD_LIST  (default "1e-4 1e-5")        → 2
#   EPOCHS   (default 100)
#   JOBS_PER_GPU (default 4)
#   → 1 model (gru) × 3 lr × 2 wd = 6 runs.
#
# Output: highlight_ckpt/gru_stats_lr<lr>_wd<wd>/
# (the _stats_ tag keeps results separate from the embedding runs.)
#
# Resume: a run is skipped if its results_gru.json already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then sed -n '3,22p' "$0" >&2; exit 1; fi

KLM3_DATA_DIR="$1"
GPU_LIST="${2:-0}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

IFS=',' read -ra GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "[error] no GPUs parsed from '$GPU_LIST'" >&2; exit 1; }
for g in "${GPUS[@]}"; do
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "[error] invalid GPU id '$g'" >&2; exit 1; }
done

IFS=' ' read -ra LR_LIST <<< "${LR_LIST:-1e-3 5e-4 1e-4}" || true
IFS=' ' read -ra WD_LIST <<< "${WD_LIST:-1e-4 1e-5}" || true
EPOCHS="${EPOCHS:-100}"
MODEL="gru"                             # fixed: contrast is about input, not architecture
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

[[ ${#LR_LIST[@]} -gt 0 && ${#WD_LIST[@]} -gt 0 ]] || {
    echo "[error] empty LR_LIST or WD_LIST" >&2; exit 1; }

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 1) )); then
    echo "[error] needs bash >= 4.1 (you have ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]})." >&2
    exit 1
fi
N_GPUS=${#GPUS[@]}
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
    echo "[error] JOBS_PER_GPU must be a positive integer" >&2; exit 1; }
N_SLOTS=$(( N_GPUS * JOBS_PER_GPU ))

# Extra train flags (e.g. --max_seq_len 400). --lr/--weight_decay/--input_mode are
# managed here; everything else is forwarded verbatim.
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr|--weight_decay|--input_mode)
            echo "[warn] '$1 $2' is managed by this script; ignoring." >&2; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *)        TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# Model-size defaults (env-overridable). Smaller d_model fits stats features;
# GRU on ~10-dim input doesn't need a wide hidden layer.
TRAIN_ARGS=(--d_model="${D_MODEL:-128}" --n_layers="${N_LAYERS:-4}" \
            ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"})

DATA_DIR="$SCRIPT_DIR/highlight_data"
CKPT_DIR="$SCRIPT_DIR/highlight_ckpt"
LOG_DIR="$CKPT_DIR/logs"
ERR_LOG="$LOG_DIR/_scheduler_stats.err"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Preprocess must have produced stats columns. Verify before scheduling. ──
python - "$DATA_DIR/stats.json" <<'PY'
import json, os, sys
s = json.load(open(sys.argv[1]))
if "stats_feat_dim" not in s:
    print("[error] stats_feat_dim not in stats.json — re-run preprocess_highlight.py "
          "(it now emits causal stat features).", file=sys.stderr)
    sys.exit(1)
print(f"[stats] stats_feat_dim={s['stats_feat_dim']}  rooms train/val/test="
      f"{s['n_train_rooms']}/{s['n_val_rooms']}/{s['n_test_rooms']}")
PY

# Build task list: lr × wd
TASKS=()
for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        TASKS+=("${lr}|${wd}")
    done
done
N_TASKS=${#TASKS[@]}

echo ""
echo "[scheduler] GPUs=[${GPUS[*]}]  jobs/GPU=$JOBS_PER_GPU  slots=$N_SLOTS  epochs=$EPOCHS"
echo "[scheduler] model=$MODEL  input=stats  lr=[${LR_LIST[*]}]  wd=[${WD_LIST[@]}]"
echo "[scheduler] tasks=$N_TASKS  (${#LR_LIST[@]} lr × ${#WD_LIST[@]} wd)"
echo ""

# ── Multi-GPU scheduler (bash 4.x compatible, same as run_highlight.sh) ──
declare -A gpu_pid gpu_tag
declare -a DONE_OK=() DONE_FAIL=()

slot_to_gpu() { echo "${GPUS[$(( $1 / JOBS_PER_GPU ))]}"; }

find_free_gpu() {
    for ((s = 0; s < N_SLOTS; s++)); do
        [[ -z "${gpu_pid[$s]:-}" ]] && { echo "$s"; return; }
    done
    echo -1
}

run_exp() {                            # <slot> <tag> <lr> <wd>
    local slot="$1" tag="$2" lr="$3" wd="$4"
    local gpu; gpu=$(slot_to_gpu "$slot")
    local run_dir="$CKPT_DIR/$tag"
    local logf="$LOG_DIR/${tag}.log"
    mkdir -p "$run_dir"
    echo "  ▶ start  $tag  →  GPU $gpu (slot $slot)   (log: $logf)"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/train_highlight.py" \
        --data_dir "$DATA_DIR" \
        --model "$MODEL" \
        --input_mode stats \
        --out_dir "$run_dir" \
        --epochs "$EPOCHS" \
        --lr "$lr" \
        --weight_decay "$wd" \
        ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} > "$logf" 2>&1 &
    gpu_pid[$slot]=$!
    gpu_tag[$slot]="$tag"
}

reap_finished() {
    wait -n || true
    for s in "${!gpu_pid[@]}"; do
        if ! kill -0 "${gpu_pid[$s]}" 2>"$ERR_LOG"; then
            local tag="${gpu_tag[$s]}"
            local gpu; gpu=$(slot_to_gpu "$s")
            unset "gpu_pid[$s]" "gpu_tag[$s]"
            if [[ -f "$CKPT_DIR/$tag/results_${MODEL}.json" ]]; then
                echo "  ✓ done   $tag  (GPU $gpu freed)"
                DONE_OK+=("$tag")
            else
                echo "  ✗ FAILED $tag  (GPU $gpu freed; see $LOG_DIR/${tag}.log)"
                DONE_FAIL+=("$tag")
            fi
        fi
    done
}

submitted=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r lr wd <<< "$task"
    tag="${MODEL}_stats_lr${lr}_wd${wd}"

    if [[ -f "$CKPT_DIR/$tag/results_${MODEL}.json" ]]; then
        echo "  ⊙ skip   $tag  (results exist)"
        DONE_OK+=("$tag")
        continue
    fi

    slot=$(find_free_gpu)
    while [[ "$slot" -eq -1 ]]; do
        reap_finished
        slot=$(find_free_gpu)
    done
    run_exp "$slot" "$tag" "$lr" "$wd"
    ((submitted++)) || true
    echo "  [dispatch] $submitted/$N_TASKS → GPU$(slot_to_gpu "$slot") slot$slot  ($tag)"
done

while (( ${#gpu_pid[@]} > 0 )); do
    reap_finished
done

echo ""
echo "[scheduler] finished: ok=${#DONE_OK[@]}/${N_TASKS}  failed=${#DONE_FAIL[@]}"
[[ ${#DONE_FAIL[@]} -gt 0 ]] && echo "[scheduler] failed: ${DONE_FAIL[*]}"

# ── Summary: stats GRU results, contrast with embedding GRU if available ──
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  Stats-input GRU results (leakage-free causal statistics)"
echo "══════════════════════════════════════════════════════════════════════"
printf "  %-28s %8s %7s %7s %8s %8s %8s %8s\n" \
    "run" "val_mAP" "mAP" "F1@50" "Kτ(0)" "Kτ(.2)" "Kτ(.4)" "Spear"
echo "  ────────────────────────────────────────────────────────────────────"

python - "$CKPT_DIR" "${MODEL}" <<'PY'
import json, os, glob, sys
ckpt_dir, model = sys.argv[1], sys.argv[2]
row = "  %-28s %8.4f %7.4f %7.4f %8.4f %8.4f %8.4f %8.4f"
runs = []
for p in glob.glob(os.path.join(ckpt_dir, f"{model}_stats_lr*_wd*", f"results_{model}.json")):
    try: d = json.load(open(p))
    except Exception: continue
    a = d.get("args", {})
    runs.append((a.get("lr"), a.get("weight_decay"), d.get("val",{}), d.get("test",{}), p))
runs.sort(key=lambda r: r[2].get("mAP", -1), reverse=True)
for lr, wd, v, t, p in runs:
    tag = os.path.basename(os.path.dirname(p))
    print(row % (tag, v.get("mAP",0), t.get("mAP",0), t.get("F1@50%",0),
                t.get("Kendall_tau_d0.0",0), t.get("Kendall_tau_d0.2",0),
                t.get("Kendall_tau_d0.4",0), t.get("Spearman_rho",0)))
if runs:
    best = runs[0]
    print(f"\n  ★ BEST: lr={best[0]} wd={best[1]}  test mAP={best[3].get('mAP',0):.4f}")

# Contrast with embedding GRU if present
emb = glob.glob(os.path.join(ckpt_dir, f"{model}_lr*_wd*", f"results_{model}.json"))
emb += glob.glob(os.path.join(ckpt_dir, f"{model}_lr*_wd*", f"results_{model}.json"))
emb_runs = []
for p in emb:
    try: d = json.load(open(p))
    except Exception: continue
    if d.get("args",{}).get("input_mode","embedding") == "embedding":
        emb_runs.append(d.get("test",{}).get("mAP", -1))
if emb_runs:
    print(f"\n  [contrast] embedding-GRU best test mAP = {max(emb_runs):.4f}")
    if runs:
        stats_best = max(r[3].get("mAP",0) for r in runs)
        print(f"  [contrast] stats-GRU     best test mAP = {stats_best:.4f}")
        print(f"  → stats vs embedding gap = {stats_best - max(emb_runs):+.4f}  "
              f"({'stats远低→embedding必要' if stats_best < max(emb_runs) - 0.05 else '差距不大,需检查'})")
PY

echo "══════════════════════════════════════════════════════════════════════"
echo "  Logs: $LOG_DIR/gru_stats_*.log"
echo "══════════════════════════════════════════════════════════════════════"

[[ ${#DONE_FAIL[@]} -eq 0 ]]
