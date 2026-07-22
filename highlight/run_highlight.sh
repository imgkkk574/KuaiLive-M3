#!/usr/bin/env bash
# Run KLM3 Highlight benchmark: hyperparameter sweep + multi-GPU scheduling.
#
# Usage:
#   bash run_highlight.sh <klm3_data_dir> <gpu_list> [extra_args...]
#
# Examples:
#   bash run_highlight.sh /data/klm3 0,1,2,3                   # 4 GPUs × 4 jobs = 16 concurrent
#   JOBS_PER_GPU=2 bash run_highlight.sh /data/klm3 0,1        # 2/GPU if VRAM is tight
#   bash run_highlight.sh /data/klm3 0,1 --epochs 5           # quick test
#   LR_LIST="1e-3 5e-4" bash run_highlight.sh /data/klm3 0,1   # custom lr grid
#   bash run_highlight.sh /data/klm3 0 --top_k 500            # small data, 1 GPU
#   VARIANTS="es" bash run_highlight.sh /data/klm3 0,1       # only embedding+stats
#   VARIANTS="plain stats es" bash run_highlight.sh /data/klm3 0,1   # all 3 variants
#
# Sweep grid (env-overridable):
#   LR_LIST  (default "1e-3 5e-3 1e-4 5e-4 1e-5")   → 5
#   WD_LIST  (default "1e-5 1e-3 1e-4")              → 3
#   EPOCHS   (default 100)
#   VARIANTS (default "stats es")  → space-separated input variants to run:
#                stats = stats-only (--input_mode stats)
#                es    = embedding + stat branch (--use_stats)
#                plain = embedding-only
#   → 2 groups × 4 models × 5 lr × 3 wd = 120 runs.
#
# Model size (env-overridable, injected as --key=value):
#   BATCH_SIZE (512)  D_MODEL (128)  N_HEADS (8)  N_LAYERS (4)  WINDOW (7)
#
# Scheduling:
#   - Each (group, model, lr, wd) runs on one slot; JOBS_PER_GPU concurrent jobs
#     per physical GPU (default 4). Total concurrent slots = N_GPUS × JOBS_PER_GPU.
#   - Tasks are generated group-major then model-major (all group-0 mlp configs
#     first, …), so the first wave is one group/model until slots free up.
#   - `wait -n` + `kill -0` sweep reclaims freed slots (bash 4.x compatible).
#   - Per-run output: highlight_ckpt/<model>_lr<lr>_wd<wd><suffix>/ where
#     <suffix> = _stats (stats-only), _es (emb+stats), or "" (embedding-only).
#
# Resume:
#   - Preprocessing skipped if highlight_data/stats.json exists.
#   - A run is skipped if its results_<model>.json already exists.
#
# Notes:
#   - This env blocks writes to /dev/null (EPERM), so scheduler stderr goes to
#     highlight_ckpt/logs/_scheduler.err instead.
#   - --lr / --weight_decay are sweep-controlled; passing them on the CLI is
#     ignored (edit LR_LIST / WD_LIST).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──────────────────────────────────────────────────────────────────────────────
# Args: <data_dir> <gpu_list> [extra_args...]
# ──────────────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    sed -n '3,30p' "$0" >&2
    exit 1
fi

KLM3_DATA_DIR="$1"
GPU_LIST="${2:-0}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

IFS=',' read -ra GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "[error] no GPUs parsed from '$GPU_LIST'" >&2; exit 1; }
for g in "${GPUS[@]}"; do
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "[error] invalid GPU id '$g'" >&2; exit 1; }
done

# ──────────────────────────────────────────────────────────────────────────────
# Sweep + scheduling config (all env-overridable)
# ──────────────────────────────────────────────────────────────────────────────
IFS=' ' read -ra LR_LIST <<< "${LR_LIST:-1e-3 5e-3 1e-4 5e-4 1e-5}" || true
IFS=' ' read -ra WD_LIST <<< "${WD_LIST:-1e-5 1e-3 1e-4}" || true
EPOCHS="${EPOCHS:-100}"
MODELS=(mlp gru causal hierarchical)
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"          # concurrent runs per physical GPU

# Input-variant groups to sweep in ONE run. Each group is "name|cli_flag|tag_suffix":
#   stats    : --input_mode stats                 (10-dim leakage-free stats only)
#   es       : --use_stats                        (128-dim embedding + additive 10-dim stat branch)
# Both run for every (model, lr, wd). Tags carry the suffix so the two groups are
# isolated under highlight_ckpt/ and never overwrite each other.
#
# Override the set of groups via the VARIANTS env var, e.g.:
#   VARIANTS="stats es"        bash run_highlight.sh ...   # default: stats + emb+stats
#   VARIANTS="es"              bash run_highlight.sh ...   # only emb+stats
#   VARIANTS="plain stats es"  bash run_highlight.sh ...   # also add embedding-only
# Recognised names: plain (embedding only), stats (stats only), es (emb+stats).
# NOTE: the env var is VARIANTS, not GROUPS — bash reserves GROUPS as a readonly
# array of the user's GIDs, so ${GROUPS:-...} never falls through.
VARIANTS="${VARIANTS:-stats es}"
IFS=' ' read -ra GROUP_LIST <<< "$VARIANTS" || true

# Resolve each requested group name to (cli flag(s), tag suffix).
declare -a GROUP_FLAGS GROUP_SUFFIX
_valid_groups=()
for _g in "${GROUP_LIST[@]}"; do
    case "$_g" in
        plain) GROUP_FLAGS+=(--input_mode=embedding);     GROUP_SUFFIX+=("")       ;;
        stats) GROUP_FLAGS+=(--input_mode=stats);         GROUP_SUFFIX+=("_stats") ;;
        es)    GROUP_FLAGS+=(--use_stats);                GROUP_SUFFIX+=("_es")    ;;
        *) echo "[error] unknown group '$_g' (valid: plain stats es)" >&2; exit 1 ;;
    esac
    _valid_groups+=("$_g")
done
GROUP_LIST=("${_valid_groups[@]}")
N_GROUPS=${#GROUP_LIST[@]}

[[ ${#LR_LIST[@]} -gt 0 && ${#WD_LIST[@]} -gt 0 ]] || {
    echo "[error] empty LR_LIST or WD_LIST" >&2; exit 1; }

# bash version check: `wait -n` needs >= 4.1.
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 1) )); then
    echo "[error] needs bash >= 4.1 (you have ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]})." >&2
    exit 1
fi
N_GPUS=${#GPUS[@]}
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
    echo "[error] JOBS_PER_GPU must be a positive integer (got '$JOBS_PER_GPU')" >&2; exit 1; }
N_SLOTS=$(( N_GPUS * JOBS_PER_GPU ))

# ──────────────────────────────────────────────────────────────────────────────
# Split remaining CLI args: preprocess flags vs train flags.
# --lr / --weight_decay are sweep-controlled (ignored if passed).
# ──────────────────────────────────────────────────────────────────────────────
PREPROCESS_ARGS=()
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --top_k|--min_segs|--min_viewers|--train_frac|--val_frac)
            PREPROCESS_ARGS+=("$1" "${2:?missing value for $1}"); shift 2 ;;
        --out_dir)  echo "[warn] --out_dir is managed by this script; ignoring." >&2; shift 2 ;;
        --lr|--weight_decay)
            echo "[warn] '$1 $2' is a sweep variable; ignoring (edit LR_LIST/WD_LIST)." >&2; shift 2 ;;
        --epochs)   EPOCHS="$2"; echo "[info] --epochs overridden to $EPOCHS"; shift 2 ;;
        *)          TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# Model-size defaults (env-overridable), prepended so CLI copies override later.
MODEL_SIZE_DEFAULTS=(
    --batch_size="${BATCH_SIZE:-512}"
    --d_model="${D_MODEL:-128}"
    --n_heads="${N_HEADS:-8}"
    --n_layers="${N_LAYERS:-4}"
    --window="${WINDOW:-7}"
)
TRAIN_ARGS=("${MODEL_SIZE_DEFAULTS[@]}" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"})

DATA_DIR="$SCRIPT_DIR/highlight_data"
CKPT_DIR="$SCRIPT_DIR/highlight_ckpt"
LOG_DIR="$CKPT_DIR/logs"
ERR_LOG="$LOG_DIR/_scheduler.err"          # stderr sink (/dev/null is blocked)
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Preprocess (skip if stats.json already exists)
# ──────────────────────────────────────────────────────────────────────────────
if [[ -f "$DATA_DIR/stats.json" ]]; then
    echo "[preprocess] highlight_data/stats.json already exists — skipping."
    echo "              Delete highlight_data/ to force re-preprocessing."
else
    echo "[preprocess] Generating highlight_data/ from $KLM3_DATA_DIR ..."
    python "$SCRIPT_DIR/preprocess_highlight.py" \
        --data_dir "$KLM3_DATA_DIR" \
        --out_dir "$DATA_DIR" \
        ${PREPROCESS_ARGS[@]+"${PREPROCESS_ARGS[@]}"}
fi

if [[ ! -f "$DATA_DIR/segments_labeled.parquet" ]]; then
    echo "[error] $DATA_DIR/segments_labeled.parquet not found. Preprocessing failed?" >&2
    exit 1
fi

python - "$DATA_DIR/stats.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
print(f"[preprocess] rooms  train={s['n_train_rooms']}  val={s['n_val_rooms']}  "
      f"test={s['n_test_rooms']}")
print(f"[preprocess] segs   train={s['n_train_segments']}  val={s['n_val_segments']}  "
      f"test={s['n_test_segments']}")
if s['n_test_rooms'] < 1000:
    print(f"[warn] test rooms ({s['n_test_rooms']}) < 1000 (design doc §6.1). "
          f"Consider increasing --top_k.")
PY

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Build sweep task list = groups × models × lr × wd
#   Each task encodes group_index so run_exp can replay the right CLI flags.
#   Field layout: "gi|model|lr|wd"
# ──────────────────────────────────────────────────────────────────────────────
TASKS=()
for ((gi = 0; gi < N_GROUPS; gi++)); do
    for model in "${MODELS[@]}"; do
        for lr in "${LR_LIST[@]}"; do
            for wd in "${WD_LIST[@]}"; do
                TASKS+=("${gi}|${model}|${lr}|${wd}")
            done
        done
    done
done
N_TASKS=${#TASKS[@]}

echo ""
echo "[scheduler] GPUs=[${GPUS[*]}]  jobs/GPU=$JOBS_PER_GPU  concurrent_slots=$N_SLOTS  epochs=$EPOCHS"
echo "[scheduler] lr=[${LR_LIST[*]}]  wd=[${WD_LIST[*]}]"
echo "[scheduler] groups=[$(printf '%s ' "${GROUP_LIST[@]}")]  (${N_GROUPS} input variants: ${GROUP_LIST[*]})"
echo "[scheduler] tasks=$N_TASKS  (${N_GROUPS} groups × ${#MODELS[@]} models × ${#LR_LIST[@]} lr × ${#WD_LIST[@]} wd)"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU scheduler (bash 4.x compatible), with N jobs per GPU.
#   - A "slot" is one concurrent job slot; total slots = N_GPUS × JOBS_PER_GPU.
#   - gpu_pid/gpu_model/gpu_tag: slot_idx -> pid / model / tag of the running job.
#   - slot_to_gpu <s> maps a slot to its physical GPU id: GPUS[s / JOBS_PER_GPU].
#     e.g. JOBS_PER_GPU=4, GPUS=(3 4): slots 0-3 → GPU3, slots 4-7 → GPU4.
#   - find_free_gpu:   returns an idle slot index, or -1 if all busy.
#   - reap_finished:  `wait -n` blocks until ANY job exits, then `kill -0`
#     scans gpu_pid to reclaim every slot whose pid is gone. Uses $ERR_LOG
#     instead of /dev/null (this env blocks /dev/null writes).
# ──────────────────────────────────────────────────────────────────────────────
declare -A gpu_pid gpu_model gpu_tag
declare -a DONE_OK=() DONE_FAIL=()

slot_to_gpu() {                        # <slot>  →  physical GPU id
    echo "${GPUS[$(( $1 / JOBS_PER_GPU ))]}"
}

find_free_gpu() {
    for ((s = 0; s < N_SLOTS; s++)); do
        [[ -z "${gpu_pid[$s]:-}" ]] && { echo "$s"; return; }
    done
    echo -1
}

run_exp() {                            # <slot> <tag> <model> <lr> <wd> <group_index>
    local slot="$1" tag="$2" model="$3" lr="$4" wd="$5" gi="$6"
    local gpu; gpu=$(slot_to_gpu "$slot")   # physical GPU id
    local run_dir="$CKPT_DIR/$tag"
    local logf="$LOG_DIR/${tag}.log"
    mkdir -p "$run_dir"
    # Replay this group's CLI flags via PARALLEL arrays indexed by gi.
    local -a grp_flag=()
    local _f
    # GROUP_FLAGS[gi] is a space-separated string of one or more --key[=val] tokens.
    for _f in ${GROUP_FLAGS[$gi]}; do
        grp_flag+=("$_f")
    done
    echo "  ▶ start  $tag  →  GPU $gpu (slot $slot)   (log: $logf)"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/train_highlight.py" \
        --data_dir "$DATA_DIR" \
        --model "$model" \
        --out_dir "$run_dir" \
        --epochs "$EPOCHS" \
        --lr "$lr" \
        --weight_decay "$wd" \
        "${grp_flag[@]}" \
        ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} > "$logf" 2>&1 &
    gpu_pid[$slot]=$!
    gpu_model[$slot]="$model"
    gpu_tag[$slot]="$tag"
}

reap_finished() {
    # Block until any background job exits, then free every slot whose pid is gone.
    wait -n || true
    for s in "${!gpu_pid[@]}"; do
        if ! kill -0 "${gpu_pid[$s]}" 2>"$ERR_LOG"; then
            local tag="${gpu_tag[$s]}" model="${gpu_model[$s]}"
            local gpu; gpu=$(slot_to_gpu "$s")
            unset "gpu_pid[$s]" "gpu_model[$s]" "gpu_tag[$s]"
            # Success iff results_<model>.json was produced in this run's dir.
            if [[ -f "$CKPT_DIR/$tag/results_${model}.json" ]]; then
                echo "  ✓ done   $tag  (GPU $gpu freed)"
                DONE_OK+=("$tag")
            else
                echo "  ✗ FAILED $tag  (GPU $gpu freed; see $LOG_DIR/${tag}.log)"
                DONE_FAIL+=("$tag")
            fi
        fi
    done
}

# ── Main dispatch ──
submitted=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r gi model lr wd <<< "$task"
    tag="${model}_lr${lr}_wd${wd}${GROUP_SUFFIX[$gi]}"

    # Resume: skip runs that already produced results.
    if [[ -f "$CKPT_DIR/$tag/results_${model}.json" ]]; then
        echo "  ⊙ skip   $tag  (results exist)"
        DONE_OK+=("$tag")
        continue
    fi

    slot=$(find_free_gpu)
    while [[ "$slot" -eq -1 ]]; do
        reap_finished
        slot=$(find_free_gpu)
    done
    run_exp "$slot" "$tag" "$model" "$lr" "$wd" "$gi"
    ((submitted++)) || true
    echo "  [dispatch] $submitted/$N_TASKS → GPU$(slot_to_gpu "$slot") slot$slot  ($tag)"
done

# Drain remaining jobs.
while (( ${#gpu_pid[@]} > 0 )); do
    reap_finished
done

echo ""
echo "[scheduler] finished: ok=${#DONE_OK[@]}/${N_TASKS}  failed=${#DONE_FAIL[@]}"
[[ ${#DONE_FAIL[@]} -gt 0 ]] && echo "[scheduler] failed: ${DONE_FAIL[*]}"

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Summarise sweep results
#   - For each model, pick the config with the best val mAP, report its test metrics.
#   - Then print the full grid (val mAP | test mAP) sorted by val mAP desc.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  Highlight sweep — best config per model (by val mAP) → test metrics"
echo "══════════════════════════════════════════════════════════════════════"
printf "  %-13s %-18s %8s %7s %7s %8s %8s %8s %8s\n" \
    "Model" "best(lr,wd)" "valmAP" "mAP" "F1@50" "Kτ(0)" "Kτ(.2)" "Kτ(.4)" "Spear"
echo "  ────────────────────────────────────────────────────────────────────"

python - "$CKPT_DIR" "${MODELS[@]}" <<'PY'
import json, os, glob, sys
ckpt_dir, *models = sys.argv[1:]

# Classify each run into an input-variant group from its args.
def group_of(a):
    if a.get("use_stats"):
        return "es"            # embedding + stat branch
    if a.get("input_mode") == "stats":
        return "stats"         # stats only
    return "plain"             # embedding only

GROUP_TITLE = {"stats": "stats-only", "es": "embedding+stats",
               "plain": "embedding-only"}

# Collect every results_*.json under per-run subdirs.
# run dict: group, model, lr, wd, val_mAP, test(dict)
runs = []
for p in glob.glob(os.path.join(ckpt_dir, "*", "results_*.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    a = d.get("args", {})
    runs.append({
        "group": group_of(a),
        "model": d.get("model"),
        "lr": a.get("lr"),
        "wd": a.get("weight_decay"),
        "val_mAP": d.get("val", {}).get("mAP", -1.0),
        "test": d.get("test", {}),
    })

best_row = "  %-13s lr=%-7s wd=%-7s %8.4f %7.4f %7.4f %8.4f %8.4f %8.4f %8.4f"
nomodel  = "  %-13s (no results)"

# Groups actually present, in a stable order.
present_groups = [g for g in ("stats", "es", "plain")
                  if any(r["group"] == g for r in runs)]

for g in present_groups:
    print()
    print(f"══ {GROUP_TITLE[g]} ({g}) — best config per model (by val mAP) → test ══")
    gruns = [r for r in runs if r["group"] == g]
    for m in models:
        mruns = [r for r in gruns if r["model"] == m]
        if not mruns:
            print(nomodel % m)
            continue
        best = max(mruns, key=lambda r: r["val_mAP"])
        t = best["test"]
        print(best_row % (m, best["lr"], best["wd"], best["val_mAP"],
                          t.get("mAP", 0), t.get("F1@50%", 0),
                          t.get("Kendall_tau_d0.0", 0), t.get("Kendall_tau_d0.2", 0),
                          t.get("Kendall_tau_d0.4", 0), t.get("Spearman_rho", 0)))

print()
print("══ Cross-variant contrast (best test mAP per model × group) ══")
# header
groups_hdr = present_groups
hdr = "  %-13s" % "model"
for g in groups_hdr:
    hdr += f" {GROUP_TITLE[g]:>14}"
print(hdr)
for m in models:
    line = f"  {m:<13s}"
    for g in groups_hdr:
        mruns = [r for r in runs if r["group"] == g and r["model"] == m]
        if mruns:
            best = max(mruns, key=lambda r: r["val_mAP"])
            tm = best["test"].get("mAP", 0)
            line += f" {tm:>14.4f}"
        else:
            line += f" {'—':>14}"
    print(line)
PY

echo "══════════════════════════════════════════════════════════════════════"
echo "  Done. Logs:     $LOG_DIR/<model>_lr<lr>_wd<wd><suffix>.log"
echo "  Run dirs:       $CKPT_DIR/<model>_lr<lr>_wd<wd><suffix>/  (suffix: _stats|_es|)"
echo "  Scheduler err:  $ERR_LOG"
echo "══════════════════════════════════════════════════════════════════════"

# Non-zero exit if any run failed
[[ ${#DONE_FAIL[@]} -eq 0 ]]
