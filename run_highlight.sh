#!/usr/bin/env bash
# Highlight benchmark — one-line entry point.
#
# Thin wrapper that forwards to highlight/run_highlight.sh (the real scheduler).
# Output (highlight_data/, highlight_ckpt/) is produced under highlight/.
#
# Usage (same as highlight/run_highlight.sh):
#   bash run_highlight.sh <klm3_data_dir> <gpu_list> [extra_args...]
#
# Examples:
#   bash run_highlight.sh /data/klm3 0,1,2,3                    # default: stats + es variants
#   VARIANTS="plain stats es" bash run_highlight.sh /data/klm3 0,1,2,3   # all 3 variants
#   bash run_highlight.sh /data/klm3 0 --top_k 500 --epochs 5   # quick sanity
#
# See highlight/run_highlight.sh for full env docs (LR_LIST, WD_LIST, EPOCHS,
# JOBS_PER_GPU, VARIANTS, D_MODEL, ...).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/highlight/run_highlight.sh" "$@"
