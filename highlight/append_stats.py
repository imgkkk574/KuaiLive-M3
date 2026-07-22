#!/usr/bin/env python3
"""
append_stats.py — Append leakage-free causal stat features to an existing
segments_labeled.parquet, WITHOUT re-running the full preprocess.

Why: the full preprocess re-scans the 88M-row segment-embedding shards (~450s
fixed cost), but the stat features only depend on columns already in the
parquet (viewers_entered / like_cnt_seg / comment_cnt_seg / seg_start_ts /
seg_end_ts). This script just reads the parquet, calls build_stat_features,
writes the stat_* columns back, and updates stats.json.

Safe to re-run: drops any existing stat_* columns first.
No raw data access, no shard scan — finishes in seconds.

Usage
-----
    python append_stats.py --data_dir highlight_data
    python append_stats.py --data_dir highlight_data --dry-run   # check deps, don't write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_highlight import build_stat_features, STAT_FEAT_COLS, STAT_FEAT_DIM


REQUIRED_COLS = [
    "live_stream_id", "segment_id", "timestamp",
    "viewers_entered", "like_cnt_seg", "comment_cnt_seg",
    "seg_start_ts", "seg_end_ts",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Append causal stat features to existing parquet.")
    p.add_argument("--data_dir", required=True,
                   help="Output directory of preprocess_highlight.py (holds segments_labeled.parquet)")
    p.add_argument("--dry-run", action="store_true",
                   help="Check dependencies only, do not write")
    args = p.parse_args()

    seg_path = os.path.join(args.data_dir, "segments_labeled.parquet")
    stats_path = os.path.join(args.data_dir, "stats.json")

    if not os.path.exists(seg_path):
        print(f"[error] {seg_path} not found — run preprocess_highlight.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {seg_path} ...")
    t0 = time.time()
    df = pd.read_parquet(seg_path)
    print(f"  {len(df):,} rows, {df['live_stream_id'].nunique():,} rooms  ({time.time()-t0:.1f}s)")

    # Dependency check
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[error] parquet missing required columns: {missing}", file=sys.stderr)
        print("       These come from a full preprocess_highlight.py run. Re-run it.", file=sys.stderr)
        sys.exit(1)

    # Sort exactly as build_stat_features expects (per-room time order).
    df = df.sort_values(["live_stream_id", "timestamp"]).reset_index(drop=True)

    # Drop any pre-existing stat columns so a re-run is idempotent.
    pre_existing = [c for c in df.columns if c.startswith("stat_")]
    if pre_existing:
        print(f"  dropping {len(pre_existing)} pre-existing stat_* columns (idempotent re-run)")
        df = df.drop(columns=pre_existing)

    if args.dry_run:
        print(f"\n[dry-run] OK: all {len(REQUIRED_COLS)} required columns present.")
        print(f"[dry-run] Would add {STAT_FEAT_DIM} columns: {STAT_FEAT_COLS}")
        return

    print(f"\nBuilding {STAT_FEAT_DIM} causal stat features (leakage-free) ...")
    t0 = time.time()
    df = build_stat_features(df)
    print(f"  done ({time.time()-t0:.1f}s)")

    # Self-check: seg-0 history must be all zero (no leakage from current segment).
    hist_cols = ["stat_cum_viewers", "stat_cum_likes", "stat_cum_comments",
                 "stat_roll_view_mean", "stat_roll_view_std",
                 "stat_roll_like_mean", "stat_roll_cmt_mean", "stat_view_trend"]
    seg0 = df.groupby("live_stream_id").head(1)
    if not (seg0[hist_cols].sum(axis=1) == 0).all():
        print("[warn] seg-0 history not all-zero — possible leakage. Inspect before using.",
              file=sys.stderr)
    else:
        print("  ✓ leakage self-check: seg-0 history all-zero")

    # Write back. (Sort order changed; that's fine — downstream sorts by timestamp too.)
    print(f"\nWriting {len(df):,} rows back to {seg_path} ...")
    df.to_parquet(seg_path, index=False)
    print(f"  columns now: {len(df.columns)} (added {STAT_FEAT_DIM} stat_*)")

    # Update stats.json with the stat metadata.
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    stats["stats_feat_dim"] = STAT_FEAT_DIM
    stats["stats_feat_cols"] = STAT_FEAT_COLS
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  updated stats.json: stats_feat_dim={STAT_FEAT_DIM}")

    print("\nDone. You can now run run_highlight_stats.sh without a full preprocess.")


if __name__ == "__main__":
    main()
