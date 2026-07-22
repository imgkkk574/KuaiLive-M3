"""
preprocess_highlight.py — KuaiLive-M3 Highlight Task Preprocessing
====================================================================

Steps:
  1. Select top-K most active live rooms by interaction count
  2. Filter interactions / behaviors (like, comment) to those rooms
  3. Load segment embeddings (live_emb_128_ts) for those rooms from shards
  4. Compute per-segment highlight labels:
       LVTR proxy   — watch retention: viewers who stayed / viewers who entered
       ED           — engagement density: (likes + 2*comments) / viewers_entered
       hl_score     — 0.6 * LVTR_norm + 0.4 * ED_norm  (normalised per room)
       hl_binary    — 1 if hl_score ≥ 70th-percentile within room, else 0
  5. Chronological train / val / test split by room start_timestamp
  6. Save to highlight_data/

Output files
------------
  highlight_data/
    segments_labeled.parquet   — per-segment embeddings + labels + split
    rooms_split.parquet        — room metadata with split assignment
    interactions_filtered.parquet  — filtered interactions for user history
    stats.json                 — preprocessing statistics

Usage
-----
  python preprocess_highlight.py --data_dir /path/to/klm3 [--top_k 10000]

Notes on column naming
----------------------
  live_interaction  uses  live_id
  live_emb_128_ts   uses  live_stream_id
  live_comment      uses  live_stream_id
  live_like         uses  live_id
  → We treat live_id == live_stream_id throughout and join on that key.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# klm3.py lives in the repo root (shared with the CDR pipeline), one level
# above this highlight/ subdir. Add the parent dir to sys.path so `import klm3`
# resolves regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klm3 import KLM3Config, KLM3Dataset

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "highlight_data")
SEGMENT_EMB_DIR     = "live_emb_128_ts"
SEGMENT_EMB_PATTERN = "live_emb_128_ts_part*.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ──────────────────────────────────────────────────────────────────────────────

def ts_to_unix(s: pd.Series) -> pd.Series:
    """Convert a timestamp series to float64 Unix seconds.

    Handles:
      - datetime64 / Timestamp  → divide int64 ns by 1e9
      - large numeric (> 1e12)  → assume milliseconds, divide by 1e3
      - small numeric           → assume seconds already
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("int64") / 1e9
    s_num = pd.to_numeric(s, errors="coerce")
    # median > 1e12 → milliseconds
    med = s_num.median()
    if pd.notna(med) and med > 1e12:
        return s_num / 1e3
    return s_num.astype("float64")


# ──────────────────────────────────────────────────────────────────────────────
# Per-room label computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_segment_labels(
    room_segs: pd.DataFrame,
    room_ints: pd.DataFrame,
    room_likes: pd.DataFrame,
    room_comments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute LVTR and engagement density for every segment in one room.

    Parameters
    ----------
    room_segs      sorted segment rows for this room; must have columns
                   [segment_id, timestamp (=seg_end_ts), seg_start_ts, seg_end_ts]
    room_ints      interaction rows for this room; must have columns
                   [play_start_ts, play_end_ts]  (float Unix seconds)
    room_likes     like events; must have column [like_ts] (float Unix s)
    room_comments  comment events; must have column [comment_ts_unix] (float Unix s)

    Timestamp semantics
    -------------------
    timestamp in live_emb_128_ts is the segment END timestamp.
    seg_start_ts = end_ts of the previous segment (shift(1) within room),
                   or room start_timestamp for the first kept segment.
    seg_end_ts   = timestamp (direct alias).

    Returns
    -------
    room_segs with additional columns:
        lvtr, ed, viewers_entered, like_cnt_seg, comment_cnt_seg
    """
    seg_starts = room_segs["seg_start_ts"].values   # [K]  start of each segment
    seg_ends   = room_segs["seg_end_ts"].values     # [K]  end of each segment (= timestamp)
    K = len(seg_starts)

    # ── LVTR (watch retention) ────────────────────────────────────────────────
    if len(room_ints) > 0:
        play_starts = room_ints["play_start_ts"].values   # [N]
        play_ends   = room_ints["play_end_ts"].values     # [N]

        # Boolean [K, N] — broadcast to avoid Python loops
        entered_mat = play_starts[None, :] <= seg_starts[:, None]   # user started ≤ seg start
        stayed_mat  = play_ends[None, :]   >= seg_ends[:, None]     # user still watching at seg end

        entered_cnt = entered_mat.sum(axis=1).astype(np.float32)                 # [K]
        stayed_cnt  = (entered_mat & stayed_mat).sum(axis=1).astype(np.float32) # [K]
        lvtr        = stayed_cnt / np.clip(entered_cnt, 1.0, None)
    else:
        entered_cnt = np.zeros(K, dtype=np.float32)
        lvtr        = np.zeros(K, dtype=np.float32)

    # ── Engagement density ────────────────────────────────────────────────────
    like_ts = room_likes["like_ts"].values if len(room_likes) > 0 else np.array([], dtype=np.float64)
    cmt_ts  = room_comments["comment_ts_unix"].values if len(room_comments) > 0 else np.array([], dtype=np.float64)

    if len(like_ts) > 0:
        in_win_like = (like_ts[None, :] >= seg_starts[:, None]) & (like_ts[None, :] < seg_ends[:, None])
        like_cnt_seg = in_win_like.sum(axis=1).astype(np.float32)
    else:
        like_cnt_seg = np.zeros(K, dtype=np.float32)

    if len(cmt_ts) > 0:
        in_win_cmt = (cmt_ts[None, :] >= seg_starts[:, None]) & (cmt_ts[None, :] < seg_ends[:, None])
        cmt_cnt_seg = in_win_cmt.sum(axis=1).astype(np.float32)
    else:
        cmt_cnt_seg = np.zeros(K, dtype=np.float32)

    engagement = like_cnt_seg + 2.0 * cmt_cnt_seg
    ed = engagement / np.clip(entered_cnt, 1.0, None)

    # ── Assemble ──────────────────────────────────────────────────────────────
    out = room_segs.copy()
    out["lvtr"]            = lvtr
    out["ed"]              = ed
    out["viewers_entered"] = entered_cnt.astype(np.int32)
    out["like_cnt_seg"]    = like_cnt_seg.astype(np.int32)
    out["comment_cnt_seg"] = cmt_cnt_seg.astype(np.int32)
    return out


def _minmax_norm_per_room(df: pd.DataFrame, col: str) -> pd.Series:
    """Min-max normalise `col` within each room independently."""
    def _norm(x: pd.Series) -> pd.Series:
        lo, hi = x.min(), x.max()
        if hi == lo:
            return pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)
        return ((x - lo) / (hi - lo)).astype(np.float32)
    return df.groupby("live_stream_id")[col].transform(_norm)


# Column names of the leakage-free stats feature vector (order matters —
# dataset_highlight.py reads these names). STAT_FEAT_DIM must match the count.
STAT_FEAT_COLS = [
    "stat_rel_pos",        # k / K            (positional, no leakage)
    "stat_seg_dur_log",    # log(seg duration) (positional)
    "stat_cum_viewers",    # cumsum(viewers).shift(1)   (causal history)
    "stat_cum_likes",      # cumsum(likes).shift(1)
    "stat_cum_comments",   # cumsum(comments).shift(1)
    "stat_roll_view_mean", # viewers.shift(1).rolling(5).mean()
    "stat_roll_view_std",  # viewers.shift(1).rolling(5).std()
    "stat_view_trend",     # diff of rolling mean
    "stat_roll_like_mean", # likes.shift(1).rolling(5).mean()
    "stat_roll_cmt_mean",  # comments.shift(1).rolling(5).mean()
]
STAT_FEAT_DIM = len(STAT_FEAT_COLS)   # 10


def build_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-free per-segment statistics features.

    For segment k, every feature uses only segments 0..k-1 (via shift(1)) plus
    positional features — NEVER the current segment's viewers/likes/comments
    (those are direct components of the LVTR/ED labels → using them = leakage).

    Features are min-max normalised globally (across all rooms) so values land
    in [0, 1]; NaN (first segment, no history) → 0.

    Parameters
    ----------
    df : must already be sorted by (live_stream_id, timestamp) and contain
         viewers_entered, like_cnt_seg, comment_cnt_seg, seg_start_ts,
         seg_end_ts columns.

    Returns
    -------
    df with the STAT_FEAT_COLS columns added (float32).
    """
    out = df.copy()

    # Per-room operations (groupby ensures no cross-room leakage).
    g = out.groupby("live_stream_id", sort=False)

    # Positional features (no behavior → no leakage risk)
    room_len = g["segment_id"].transform("count").astype(np.float32)
    out["stat_rel_pos"] = (g.cumcount().astype(np.float32) + 1.0) / room_len
    dur = (out["seg_end_ts"] - out["seg_start_ts"]).clip(lower=0.0)
    out["stat_seg_dur_log"] = np.log1p(dur).astype(np.float32)

    # Causal history: cumsum over 0..k, then shift(1) so segment k sees only
    # 0..k-1. Done within each room (groupby) so no cross-room leakage.
    out["stat_cum_viewers"]  = g["viewers_entered"].cumsum().groupby(
        out["live_stream_id"], sort=False).shift(1).astype(np.float32)
    out["stat_cum_likes"]    = g["like_cnt_seg"].cumsum().groupby(
        out["live_stream_id"], sort=False).shift(1).astype(np.float32)
    out["stat_cum_comments"] = g["comment_cnt_seg"].cumsum().groupby(
        out["live_stream_id"], sort=False).shift(1).astype(np.float32)

    # Rolling stats on shift(1)-ed counts: segment k uses only past segments.
    def _roll_mean(shifted):
        return shifted.groupby(out["live_stream_id"], sort=False).rolling(
            5, min_periods=1).mean().reset_index(level=0, drop=True)
    def _roll_std(shifted):
        return shifted.groupby(out["live_stream_id"], sort=False).rolling(
            5, min_periods=1).std().reset_index(level=0, drop=True)

    v_shift = g["viewers_entered"].shift(1)
    l_shift = g["like_cnt_seg"].shift(1)
    c_shift = g["comment_cnt_seg"].shift(1)

    out["stat_roll_view_mean"] = _roll_mean(v_shift).astype(np.float32)
    out["stat_roll_view_std"]  = _roll_std(v_shift).astype(np.float32)
    out["stat_roll_like_mean"] = _roll_mean(l_shift).astype(np.float32)
    out["stat_roll_cmt_mean"]  = _roll_mean(c_shift).astype(np.float32)

    # Trend = change in rolling viewer mean (also causal)
    out["stat_view_trend"] = (
        out.groupby("live_stream_id", sort=False)["stat_roll_view_mean"].diff()
    ).astype(np.float32)

    # Fill NaN (first segments with no history) with 0.
    out[STAT_FEAT_COLS] = out[STAT_FEAT_COLS].fillna(0.0)

    # Global min-max normalisation (across all rows) so each col ∈ [0, 1].
    # Embedding features are left untouched. stat_view_trend is a signed
    # difference (already centred at 0 = "no change"); min-max would map 0 to
    # a non-zero value and inject a spurious signal into history-free segments,
    # so it is left in raw (clipped) form instead.
    _CLIP_COLS = {
        "stat_view_trend",
    }
    for col in STAT_FEAT_COLS:
        if col in _CLIP_COLS:
            out[col] = out[col].clip(-1.0, 1.0).astype(np.float32)
            continue
        lo, hi = out[col].min(), out[col].max()
        if pd.notna(hi) and hi > lo:
            out[col] = ((out[col] - lo) / (hi - lo)).astype(np.float32)
        else:
            out[col] = np.zeros(len(out), dtype=np.float32)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess KLM3 → Highlight Task")
    parser.add_argument("--data_dir",     required=True,
                        help="Path to KuaiLive-M3 raw data directory")
    parser.add_argument("--top_k",        type=int,   default=10000,
                        help="Number of most-active live rooms to keep (default: 10000)")
    parser.add_argument("--min_segs",     type=int,   default=5,
                        help="Minimum segments per room after filtering (default: 5)")
    parser.add_argument("--min_viewers",  type=int,   default=5,
                        help="Minimum peak viewer count per room (default: 5)")
    parser.add_argument("--train_frac",   type=float, default=0.7)
    parser.add_argument("--val_frac",     type=float, default=0.1)
    # test_frac is implicitly 1 - train_frac - val_frac = 0.2
    parser.add_argument("--out_dir",      default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("KuaiLive-M3 Highlight Task Preprocessing")
    print("=" * 60)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  top_k       : {args.top_k}")
    print(f"  min_segs    : {args.min_segs}")
    print(f"  min_viewers : {args.min_viewers}")
    print(f"  out_dir     : {args.out_dir}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Load core tables
    # ──────────────────────────────────────────────────────────────────────────
    config = KLM3Config(load_behaviors=True, load_mm_live=True)
    ds = KLM3Dataset(args.data_dir, config, verbose=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Select top-K rooms
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/7] Selecting top rooms by interaction count ...")
    live_int  = ds.live_interaction
    room_meta = ds.live_room_meta

    room_counts = live_int["live_id"].value_counts()
    top_room_ids = set(room_counts.head(args.top_k).index)
    min_cnt = room_counts.iloc[min(args.top_k, len(room_counts)) - 1]
    print(f"  Selected {len(top_room_ids):,} rooms "
          f"(min interactions per room: {min_cnt:,})")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Filter interactions → user set
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/7] Filtering interactions and deriving user set ...")
    live_filt = live_int[live_int["live_id"].isin(top_room_ids)].copy()
    valid_users = set(live_filt["user_id"].unique())
    print(f"  Interactions : {len(live_filt):,}")
    print(f"  Users        : {len(valid_users):,}")

    # Normalise timestamps → float Unix seconds
    live_filt["play_start_ts"] = ts_to_unix(live_filt["live_play_start_timestamp"])
    live_filt["play_end_ts"]   = ts_to_unix(live_filt["live_play_end_timestamp"])

    # Fallback: infer play_end from play_duration when end timestamp is missing
    missing_end = live_filt["play_end_ts"].isna()
    if missing_end.any():
        dur_raw = pd.to_numeric(live_filt.loc[missing_end, "play_duration"], errors="coerce")
        # play_duration might be in milliseconds (live_room_meta uses ms)
        dur_med = dur_raw.median()
        dur_s = dur_raw / (1e3 if pd.notna(dur_med) and dur_med > 1e4 else 1.0)
        live_filt.loc[missing_end, "play_end_ts"] = (
            live_filt.loc[missing_end, "play_start_ts"] + dur_s
        )

    live_filt = live_filt.dropna(subset=["play_start_ts", "play_end_ts"])
    # Sanity check: end must be after start
    live_filt = live_filt[live_filt["play_end_ts"] >= live_filt["play_start_ts"]]
    print(f"  Interactions after timestamp cleaning: {len(live_filt):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Filter behavior tables (like / comment)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/7] Filtering behavior tables ...")

    # live_like: user_id, live_id, like_timestamp
    likes_raw  = ds.live_like
    likes_filt = likes_raw[likes_raw["live_id"].isin(top_room_ids)].copy()
    likes_filt["like_ts"] = ts_to_unix(likes_filt["like_timestamp"])
    likes_filt = likes_filt.dropna(subset=["like_ts"])
    print(f"  Likes    : {len(likes_filt):,}")

    # live_comment: live_stream_id, user_id, author_id, content, comment_ts
    comments_raw  = ds.live_comment
    # live_comment uses live_stream_id (string, same key space as live_emb_128_ts);
    # coerce to int64 to match live_id in the top_room_ids set.
    comments_filt = comments_raw.copy()
    comments_filt["live_stream_id"] = pd.to_numeric(
        comments_filt["live_stream_id"], errors="coerce"
    )
    comments_filt = comments_filt.dropna(subset=["live_stream_id"]).copy()
    comments_filt["live_stream_id"] = comments_filt["live_stream_id"].astype("int64")
    comments_filt = comments_filt[comments_filt["live_stream_id"].isin(top_room_ids)]
    comments_filt["comment_ts_unix"] = ts_to_unix(comments_filt["comment_ts"])
    comments_filt = comments_filt.dropna(subset=["comment_ts_unix"])
    print(f"  Comments : {len(comments_filt):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Load segment embeddings for top rooms (scan all shards)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[4/7] Loading segment embeddings for top-{args.top_k} rooms ...")
    shard_dir = os.path.join(args.data_dir, SEGMENT_EMB_DIR)
    shards    = sorted(glob.glob(os.path.join(shard_dir, SEGMENT_EMB_PATTERN)))
    if not shards:
        raise FileNotFoundError(f"No segment shards found in {shard_dir!r}. "
                                f"Check that live_emb_128_ts/ exists under data_dir.")
    print(f"  Found {len(shards)} shards — scanning ...")

    t0 = time.time()
    seg_parts: list[pd.DataFrame] = []
    for i, shard_path in enumerate(shards, 1):
        df_shard = pd.read_parquet(shard_path)
        # live_stream_id is stored as a string in the parquet shards, while
        # live_id (interactions / room_meta) is int64. Coerce per shard BEFORE
        # the .isin() filter — otherwise the int/str mismatch drops every row.
        df_shard["live_stream_id"] = pd.to_numeric(df_shard["live_stream_id"], errors="coerce")
        df_shard = df_shard.dropna(subset=["live_stream_id"]).copy()
        df_shard["live_stream_id"] = df_shard["live_stream_id"].astype("int64")
        df_hit = df_shard[df_shard["live_stream_id"].isin(top_room_ids)]
        if len(df_hit) > 0:
            seg_parts.append(df_hit)
        print(f"  Shard {i:2d}/{len(shards)}: {len(df_hit):>8,} segments matched", end="\r")
    print()

    if not seg_parts:
        raise RuntimeError("No segments found for any of the top rooms. "
                           "Check that live_stream_id in segments matches live_id in interactions.")

    segs = pd.concat(seg_parts, ignore_index=True)
    segs["timestamp"] = ts_to_unix(segs["timestamp"])
    segs = segs.sort_values(["live_stream_id", "timestamp"]).reset_index(drop=True)
    print(f"  Total segments loaded : {len(segs):,}  ({time.time()-t0:.1f}s)")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Compute segment end timestamps
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[5/7] Computing segment boundaries ...")
    # KEY: timestamp in live_emb_128_ts is the segment END timestamp.
    # Segment durations are non-uniform and some segments (0 viewers) are dropped,
    # so we cannot infer start from a fixed duration.
    #
    # Safe alignment:
    #   seg_end_ts   = timestamp  (directly stored)
    #   seg_start_ts = end_ts of the PREVIOUS segment within the same room
    #                  (shift(1) on timestamp, sorted ascending)
    #   First kept segment → fall back to room start_timestamp.

    room_meta_filt = room_meta[room_meta["live_id"].isin(top_room_ids)].copy()
    room_meta_filt["room_start_ts"] = ts_to_unix(room_meta_filt["start_timestamp"])
    room_meta_filt["room_end_ts"]   = ts_to_unix(room_meta_filt["end_timestamp"])
    room_start_map = room_meta_filt.set_index("live_id")["room_start_ts"]

    # seg_end_ts = this segment's end timestamp (the stored field)
    segs["seg_end_ts"] = segs["timestamp"]

    # seg_start_ts = previous segment's end_ts; NaN for first segment in each room
    segs["seg_start_ts"] = segs.groupby("live_stream_id")["timestamp"].shift(1)

    # Fill first segment's start with the room's start_timestamp
    segs["room_start_ts"] = segs["live_stream_id"].map(room_start_map)
    segs["seg_start_ts"]  = segs["seg_start_ts"].fillna(segs["room_start_ts"])

    # Drop rows where boundaries are still NaN or end ≤ start (data noise)
    valid_boundary = (
        segs["seg_start_ts"].notna()
        & segs["seg_end_ts"].notna()
        & (segs["seg_end_ts"] > segs["seg_start_ts"])
    )
    n_dropped = (~valid_boundary).sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} segments with invalid boundaries")
    segs = segs[valid_boundary].reset_index(drop=True)
    print(f"  Segments after boundary filter: {len(segs):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Compute per-segment highlight labels (room by room)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[6/7] Computing highlight labels ...")
    print("  (processing room by room with vectorised numpy ops)")

    # Pre-group tables for O(1) lookup per room
    ints_by_room     = {k: v for k, v in live_filt.groupby("live_id")}
    likes_by_room    = {k: v for k, v in likes_filt.groupby("live_id")}
    comments_by_room = {k: v for k, v in comments_filt.groupby("live_stream_id")}
    segs_by_room     = {k: v for k, v in segs.groupby("live_stream_id")}

    _empty_df = pd.DataFrame()
    labeled_parts: list[pd.DataFrame] = []
    room_ids_in_segs = list(segs_by_room.keys())
    t0 = time.time()

    for ridx, live_id in enumerate(room_ids_in_segs):
        if ridx % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(ridx, 1) * (len(room_ids_in_segs) - ridx)
            print(f"  Room {ridx+1:>6,}/{len(room_ids_in_segs):,}  "
                  f"elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s", end="\r")

        room_segs     = segs_by_room[live_id].reset_index(drop=True)
        room_ints     = ints_by_room.get(live_id, _empty_df)
        room_likes    = likes_by_room.get(live_id, _empty_df)
        room_comments = comments_by_room.get(live_id, _empty_df)

        labeled = compute_segment_labels(room_segs, room_ints, room_likes, room_comments)
        labeled_parts.append(labeled)

    print()
    segs = pd.concat(labeled_parts, ignore_index=True)
    print(f"  Label computation done: {len(segs):,} segments  ({time.time()-t0:.1f}s)")

    # ── Filter low-quality rooms ──────────────────────────────────────────────
    room_stats = segs.groupby("live_stream_id").agg(
        n_segs      =("segment_id", "count"),
        max_viewers =("viewers_entered", "max"),
    )
    valid_rooms_mask = (
        (room_stats["n_segs"]      >= args.min_segs) &
        (room_stats["max_viewers"] >= args.min_viewers)
    )
    valid_rooms = set(room_stats[valid_rooms_mask].index)
    segs = segs[segs["live_stream_id"].isin(valid_rooms)].reset_index(drop=True)
    print(f"  After quality filters  : {len(segs):,} segments, "
          f"{segs['live_stream_id'].nunique():,} rooms")

    # ── Normalise and combine labels ──────────────────────────────────────────
    segs["lvtr_norm"] = _minmax_norm_per_room(segs, "lvtr")
    segs["ed_norm"]   = _minmax_norm_per_room(segs, "ed")
    segs["hl_score"]  = (0.6 * segs["lvtr_norm"] + 0.4 * segs["ed_norm"]).astype(np.float32)

    # Binary label: top-30% within each room = highlight
    segs["hl_binary"] = (
        segs["hl_score"] >= segs.groupby("live_stream_id")["hl_score"]
                                 .transform(lambda x: np.percentile(x, 70))
    ).astype(np.int8)

    # ── Causal interaction-statistics features (for the stats-input baseline) ──
    # A leakage-free alternative input: per-segment features built ONLY from
    # past segments (shift(1)), plus positional features. Used to test whether
    # highlight signal comes from content (segment embedding) or from behavior
    # statistics alone. Crucially these never use the current segment's
    # viewers/likes/comments (which are direct components of the LVTR/ED labels).
    segs = build_stat_features(segs)

    # Ensure feature_128 is stored as numpy arrays (may come as lists from parquet)
    if segs["feature_128"].dtype == object:
        segs["feature_128"] = segs["feature_128"].apply(
            lambda x: np.asarray(x, dtype=np.float32)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Train / val / test split — chronological by room start_timestamp
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[7/7] Chronological train / val / test split ...")
    valid_room_meta = room_meta_filt[room_meta_filt["live_id"].isin(valid_rooms)].copy()
    valid_room_meta = valid_room_meta.sort_values("room_start_ts").reset_index(drop=True)
    n_rooms  = len(valid_room_meta)
    n_train  = int(n_rooms * args.train_frac)
    n_val    = int(n_rooms * args.val_frac)

    train_rooms = set(valid_room_meta.iloc[:n_train]["live_id"])
    val_rooms   = set(valid_room_meta.iloc[n_train : n_train + n_val]["live_id"])
    test_rooms  = set(valid_room_meta.iloc[n_train + n_val:]["live_id"])

    def _assign_split(lid: int) -> str:
        if lid in train_rooms:
            return "train"
        if lid in val_rooms:
            return "val"
        return "test"

    segs["split"] = segs["live_stream_id"].map(_assign_split)

    for sp in ("train", "val", "test"):
        ss = segs[segs["split"] == sp]
        print(f"  {sp:5s}: {ss['live_stream_id'].nunique():>6,} rooms, "
              f"{len(ss):>9,} segments")

    # ──────────────────────────────────────────────────────────────────────────
    # Save outputs
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\nSaving to {args.out_dir} ...")

    # Core segments file
    segs_out = segs.drop(columns=["room_start_ts"], errors="ignore")
    segs_path = os.path.join(args.out_dir, "segments_labeled.parquet")
    segs_out.to_parquet(segs_path, index=False)
    print(f"  segments_labeled.parquet   : {len(segs_out):,} rows")

    # Room metadata with split
    valid_room_meta["split"] = valid_room_meta["live_id"].map(_assign_split)
    rooms_path = os.path.join(args.out_dir, "rooms_split.parquet")
    valid_room_meta.to_parquet(rooms_path, index=False)
    print(f"  rooms_split.parquet        : {len(valid_room_meta):,} rows")

    # Filtered interactions (user history — useful for future experiments)
    keep_cols = [c for c in ["user_id", "live_id", "play_start_ts", "play_end_ts",
                              "like_cnt", "comment_cnt", "send_gift_cnt"]
                 if c in live_filt.columns]
    live_save = live_filt[live_filt["live_id"].isin(valid_rooms)][keep_cols]
    ints_path = os.path.join(args.out_dir, "interactions_filtered.parquet")
    live_save.to_parquet(ints_path, index=False)
    print(f"  interactions_filtered.parquet: {len(live_save):,} rows")

    # Statistics
    stats = {
        "top_k_rooms_requested": args.top_k,
        "min_segs": args.min_segs,
        "min_viewers": args.min_viewers,
        "n_valid_rooms": int(len(valid_rooms)),
        "n_train_rooms": int(len(train_rooms)),
        "n_val_rooms":   int(len(val_rooms)),
        "n_test_rooms":  int(len(test_rooms)),
        "n_total_segments": int(len(segs_out)),
        "n_train_segments": int((segs_out["split"] == "train").sum()),
        "n_val_segments":   int((segs_out["split"] == "val").sum()),
        "n_test_segments":  int((segs_out["split"] == "test").sum()),
        "n_users": int(len(valid_users)),
        "segment_emb_dim": 128,
        "stats_feat_dim": STAT_FEAT_DIM,
        "stats_feat_cols": STAT_FEAT_COLS,
        "train_frac": args.train_frac,
        "val_frac":   args.val_frac,
    }
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  stats.json")
    print()
    print(json.dumps(stats, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
