"""
dataset_highlight.py — PyTorch Dataset for Highlight Prediction
===============================================================

Loads the preprocessed highlight_data/segments_labeled.parquet and exposes
per-room sequences for training causal / hierarchical highlight models.

Each sample corresponds to one live room with one-step-ahead alignment:
    embeddings  : float32 tensor  [K-1, 128] (segments 0 ... K-2)
    hl_scores   : float32 tensor  [K-1]      (scores of segments 1 ... K-1)
    hl_binary   : int8   tensor   [K-1]      (labels of segments 1 ... K-1)
    lengths     : int              K-1        (number of predictions)

Output position t consumes segment t and is supervised by segment t+1.
Rooms with fewer than two segments are discarded.

When batching variable-length sequences, use the provided `collate_fn` which
right-pads all tensors to the longest sequence in the batch and returns a
padding mask.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HighlightDataset(Dataset):
    """
    One sample = one live room's segment sequence.

    Parameters
    ----------
    data_dir : str
        Directory produced by preprocess_highlight.py
        (contains segments_labeled.parquet and stats.json)
    split : str
        'train', 'val', or 'test'
    max_seq_len : int
        Truncate prediction sequences longer than this (rare very-long streams).
        0 = no truncation.
    """

    # These describe the observed/current segment and must not be taken from
    # target segment t+1. All other stat_* columns are causal aggregates whose
    # row t+1 value contains behavior observed through segment t.
    _POSITIONAL_STAT_COLS = frozenset({"stat_rel_pos", "stat_seg_dur_log"})

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_seq_len: int = 200,
        input_mode: str = "embedding",
        use_stats: bool = False,
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split!r}"
        assert input_mode in ("embedding", "stats"), \
            f"input_mode must be 'embedding' or 'stats', got {input_mode!r}"
        assert not (use_stats and input_mode != "embedding"), \
            "use_stats=True requires input_mode='embedding' (stats are concatenated onto embeddings)"
        self.split       = split
        self.max_seq_len = max_seq_len
        self.input_mode  = input_mode
        self.use_stats   = use_stats

        # Load stats
        stats_path = os.path.join(data_dir, "stats.json")
        with open(stats_path) as f:
            self.stats: Dict = json.load(f)
        # Input dimension depends on the mode.
        if input_mode == "embedding":
            self.emb_dim: int = self.stats.get("segment_emb_dim", 128)
        else:
            self.emb_dim: int = self.stats.get("stats_feat_dim", 10)
        # Stat columns are needed both for the 'stats' input mode (the features
        # ARE the input) and for use_stats=True (features concatenated alongside
        # the embedding). Load them whenever either is active.
        self.stat_dim: int = 0
        self._stat_cols: List[str] = []
        if input_mode == "stats" or use_stats:
            self.stat_dim = self.stats.get("stats_feat_dim", 10)
            self._stat_cols = self.stats.get(
                "stats_feat_cols",
                [f"stat_{i}" for i in range(self.stat_dim)],
            )

        # Load parquet and filter to the requested split
        seg_path = os.path.join(data_dir, "segments_labeled.parquet")
        df = pd.read_parquet(seg_path)
        df = df[df["split"] == split].copy()

        # Sort within each room by timestamp
        df = df.sort_values(["live_stream_id", "timestamp"]).reset_index(drop=True)

        # Build per-room index: list of (embeddings, hl_scores, hl_binary, [stats])
        self._room_ids: List[int] = []
        self._embeddings: List[np.ndarray] = []
        self._hl_scores:  List[np.ndarray] = []
        self._hl_binary:  List[np.ndarray] = []
        self._stats: Optional[List[np.ndarray]] = [] if use_stats else None

        for room_id, grp in df.groupby("live_stream_id", sort=False):
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue

            # N+1 raw segments yield N one-step-ahead predictions.
            if max_seq_len > 0:
                grp = grp.iloc[:max_seq_len + 1]

            raw_stats = None
            if input_mode == "stats" or use_stats:
                raw_stats = grp[self._stat_cols].to_numpy(dtype=np.float32)
                aligned_stats = self._align_stats_for_next_segment(raw_stats)

            if input_mode == "embedding":
                # feature_128: may be list-of-lists or list-of-ndarrays
                all_feats = np.vstack(grp["feature_128"].values).astype(np.float32)
                feats = all_feats[:-1]
            else:
                # Stats observed through segment t predict target segment t+1.
                feats = aligned_stats

            # Segment t is the input; segment t+1 is the prediction target.
            hl_score  = grp["hl_score"].to_numpy(dtype=np.float32)[1:]
            hl_binary = grp["hl_binary"].to_numpy(dtype=np.int64)[1:]

            self._room_ids.append(int(room_id))
            self._embeddings.append(feats)
            self._hl_scores.append(hl_score)
            self._hl_binary.append(hl_binary)
            if use_stats:
                self._stats.append(aligned_stats)

        self._n_rooms = len(self._room_ids)

    def _align_stats_for_next_segment(self, raw_stats: np.ndarray) -> np.ndarray:
        """Return stats available after t, aligned with target segment t+1.

        The preprocessed behavior aggregates in row k use segments 0..k-1.
        Therefore row t+1 contains exactly the behavior history available after
        observing segment t. Positional features remain from row t so no target
        segment duration or position is exposed.
        """
        aligned = raw_stats[:-1].copy()
        for col_idx, col in enumerate(self._stat_cols):
            if col not in self._POSITIONAL_STAT_COLS:
                aligned[:, col_idx] = raw_stats[1:, col_idx]
        return aligned

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_rooms

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        emb    = torch.from_numpy(self._embeddings[idx])  # [L, emb_dim], segment t
        scores = torch.from_numpy(self._hl_scores[idx])   # [L], target segment t+1
        binary = torch.from_numpy(self._hl_binary[idx])   # [L], target segment t+1
        out = {
            "embeddings": emb,     # [L, emb_dim]
            "hl_scores":  scores,  # [L]
            "hl_binary":  binary,  # [L]
            "length":     emb.shape[0],
            "room_id":    self._room_ids[idx],
        }
        if self.use_stats:
            out["stats"] = torch.from_numpy(self._stats[idx])   # [L, stat_dim]
        return out

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def seq_lengths(self) -> np.ndarray:
        """Return array of sequence lengths (one per room)."""
        return np.array([emb.shape[0] for emb in self._embeddings])

    def describe(self) -> None:
        lengths = self.seq_lengths()
        print(f"HighlightDataset [{self.split}]")
        print(f"  Rooms    : {self._n_rooms:,}")
        print(f"  Segments : {lengths.sum():,}")
        print(f"  Seq len  : min={lengths.min()}  mean={lengths.mean():.1f}  "
              f"median={np.median(lengths):.0f}  max={lengths.max()}")
        all_scores = np.concatenate(self._hl_scores)
        print(f"  HL score : mean={all_scores.mean():.3f}  std={all_scores.std():.3f}")
        all_binary = np.concatenate(self._hl_binary)
        print(f"  HL rate  : {all_binary.mean():.3f}  (fraction of highlight segments)")


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (variable-length padding)
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(
    batch: List[Dict],
) -> Dict[str, torch.Tensor]:
    """
    Pad a list of variable-length room samples to the longest in the batch.

    Returns
    -------
    dict with keys:
        embeddings : [B, T, emb_dim]   float32  — padded with zeros
        hl_scores  : [B, T]            float32  — padded with -1.0 (ignored in loss)
        hl_binary  : [B, T]            int64    — padded with -1    (ignored in loss)
        pad_mask   : [B, T]            bool     — True = padded position (ignore)
        lengths    : [B]               int64    — actual sequence lengths
        room_ids   : [B]               int64
    """
    B   = len(batch)
    T   = max(s["length"] for s in batch)
    dim = batch[0]["embeddings"].shape[1]
    has_stats = "stats" in batch[0]
    stat_dim = batch[0]["stats"].shape[1] if has_stats else 0

    emb_out    = torch.zeros(B, T, dim,    dtype=torch.float32)
    score_out  = torch.full((B, T),  -1.0, dtype=torch.float32)
    binary_out = torch.full((B, T),  -1,   dtype=torch.long)
    pad_mask   = torch.ones(B, T,           dtype=torch.bool)   # True = pad
    lengths    = torch.zeros(B,             dtype=torch.long)
    room_ids   = torch.zeros(B,             dtype=torch.long)
    stats_out  = (torch.zeros(B, T, stat_dim, dtype=torch.float32)
                  if has_stats else None)

    for i, s in enumerate(batch):
        k = s["length"]
        emb_out[i, :k]    = s["embeddings"]
        score_out[i, :k]  = s["hl_scores"]
        binary_out[i, :k] = s["hl_binary"]
        pad_mask[i, :k]   = False  # real positions
        lengths[i]        = k
        room_ids[i]       = s["room_id"]
        if has_stats:
            stats_out[i, :k] = s["stats"]

    out = {
        "embeddings": emb_out,
        "hl_scores":  score_out,
        "hl_binary":  binary_out,
        "pad_mask":   pad_mask,
        "lengths":    lengths,
        "room_ids":   room_ids,
    }
    if has_stats:
        out["stats"] = stats_out
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    max_seq_len: int = 200,
    num_workers: int = 4,
    seed: int = 42,
    input_mode: str = "embedding",
    use_stats: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders.

    Parameters
    ----------
    data_dir    : output directory of preprocess_highlight.py
    batch_size  : rooms per batch
    max_seq_len : maximum number of next-segment predictions (0 = no limit)
    num_workers : DataLoader workers
    seed        : random seed for train shuffle
    input_mode  : 'embedding' (segment embeddings) or 'stats' (causal stats)
    use_stats   : if True (requires input_mode='embedding'), also load the
                  per-segment stat vector into each batch's 'stats' key for the
                  model's additive stat branch.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    def _make(split: str, shuffle: bool) -> DataLoader:
        ds = HighlightDataset(data_dir, split, max_seq_len=max_seq_len,
                              input_mode=input_mode, use_stats=use_stats)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=lambda w: np.random.seed(seed + w),
        )

    return _make("train", shuffle=True), _make("val", shuffle=False), _make("test", shuffle=False)
