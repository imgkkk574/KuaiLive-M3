from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PAD = 0


def full_period_k_core(events: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Iteratively retain users and authors with at least `min_interactions` events.

    This deliberately follows the full-period 5-core convention used in the
    SAQRec paper.  It is run at the author, not live-room, level.
    """
    required = {"user_id", "author_id"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    result = events.copy()
    while True:
        users = result["user_id"].value_counts()
        authors = result["author_id"].value_counts()
        filtered = result[
            result["user_id"].isin(users[users >= min_interactions].index)
            & result["author_id"].isin(authors[authors >= min_interactions].index)
        ]
        if len(filtered) == len(result):
            return filtered.reset_index(drop=True)
        result = filtered


def chronological_leave_one_out(events: pd.DataFrame) -> pd.DataFrame:
    """Add split labels, retaining users with at least three interactions."""
    ordered = events.sort_values(["user_id", "timestamp", "event_id"]).copy()
    sizes = ordered.groupby("user_id")["event_id"].transform("size")
    ordered = ordered[sizes >= 3].copy()
    ordered["order"] = ordered.groupby("user_id").cumcount()
    ordered["size"] = ordered.groupby("user_id")["event_id"].transform("size")
    ordered["split"] = "train"
    ordered.loc[ordered["order"] == ordered["size"] - 2, "split"] = "valid"
    ordered.loc[ordered["order"] == ordered["size"] - 1, "split"] = "test"
    return ordered.drop(columns=["order", "size"]).reset_index(drop=True)


@dataclass
class DataBundle:
    """Compact, chronologically ordered arrays backing all dataset splits.

    The raw KLM3 event table has 34M+ rows.  Keeping Python history lists for
    every event would require hundreds of GB, so histories are derived from
    these arrays only when an example is requested.
    """
    uid: np.ndarray
    iid: np.ndarray
    observed: np.ndarray
    satisfaction: np.ndarray
    event_id: np.ndarray
    user_start: np.ndarray
    split_indices: Dict[str, np.ndarray]
    positive_positions: Tuple[np.ndarray, ...]
    negative_positions: Tuple[np.ndarray, ...]
    seen_items: Tuple[np.ndarray, ...]
    n_users: int
    n_items: int


def load_bundle(data_dir: str | Path) -> DataBundle:
    path = Path(data_dir)
    events = pd.read_parquet(path / "events.parquet")
    if "split" not in events:
        raise ValueError("events.parquet has no split column; run preprocess.py first")
    events = events.sort_values(["user_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    # factorize is substantially more memory efficient than Python dictionaries
    # for KLM3's 456K author vocabulary.  Reserve zero for padding.
    uid, users = pd.factorize(events["user_id"], sort=False)
    iid, items = pd.factorize(events["author_id"], sort=False)
    uid = uid.astype(np.int32) + 1
    iid = iid.astype(np.int32) + 1
    observed = events["observed"].to_numpy(dtype=np.uint8, copy=True)
    satisfaction = events["satisfaction"].fillna(-1.0).to_numpy(dtype=np.float32, copy=True)
    event_id = events["event_id"].to_numpy(dtype=np.int64, copy=True)
    split_indices = {
        split: np.flatnonzero(events["split"].to_numpy() == split).astype(np.int64)
        for split in ("train", "valid", "test")
    }
    starts = np.empty(len(events), dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(uid[1:] != uid[:-1]) + 1, len(uid)]
    positive, negative, seen_items = [], [], []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        starts[begin:end] = begin
        indices = np.arange(begin, end, dtype=np.int64)
        positive.append(indices[(observed[begin:end] == 1) & (satisfaction[begin:end] == 1.0)])
        negative.append(indices[(observed[begin:end] == 1) & (satisfaction[begin:end] == 0.0)])
        seen_items.append(np.unique(iid[begin:end]))
    return DataBundle(
        uid=uid, iid=iid, observed=observed, satisfaction=satisfaction, event_id=event_id,
        user_start=starts, split_indices=split_indices,
        positive_positions=tuple(positive), negative_positions=tuple(negative), seen_items=tuple(seen_items),
        n_users=len(users) + 1, n_items=len(items) + 1,
    )


class EventDataset(Dataset):
    """Chronological pointwise examples with histories strictly before the target."""

    def __init__(
        self,
        bundle: DataBundle,
        split: str,
        rec_len: int = 50,
        satis_len: int = 20,
        dissatis_len: int = 10,
        num_negs: int = 2,
        seed: int = 1,
        observed_only: bool = False,
    ) -> None:
        self.bundle, self.split = bundle, split
        self.rec_len, self.satis_len, self.dissatis_len = rec_len, satis_len, dissatis_len
        self.num_negs, self.rng = num_negs, random.Random(seed)
        rows = bundle.split_indices[split]
        if observed_only:
            rows = rows[bundle.observed[rows] == 1]
        self.rows = rows.astype(np.int64, copy=False)

    @staticmethod
    def _pad(sequence: Sequence[int], length: int) -> torch.Tensor:
        tail = list(sequence[-length:])
        return torch.tensor([PAD] * (length - len(tail)) + tail, dtype=torch.long)

    def _sample_negs(self, positive: int, uid: int, n: int) -> List[int]:
        seen = self.bundle.seen_items[uid - 1]
        if self.bundle.n_items <= 2:
            return [PAD] * n
        negatives = []
        # Rejection sampling is O(n) instead of rebuilding a 456K-item candidate
        # list for every one of 34M training examples.
        attempts = 0
        while len(negatives) < n and attempts < n * 100:
            candidate = self.rng.randrange(1, self.bundle.n_items)
            attempts += 1
            found_at = np.searchsorted(seen, candidate)
            is_seen = found_at < len(seen) and seen[found_at] == candidate
            if candidate != positive and not is_seen and candidate not in negatives:
                negatives.append(candidate)
        if len(negatives) < n:
            negatives.extend([PAD] * (n - len(negatives)))
        return negatives

    def _feedback_history(self, positions: np.ndarray, row: int, length: int) -> np.ndarray:
        end = np.searchsorted(positions, row, side="left")
        return self.bundle.iid[positions[max(0, end - length):end]]

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        start = int(self.bundle.user_start[row])
        click = self.bundle.iid[max(start, row - self.rec_len):row]
        uid = int(self.bundle.uid[row])
        satis = self._feedback_history(self.bundle.positive_positions[uid - 1], row, self.satis_len)
        dissatis = self._feedback_history(self.bundle.negative_positions[uid - 1], row, self.dissatis_len)
        negs = self._sample_negs(int(self.bundle.iid[row]), uid, self.num_negs)
        satisfaction = -1.0 if int(self.bundle.observed[row]) == 0 else float(self.bundle.satisfaction[row])
        return {
            "uid": torch.tensor(uid, dtype=torch.long),
            "pos": torch.tensor(int(self.bundle.iid[row]), dtype=torch.long),
            "neg": torch.tensor(negs, dtype=torch.long),
            "rec_his": self._pad(click, self.rec_len),
            "satis_his": self._pad(satis, self.satis_len),
            "dissatis_his": self._pad(dissatis, self.dissatis_len),
            "observed": torch.tensor(float(self.bundle.observed[row]), dtype=torch.float32),
            "satisfaction": torch.tensor(satisfaction, dtype=torch.float32),
            "event_id": torch.tensor(int(self.bundle.event_id[row]), dtype=torch.long),
        }


def metric_at_ks(ranks: Sequence[int], ks: Sequence[int]) -> Dict[str, float]:
    """HR/NDCG at requested cutoffs and one untruncated MRR."""
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("ks must contain positive integers")
    keys = ["mrr"] + [f"{metric}@{k}" for k in cutoffs for metric in ("hr", "ndcg")]
    if not ranks:
        return {key: 0.0 for key in keys}
    values = np.asarray(ranks, dtype=np.float64)
    result = {"mrr": float(np.mean(1.0 / values))}
    for k in cutoffs:
        result[f"hr@{k}"] = float(np.mean(values <= k))
        result[f"ndcg@{k}"] = float(np.mean(np.where(values <= k, 1.0 / np.log2(values + 1.0), 0.0)))
    return result
