"""Standalone data interface for Table-2 questionnaire-aware baselines."""
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
CLICK = 1
SATISFIED = 2
DISSATISFIED = 3


@dataclass
class MultiBehaviorBundle:
    """Arrays read exclusively from ``feedrec_events.parquet``.

    This is deliberately independent of ``saqrec.data.DataBundle``.  The
    original SAQRec pipeline keeps using ``events.parquet`` unchanged.
    """

    uid: np.ndarray
    iid: np.ndarray
    feedback_type: np.ndarray
    is_click_target: np.ndarray
    feedback_id: np.ndarray
    user_start: np.ndarray
    split_indices: Dict[str, np.ndarray]
    seen_items: Tuple[np.ndarray, ...]
    n_users: int
    n_items: int


def load_multibehavior_bundle(data_dir: str | Path) -> MultiBehaviorBundle:
    path = Path(data_dir) / "feedrec_events.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run SAQRec/prepare_multibehavior_data.py --data_dir {Path(data_dir)} first"
        )
    events = pd.read_parquet(path)
    required = {"feedback_id", "user_id", "author_id", "feedback_type", "is_click_target", "split"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"feedrec_events.parquet missing required columns: {sorted(missing)}")
    events = events.sort_values("feedback_id", kind="stable").reset_index(drop=True)
    uid, users = pd.factorize(events["user_id"], sort=False)
    iid, items = pd.factorize(events["author_id"], sort=False)
    uid = uid.astype(np.int32) + 1
    iid = iid.astype(np.int32) + 1
    kinds = events["feedback_type"].to_numpy(dtype=np.uint8, copy=True)
    targets = events["is_click_target"].to_numpy(dtype=np.uint8, copy=True)
    if not np.isin(kinds, [CLICK, SATISFIED, DISSATISFIED]).all():
        raise ValueError("feedback_type must contain only CLICK=1, SATISFIED=2, DISSATISFIED=3")
    if not np.all(kinds[targets == 1] == CLICK):
        raise ValueError("only CLICK tokens may be ranking targets")
    starts = np.empty(len(events), dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(uid[1:] != uid[:-1]) + 1, len(uid)]
    seen_items = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        starts[begin:end] = begin
        seen_items.append(np.unique(iid[begin:end]))
    split = events["split"].to_numpy()
    split_indices = {
        name: np.flatnonzero((split == name) & (targets == 1)).astype(np.int64)
        for name in ("train", "valid", "test")
    }
    return MultiBehaviorBundle(
        uid=uid, iid=iid, feedback_type=kinds, is_click_target=targets,
        feedback_id=events["feedback_id"].to_numpy(dtype=np.int64, copy=True),
        user_start=starts, split_indices=split_indices, seen_items=tuple(seen_items),
        n_users=len(users) + 1, n_items=len(items) + 1,
    )


class MultiBehaviorDataset(Dataset):
    """Click-ranking samples whose histories are mixed questionnaire tokens."""

    def __init__(self, bundle: MultiBehaviorBundle, split: str, feedback_len: int = 100,
                 num_negs: int = 2, seed: int = 1) -> None:
        if feedback_len <= 0:
            raise ValueError("feedback_len must be positive")
        self.bundle, self.split = bundle, split
        self.feedback_len, self.num_negs = feedback_len, num_negs
        self.rng = random.Random(seed)
        self.rows = bundle.split_indices[split]

    @staticmethod
    def _pad(sequence: Sequence[int], length: int) -> torch.Tensor:
        tail = list(sequence[-length:])
        return torch.tensor([PAD] * (length - len(tail)) + tail, dtype=torch.long)

    def _sample_negs(self, positive: int, uid: int, n: int) -> List[int]:
        seen = self.bundle.seen_items[uid - 1]
        if self.bundle.n_items <= 2:
            return [PAD] * n
        negatives, attempts = [], 0
        while len(negatives) < n and attempts < n * 100:
            candidate = self.rng.randrange(1, self.bundle.n_items)
            attempts += 1
            found = np.searchsorted(seen, candidate)
            is_seen = found < len(seen) and seen[found] == candidate
            if candidate != positive and not is_seen and candidate not in negatives:
                negatives.append(candidate)
        return negatives + [PAD] * (n - len(negatives))

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        start = int(self.bundle.user_start[row])
        # The source click is ordered before its questionnaire token, so the
        # strict ``:row`` prefix never leaks the target response.
        begin = max(start, row - self.feedback_len)
        ids = self.bundle.iid[begin:row]
        kinds = self.bundle.feedback_type[begin:row]
        uid = int(self.bundle.uid[row])
        return {
            "uid": torch.tensor(uid, dtype=torch.long),
            "pos": torch.tensor(int(self.bundle.iid[row]), dtype=torch.long),
            "neg": torch.tensor(self._sample_negs(int(self.bundle.iid[row]), uid, self.num_negs), dtype=torch.long),
            "mixed_his": self._pad(ids, self.feedback_len),
            "mixed_type": self._pad(kinds, self.feedback_len),
            "feedback_id": torch.tensor(int(self.bundle.feedback_id[row]), dtype=torch.long),
        }
