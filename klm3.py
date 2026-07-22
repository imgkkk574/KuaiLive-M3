"""
klm3.py — KuaiLive-M3 Dataset Loader
======================================
Unified dataset reader and model-side interface for the KuaiLive-M3 dataset.

Quick start
-----------
    from klm3 import KLM3Config, KLM3Dataset

    # Minimal: core live domain only
    cfg = KLM3Config()
    ds  = KLM3Dataset('/path/to/kuailive-m3', cfg)

    # Full cross-domain + mm features
    cfg = KLM3Config(load_video=True, load_mm_live=True, load_mm_video=True)
    ds  = KLM3Dataset.from_config('/path/to/kuailive-m3', cfg)

    # From yaml / dict
    ds  = KLM3Dataset.from_yaml('/path/to/kuailive-m3', 'config.yaml')
    ds  = KLM3Dataset.from_dict('/path/to/kuailive-m3', {'load_video': True})
"""

from __future__ import annotations

import os
import glob
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class KLM3Config:
    """
    Loading configuration for KLM3Dataset.

    Core live domain is always loaded:
        live_interaction, live_room_meta, live_room_set,
        author_profile, user_id_set

    Optional modules
    ----------------
    load_behaviors     : live_comment, live_like, live_share
                         (fine-grained timestamped behavior tables)
    load_video         : photo_interaction, photo_meta, photo_id_set, photo_tag
    load_mm_live       : live_emb_64 (64-dim per-room aggregated embedding)
    load_segment_emb   : live_emb_128_ts/ (128-dim segment-level, ~88M rows — heavy)
                         Requires load_mm_live=True
    load_mm_video      : photo_emb_128 (128-dim video embedding)
                         Requires load_video=True
    load_questionnaire : live_questionnaire (in-room survey responses)
    load_negative      : live_show (impression / negative sample table)
    """
    load_behaviors    : bool = True
    load_video        : bool = False
    load_mm_live      : bool = False
    load_segment_emb  : bool = False   # only effective when load_mm_live=True
    load_mm_video     : bool = False   # only effective when load_video=True
    load_questionnaire: bool = False
    load_negative     : bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KLM3Config":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> "KLM3Config":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_json(cls, path: str) -> "KLM3Config":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ======================================================================
# Dataset
# ======================================================================

class KLM3Dataset:
    """
    KuaiLive-M3 dataset reader with lazy loading and model-side query interface.

    File coverage
    -------------
    Always loaded (core live domain):
        author_profile.csv            author features
        user_id_set.csv               full user ID set
        live_room_set.csv             full live room ID set
        live_interaction.csv          main interaction table
        live_room_meta.parquet        live room meta (title, timestamps, duration)

    load_behaviors=True (default):
        live_comment.csv              text comments with timestamps
        live_like.csv                 like events with timestamps
        live_share.csv                share events with timestamps

    load_video=True:
        photo_interaction.csv         user-video interaction table
        photo_meta.parquet            video meta (duration, title, OCR text)
        photo_id_set.csv              full video ID set
        photo_tag.csv                 4-level category tags

    load_mm_live=True:
        live_emb_64.parquet           64-dim per-room aggregated embedding

    load_segment_emb=True (+load_mm_live):
        live_emb_128_ts/              128-dim segment-level embeddings
            live_emb_128_ts_part*.parquet  (18 shards, ~88M rows)
            fields: live_stream_id, segment_id, feature_128, timestamp

    load_mm_video=True (+load_video):
        photo_emb_128.parquet         128-dim video embedding
        photo_play.parquet            per-play-event interaction table
                                      (~111M rows, ms-level timestamps)
                                      fields: user_id, photo_id, author_id,
                                              enter_timestamp, leave_timestamp,
                                              enter_play_type, is_complete_play,
                                              like_status_type,
                                              is_follow_before_play, is_follow_after_play

    load_questionnaire=True:
        live_questionnaire.csv        in-room questionnaire responses

    load_negative=True:
        live_show.parquet             impression / negative-sample table
    """

    # ------------------------------------------------------------------
    # File registry
    # ------------------------------------------------------------------
    _CSV_TABLES = {
        "author_profile"   : "author_profile.csv",
        "user_id_set"      : "user_id_set.csv",
        "live_room_set"    : "live_room_set.csv",
        "live_interaction" : "live_interaction.csv",
        "live_comment"     : "live_comment.csv",
        "live_like"        : "live_like.csv",
        "live_share"       : "live_share.csv",
        "live_questionnaire": "live_questionnaire.csv",
        "photo_interaction": "photo_interaction.csv",
        "photo_id_set"     : "photo_id_set.csv",
        "photo_tag"        : "photo_tag.csv",
    }
    _PARQUET_TABLES = {
        "live_room_meta"   : "live_room_meta.parquet",
        "live_show"        : "live_show.parquet",
        "live_emb_64"      : "live_emb_64.parquet",
        "photo_meta"       : "photo_meta.parquet",
        "photo_emb_128"    : "photo_emb_128.parquet",
        "photo_play"       : "photo_play.parquet",
    }
    _PARSE_DATES = {
        "live_interaction"  : ["live_play_start_timestamp", "live_play_end_timestamp"],
        "live_comment"      : ["comment_ts"],
        "live_like"         : ["like_timestamp"],
        "live_share"        : ["share_timestamp"],
        "live_room_meta"    : ["start_timestamp", "end_timestamp"],
        "live_show"         : ["show_timestamp"],
        "photo_play"        : ["enter_timestamp", "leave_timestamp"],
    }
    _SEGMENT_EMB_DIR     = "live_emb_128_ts"
    _SEGMENT_EMB_PATTERN = "live_emb_128_ts_part*.parquet"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data_dir: str,
        config: Optional[KLM3Config] = None,
        verbose: bool = True,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.config   = config or KLM3Config()
        self.verbose  = verbose

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        # Validate dependent flags
        if self.config.load_segment_emb and not self.config.load_mm_live:
            logger.warning("load_segment_emb=True requires load_mm_live=True; forcing it.")
            self.config.load_mm_live = True
        if self.config.load_mm_video and not self.config.load_video:
            logger.warning("load_mm_video=True requires load_video=True; forcing it.")
            self.config.load_video = True

        self._cache:       Dict[str, pd.DataFrame] = {}
        self._emb_matrix:  Dict[str, np.ndarray]   = {}   # name → (N, D) float32
        self._emb_index:   Dict[str, Dict]          = {}   # name → {id: row_idx}

        if verbose:
            self._print_config()

    @classmethod
    def from_config(cls, data_dir: str, config: KLM3Config,
                    verbose: bool = True) -> "KLM3Dataset":
        return cls(data_dir, config, verbose)

    @classmethod
    def from_dict(cls, data_dir: str, d: dict,
                  verbose: bool = True) -> "KLM3Dataset":
        return cls(data_dir, KLM3Config.from_dict(d), verbose)

    @classmethod
    def from_yaml(cls, data_dir: str, yaml_path: str,
                  verbose: bool = True) -> "KLM3Dataset":
        return cls(data_dir, KLM3Config.from_yaml(yaml_path), verbose)

    @classmethod
    def from_json(cls, data_dir: str, json_path: str,
                  verbose: bool = True) -> "KLM3Dataset":
        return cls(data_dir, KLM3Config.from_json(json_path), verbose)

    # ------------------------------------------------------------------
    # Core live domain properties (always available)
    # ------------------------------------------------------------------

    @property
    def author_profile(self) -> pd.DataFrame:
        """Author features. Cols: author_id, is_photo_author, is_live_author,
        gender, age_segment, fans_user_num (bucketed)."""
        return self._table("author_profile")

    @property
    def user_id_set(self) -> pd.DataFrame:
        """Full user ID set. Col: user_id (~22K rows)."""
        return self._table("user_id_set")

    @property
    def live_room_set(self) -> pd.DataFrame:
        """Full live room ID set. Col: live_id (~6.6M rows)."""
        return self._table("live_room_set")

    @property
    def live_interaction(self) -> pd.DataFrame:
        """Main user-live interaction table (~35M rows).
        Cols: user_id, live_id, author_id,
              live_play_start_timestamp, live_play_end_timestamp, p_date,
              live_source_category, enter_live_action, is_auto_play,
              is_follow_enter, is_follow_leave, play_duration,
              like_cnt, comment_cnt, send_gift_cnt, send_gift_num,
              follow_author_cnt, cancel_follow_author_cnt,
              share_cnt, report_live_cnt."""
        return self._table("live_interaction")

    @property
    def live_room_meta(self) -> pd.DataFrame:
        """Live room metadata (~6.6M rows).
        Cols: live_id, author_id, live_name,
              start_timestamp, end_timestamp, live_duration (ms)."""
        return self._table("live_room_meta")

    # ------------------------------------------------------------------
    # Fine-grained behavior tables (load_behaviors)
    # ------------------------------------------------------------------

    @property
    def live_comment(self) -> pd.DataFrame:
        """Text comments (~8.3M rows). Requires load_behaviors=True.
        Cols: live_stream_id, user_id, author_id, content, comment_ts."""
        self._require("load_behaviors", "live_comment")
        return self._table("live_comment")

    @property
    def live_like(self) -> pd.DataFrame:
        """Like events with timestamps (~49M rows). Requires load_behaviors=True.
        Cols: user_id, live_id, like_timestamp."""
        self._require("load_behaviors", "live_like")
        return self._table("live_like")

    @property
    def live_share(self) -> pd.DataFrame:
        """Share events (~221K rows). Requires load_behaviors=True.
        Cols: user_id, live_id, author_id, share_timestamp, is_share_success."""
        self._require("load_behaviors", "live_share")
        return self._table("live_share")

    # ------------------------------------------------------------------
    # Optional: questionnaire
    # ------------------------------------------------------------------

    @property
    def live_questionnaire(self) -> pd.DataFrame:
        """In-room questionnaire responses (~25K rows).
        Requires load_questionnaire=True.
        Cols: author_id, live_stream_id, user_id,
              select_option, second_select_option."""
        self._require("load_questionnaire", "live_questionnaire")
        return self._table("live_questionnaire")

    # ------------------------------------------------------------------
    # Optional: negative samples
    # ------------------------------------------------------------------

    @property
    def live_show(self) -> pd.DataFrame:
        """Impression/negative-sample table (~92M rows).
        Requires load_negative=True.
        Cols: live_id, user_id, author_id, show_timestamp."""
        self._require("load_negative", "live_show")
        return self._table("live_show")

    # ------------------------------------------------------------------
    # Optional: video domain
    # ------------------------------------------------------------------

    @property
    def photo_interaction(self) -> pd.DataFrame:
        """User-video interaction table (~55M rows). Requires load_video=True.
        Cols: user_id, photo_id, author_id, show_cnt, complete_play_cnt,
              play_progress, like_cnt, cancel_like_cnt,
              direct_comment_cnt, reply_comment_cnt, comment_stay_duration,
              follow_cnt, cancel_follow_cnt, share_cnt."""
        self._require("load_video", "photo_interaction")
        return self._table("photo_interaction")

    @property
    def photo_meta(self) -> pd.DataFrame:
        """Video metadata (~6.7M rows). Requires load_video=True.
        Cols: photo_id, author_id, duration (ms),
              display_type, cover_title, video_texts."""
        self._require("load_video", "photo_meta")
        return self._table("photo_meta")

    @property
    def photo_id_set(self) -> pd.DataFrame:
        """Full video ID set (~6.7M rows). Requires load_video=True.
        Col: photo_id."""
        self._require("load_video", "photo_id_set")
        return self._table("photo_id_set")

    @property
    def photo_tag(self) -> pd.DataFrame:
        """4-level category tags (~6.2M rows). Requires load_video=True.
        Cols: photo_id, first/second/third/fourth_level_category_name."""
        self._require("load_video", "photo_tag")
        return self._table("photo_tag")

    # ------------------------------------------------------------------
    # Optional: mm features
    # ------------------------------------------------------------------

    @property
    def live_emb_64(self) -> pd.DataFrame:
        """Per-room aggregated embedding (~6.5M rows). Requires load_mm_live=True.
        Cols: live_id, embedding (list[float], dim=64)."""
        self._require("load_mm_live", "live_emb_64")
        return self._table("live_emb_64")

    @property
    def photo_emb_128(self) -> pd.DataFrame:
        """Per-video embedding (~5.5M rows). Requires load_mm_video=True.
        Cols: photo_id, feature (list[float], dim=128)."""
        self._require("load_mm_video", "photo_emb_128")
        return self._table("photo_emb_128")

    @property
    def photo_play(self) -> pd.DataFrame:
        """Per-play-event video interaction table (~111M rows).
        Requires load_video=True.
        Cols: user_id, photo_id, author_id,
              enter_timestamp, leave_timestamp (ms-level),
              enter_play_type, is_complete_play, like_status_type,
              is_follow_before_play, is_follow_after_play."""
        self._require("load_video", "photo_play")
        return self._table("photo_play")

    @property
    def live_emb_128_ts(self) -> pd.DataFrame:
        """Segment-level embeddings with timestamps (~88M rows).
        Requires load_segment_emb=True.
        Cols: live_stream_id, segment_id, feature_128 (dim=128), timestamp.
        WARNING: heavy (~10GB RAM). Prefer get_segments() for targeted access."""
        self._require("load_segment_emb", "live_emb_128_ts")
        return self._load_segment_emb_all()

    # ------------------------------------------------------------------
    # Model-side query interface
    # ------------------------------------------------------------------

    def get_live_features(
        self,
        live_ids: List[int],
        include_emb: bool = True,
    ) -> pd.DataFrame:
        """
        Return live room feature rows for the given live_ids.
        If load_mm_live=True and include_emb=True, appends 'embedding' column.
        """
        idx = self._build_index("live_room_meta", "live_id")
        rows = [idx[i] for i in live_ids if i in idx]
        df   = self.live_room_meta.iloc[rows].copy()
        if include_emb and self.config.load_mm_live:
            embs = self.get_live_embeddings(df["live_id"].tolist())
            df   = df.reset_index(drop=True)
            df["embedding"] = list(embs)
        return df

    def get_author_features(self, author_ids: List[int]) -> pd.DataFrame:
        """Return author profile rows for the given author_ids."""
        idx  = self._build_index("author_profile", "author_id")
        rows = [idx[i] for i in author_ids if i in idx]
        return self.author_profile.iloc[rows].copy()

    def get_user_history(
        self,
        user_ids: List[int],
        domain: str = "live",
    ) -> Dict[int, pd.DataFrame]:
        """
        Return interaction history per user.

        Parameters
        ----------
        user_ids : list[int]
        domain   : 'live' | 'video' | 'all'

        Returns
        -------
        dict  user_id → DataFrame of interaction rows
        """
        result: Dict[int, pd.DataFrame] = {}
        uid_set = set(user_ids)

        if domain in ("live", "all"):
            live_grp = self.live_interaction[
                self.live_interaction["user_id"].isin(uid_set)
            ].groupby("user_id")
            for uid, grp in live_grp:
                result[uid] = grp.reset_index(drop=True)

        if domain in ("video", "all") and self.config.load_video:
            photo_grp = self.photo_interaction[
                self.photo_interaction["user_id"].isin(uid_set)
            ].groupby("user_id")
            for uid, grp in photo_grp:
                if uid in result:
                    result[uid] = pd.concat([result[uid], grp], ignore_index=True)
                else:
                    result[uid] = grp.reset_index(drop=True)

        return result

    def get_live_embeddings(self, live_ids: List[int]) -> np.ndarray:
        """
        Return (N, 64) float32 embedding matrix for the given live_ids.
        Rows with unknown live_id are filled with zeros.
        Requires load_mm_live=True.
        """
        self._require("load_mm_live", "get_live_embeddings")
        mat, idx = self._build_emb_matrix("live_emb_64", "live_id", "embedding", 64)
        out = np.zeros((len(live_ids), 64), dtype=np.float32)
        for i, lid in enumerate(live_ids):
            if lid in idx:
                out[i] = mat[idx[lid]]
        return out

    def get_photo_embeddings(self, photo_ids: List[int]) -> np.ndarray:
        """
        Return (N, 128) float32 embedding matrix for the given photo_ids.
        Rows with unknown photo_id are filled with zeros.
        Requires load_mm_video=True.
        """
        self._require("load_mm_video", "get_photo_embeddings")
        mat, idx = self._build_emb_matrix("photo_emb_128", "photo_id", "feature", 128)
        out = np.zeros((len(photo_ids), 128), dtype=np.float32)
        for i, pid in enumerate(photo_ids):
            if pid in idx:
                out[i] = mat[idx[pid]]
        return out

    def get_segments(self, live_id: int) -> pd.DataFrame:
        """
        Return segment-level embeddings for a single live room,
        sorted by timestamp (ascending).
        Efficiently scans shards without loading all 88M rows.
        Requires load_mm_live=True.
        """
        self._require("load_mm_live", "get_segments")
        key = str(live_id)
        shard_dir = os.path.join(self.data_dir, self._SEGMENT_EMB_DIR)
        shards    = sorted(glob.glob(os.path.join(shard_dir, self._SEGMENT_EMB_PATTERN)))
        parts = []
        for shard in shards:
            df = pd.read_parquet(shard)
            hit = df[df["live_stream_id"].astype(str) == key]
            if len(hit) > 0:
                parts.append(hit)
        if not parts:
            return pd.DataFrame(columns=["live_stream_id", "segment_id",
                                         "feature_128", "timestamp"])
        result = pd.concat(parts, ignore_index=True)
        return result.sort_values("timestamp").reset_index(drop=True)

    def get_negatives(
        self,
        user_id: int,
        k: int = 50,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Sample k negative (shown but not interacted) live rooms for a user.
        Negatives = impressions (live_show) - interactions (live_interaction).
        Requires load_negative=True.

        Parameters
        ----------
        user_id : int
        k       : int  number of negatives to sample
        seed    : int  optional random seed
        """
        self._require("load_negative", "get_negatives")
        shown       = set(self.live_show[
            self.live_show["user_id"] == user_id]["live_id"].tolist())
        interacted  = set(self.live_interaction[
            self.live_interaction["user_id"] == user_id]["live_id"].tolist())
        neg_ids     = list(shown - interacted)
        if seed is not None:
            rng = np.random.default_rng(seed)
            neg_ids = rng.choice(neg_ids, size=min(k, len(neg_ids)),
                                 replace=False).tolist()
        else:
            neg_ids = neg_ids[:k]
        return self.live_show[
            (self.live_show["user_id"] == user_id) &
            (self.live_show["live_id"].isin(neg_ids))
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def load_all(self) -> "KLM3Dataset":
        """Eagerly load all tables configured by KLM3Config."""
        for name in self._enabled_tables():
            _ = getattr(self, name)
        return self

    def info(self) -> None:
        """Print a summary of all loaded tables and memory usage."""
        print(f"\n{'='*65}")
        print(f"KuaiLive-M3 Dataset  |  {self.data_dir}")
        print(f"{'='*65}")
        cfg = self.config.to_dict()
        for k, v in cfg.items():
            print(f"  {k:<22} = {v}")
        print(f"{'─'*65}")
        if not self._cache:
            print("  (no tables loaded yet)")
        else:
            total_mb = 0.0
            for name in sorted(self._cache):
                df  = self._cache[name]
                mb  = df.memory_usage(deep=True).sum() / 1024**2
                total_mb += mb
                print(f"  {name:<28} {len(df):>12,} rows  {mb:>8.1f} MB")
            print(f"{'─'*65}")
            print(f"  {'TOTAL':<28} {'':>12}       {total_mb:>8.1f} MB")
        print(f"{'='*65}\n")

    def __repr__(self) -> str:
        loaded = sorted(self._cache.keys())
        return (
            f"KLM3Dataset(data_dir='{self.data_dir}', "
            f"config={self.config.to_dict()}, loaded={loaded})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table(self, name: str) -> pd.DataFrame:
        """Lazy-load a single table with caching."""
        if name in self._cache:
            return self._cache[name]

        if name in self._CSV_TABLES:
            path = os.path.join(self.data_dir, self._CSV_TABLES[name])
            kw   = {"low_memory": False}
            if name in self._PARSE_DATES:
                kw["parse_dates"] = self._PARSE_DATES[name]
            reader = lambda: pd.read_csv(path, **kw)
        elif name in self._PARQUET_TABLES:
            path   = os.path.join(self.data_dir, self._PARQUET_TABLES[name])
            reader = lambda: pd.read_parquet(path)
        else:
            raise KeyError(f"Unknown table: {name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        t0 = time.time()
        if self.verbose:
            print(f"  [klm3] Loading {name} ...", end=" ", flush=True)
        df = reader()
        elapsed = time.time() - t0
        if self.verbose:
            print(f"{len(df):,} rows  ({elapsed:.1f}s)")

        self._cache[name] = df
        return df

    def _load_segment_emb_all(self) -> pd.DataFrame:
        if "live_emb_128_ts" in self._cache:
            return self._cache["live_emb_128_ts"]
        shard_dir = os.path.join(self.data_dir, self._SEGMENT_EMB_DIR)
        shards    = sorted(glob.glob(os.path.join(shard_dir, self._SEGMENT_EMB_PATTERN)))
        if not shards:
            raise FileNotFoundError(f"No segment shards found in {shard_dir}")
        if self.verbose:
            print(f"  [klm3] Loading live_emb_128_ts ({len(shards)} shards) ...")
        t0, parts = time.time(), []
        for i, s in enumerate(shards, 1):
            parts.append(pd.read_parquet(s))
            if self.verbose:
                print(f"         shard {i}/{len(shards)}", end="\r")
        df = pd.concat(parts, ignore_index=True)
        if self.verbose:
            print(f"\n  [klm3] live_emb_128_ts: {len(df):,} rows  ({time.time()-t0:.1f}s)")
        self._cache["live_emb_128_ts"] = df
        return df

    def _build_index(self, table_name: str, key_col: str) -> Dict:
        """Build and cache a {key_value: row_position} index for a table."""
        cache_key = f"_idx_{table_name}_{key_col}"
        if cache_key not in self._emb_index:
            df = self._table(table_name)
            self._emb_index[cache_key] = {
                v: i for i, v in enumerate(df[key_col].tolist())
            }
        return self._emb_index[cache_key]

    def _build_emb_matrix(
        self,
        table_name: str,
        id_col: str,
        emb_col: str,
        dim: int,
    ):
        """Build and cache a (N, dim) float32 matrix + id→row index."""
        if table_name not in self._emb_matrix:
            df  = self._table(table_name)
            mat = np.vstack(df[emb_col].values).astype(np.float32)
            idx = {v: i for i, v in enumerate(df[id_col].tolist())}
            self._emb_matrix[table_name] = mat
            self._emb_index[table_name]  = idx
        return self._emb_matrix[table_name], self._emb_index[table_name]

    def _require(self, flag: str, name: str) -> None:
        if not getattr(self.config, flag):
            raise RuntimeError(
                f"'{name}' requires config.{flag}=True. "
                f"Re-initialise with KLM3Config({flag}=True)."
            )

    def _enabled_tables(self) -> List[str]:
        """List all attribute names enabled by current config."""
        always = ["author_profile", "user_id_set", "live_room_set",
                  "live_interaction", "live_room_meta"]
        optional = []
        cfg = self.config
        if cfg.load_behaviors:
            optional += ["live_comment", "live_like", "live_share"]
        if cfg.load_questionnaire:
            optional += ["live_questionnaire"]
        if cfg.load_negative:
            optional += ["live_show"]
        if cfg.load_video:
            optional += ["photo_interaction", "photo_meta",
                         "photo_id_set", "photo_tag", "photo_play"]
        if cfg.load_mm_live:
            optional += ["live_emb_64"]
        if cfg.load_segment_emb:
            optional += ["live_emb_128_ts"]
        if cfg.load_mm_video:
            optional += ["photo_emb_128"]
        return always + optional

    def _print_config(self) -> None:
        print(f"\n[KLM3Dataset] data_dir: {self.data_dir}")
        enabled = self._enabled_tables()
        print(f"  Enabled tables ({len(enabled)}): {enabled}")
        print(f"  (lazy loading — tables are read on first access)\n")
