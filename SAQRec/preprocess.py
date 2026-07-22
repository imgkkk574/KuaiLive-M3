from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from klm3 import KLM3Config, KLM3Dataset
from saqrec.data import chronological_leave_one_out, full_period_k_core


LABELS = {"开播就推": 1.0, "适当推荐": 1.0, "打赏": 1.0, "不想再看": 0.0}


def option_column(frame: pd.DataFrame) -> str:
    for column in ("selection_option", "select_option"):
        if column in frame.columns:
            return column
    raise ValueError("questionnaire needs selection_option or select_option")


def parse_primary_option(value: object) -> str | None:
    """Extract the first-level answer from KLM3's JSON-array representation.

    KLM3 stores first-level answers such as ``[\"开播就推\"]`` in a CSV
    field.  A bare string is accepted as a defensive fallback for future data
    exports.  Empty arrays and nulls are intentionally treated as unobserved.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else None
    text = str(value).strip()
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(decoded, list):
        return str(decoded[0]).strip() if decoded else None
    return str(decoded).strip()


def prepare(data_dir: str, output_dir: str, min_interactions: int = 5) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ds = KLM3Dataset(data_dir, KLM3Config(load_behaviors=False, load_questionnaire=True), verbose=False)
    live = ds.live_interaction[["user_id", "live_id", "author_id", "live_play_start_timestamp"]].copy()
    live = live.dropna(subset=["user_id", "live_id", "author_id", "live_play_start_timestamp"])
    live["live_id"] = live["live_id"].astype(str)
    live["timestamp"] = pd.to_datetime(live["live_play_start_timestamp"], errors="coerce", utc=True)
    live = live.dropna(subset=["timestamp"])
    live["timestamp"] = (live["timestamp"].astype("int64") // 10**9).astype("int64")

    questionnaire = ds.live_questionnaire.copy()
    column = option_column(questionnaire)
    questionnaire["live_id"] = questionnaire["live_stream_id"].astype(str)
    questionnaire["serialized_option"] = questionnaire[column].astype("string")
    questionnaire["raw_option"] = questionnaire[column].map(parse_primary_option)
    questionnaire["satisfaction"] = questionnaire["raw_option"].map(LABELS)
    valid = questionnaire.dropna(subset=["user_id", "live_id", "satisfaction"])[
        ["user_id", "live_id", "author_id", "raw_option", "satisfaction"]
    ].copy()
    if valid.empty:
        raise ValueError(
            "No questionnaire option matched LABELS. Inspect raw_option_counts in the input "
            "or extend LABELS before generating an unusable SAQRec dataset."
        )
    valid = valid.rename(columns={"author_id": "questionnaire_author_id"})
    valid["questionnaire_author_key"] = valid["questionnaire_author_id"].astype("string")
    # A response has no independent timestamp.  It is attached to the live
    # interaction event; inconsistent duplicate responses are audited and not
    # treated as an observed binary label.
    response_key = valid.groupby(["user_id", "live_id"])["satisfaction"].agg(lambda x: set(x))
    conflicting_keys = response_key[response_key.map(len) > 1]
    consistent = valid.merge(conflicting_keys.rename("_conflict"), how="left", left_on=["user_id", "live_id"],
                             right_index=True)
    consistent = consistent[consistent["_conflict"].isna()].drop(columns="_conflict")
    consistent = consistent.drop_duplicates(["user_id", "live_id"], keep="last")
    labels = consistent[["user_id", "live_id", "questionnaire_author_id", "questionnaire_author_key", "satisfaction", "raw_option"]]
    events = live.merge(labels, on=["user_id", "live_id"], how="left")
    events["questionnaire_author_matches"] = (
        events["questionnaire_author_key"].isna()
        | (events["questionnaire_author_key"] == events["author_id"].astype("string"))
    )
    events["event_id"] = range(len(events))
    # `live_interaction` can contain several entries for one user in the same
    # live room.  Keep all of them as click events, but a single questionnaire
    # answer must not be copied into every repeated row.  Questionnaires do not
    # include a response timestamp, so the last interaction is the conservative
    # choice: it guarantees that a satisfaction answer never enters an earlier
    # history before the user has completed the room interaction.
    questionnaire_candidates = events[events["satisfaction"].notna()].copy()
    repeats = questionnaire_candidates.groupby(["user_id", "live_id"], sort=False).size()
    selected_questionnaire_rows = (
        questionnaire_candidates.sort_values(["user_id", "live_id", "timestamp", "event_id"])
        .groupby(["user_id", "live_id"], sort=False)
        .tail(1)
        .index
    )
    events["observed"] = 0
    events.loc[selected_questionnaire_rows, "observed"] = 1
    non_selected = events.index.difference(selected_questionnaire_rows)
    events.loc[non_selected, "satisfaction"] = float("nan")
    events.loc[non_selected, "raw_option"] = pd.NA
    events.loc[non_selected, "questionnaire_author_id"] = pd.NA
    events.loc[non_selected, "questionnaire_author_key"] = pd.NA
    events["observed"] = events["observed"].astype("int8")
    author_mismatch_events = int((
        events["observed"].eq(1) & ~events["questionnaire_author_matches"]
    ).sum())
    events = events[["event_id", "user_id", "author_id", "live_id", "timestamp", "observed", "satisfaction", "raw_option"]]
    before = len(events)
    events = full_period_k_core(events, min_interactions)
    events = chronological_leave_one_out(events)
    events.to_parquet(output / "events.parquet", index=False)
    consistent.to_parquet(output / "questionnaire_events.parquet", index=False)

    observed = events[events.observed == 1]
    pos_users = set(observed.loc[observed.satisfaction == 1, "user_id"])
    neg_users = set(observed.loc[observed.satisfaction == 0, "user_id"])
    pair_labels = observed.groupby(["user_id", "author_id"])["satisfaction"].agg(lambda x: set(x))
    audit = {
        "questionnaire_option_column": column,
        "serialized_option_counts": questionnaire["serialized_option"].value_counts(dropna=False).to_dict(),
        "raw_option_counts": questionnaire["raw_option"].value_counts(dropna=False).to_dict(),
        "mapped_option_counts": valid["raw_option"].value_counts().to_dict(),
        "raw_live_events": before,
        "events_after_full_period_k_core": len(events),
        "users_after_filter": int(events.user_id.nunique()),
        "authors_after_filter": int(events.author_id.nunique()),
        "observed_questionnaire_events_after_filter": int(len(observed)),
        "positive_events_after_filter": int((observed.satisfaction == 1).sum()),
        "negative_events_after_filter": int((observed.satisfaction == 0).sum()),
        "questionnaire_join_rate": float(len(observed) / max(1, len(valid))),
        "questionnaire_candidate_interaction_rows": int(len(questionnaire_candidates)),
        "questionnaire_duplicate_interaction_rows_removed": int(
            len(questionnaire_candidates) - len(selected_questionnaire_rows)
        ),
        "questionnaire_interactions_per_answer": {
            "p50": float(repeats.quantile(0.50)) if len(repeats) else 0.0,
            "p90": float(repeats.quantile(0.90)) if len(repeats) else 0.0,
            "p99": float(repeats.quantile(0.99)) if len(repeats) else 0.0,
            "max": int(repeats.max()) if len(repeats) else 0,
        },
        "questionnaire_author_mismatch_events": author_mismatch_events,
        "ambiguous_user_live_questionnaire_pairs": int(len(conflicting_keys)),
        "conflicting_user_author_pairs": int((pair_labels.map(len) > 1).sum()),
        "questionnaire_users": int(observed.user_id.nunique()),
        "bipolar_questionnaire_users": int(len(pos_users & neg_users)),
        "split_counts": events["split"].value_counts().to_dict(),
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build author-level KLM3 SAQRec events.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="SAQRec/data/klm3")
    parser.add_argument("--min_interactions", type=int, default=5)
    args = parser.parse_args()
    audit = prepare(args.data_dir, args.output_dir, args.min_interactions)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
