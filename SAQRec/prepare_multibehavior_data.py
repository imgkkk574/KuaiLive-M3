"""Build the independent Table-2 multi-behavior interface from SAQRec events.

This script intentionally consumes the already processed ``events.parquet``
instead of reading raw KLM3 tables.  It leaves the SAQRec event data unchanged
and writes an expanded, chronologically ordered feedback-token table for
FeedRec and the ``*_M`` baselines only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLICK = 1
SATISFIED = 2
DISSATISFIED = 3
TYPE_NAMES = {CLICK: "CLICK", SATISFIED: "SATISFIED", DISSATISFIED: "DISSATISFIED"}


def build_multibehavior_events(data_dir: str | Path, output_file: str | Path | None = None) -> dict:
    """Expand each click into a CLICK token and observed surveys into S+/S- tokens.

    A survey token is ordered immediately *after* its source click.  Therefore
    it cannot be visible when predicting that click, while it is available for
    every later user interaction.  Split labels belong to the source click;
    only CLICK tokens will ever be used as ranking targets.
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file) if output_file else data_dir / "feedrec_events.parquet"
    events = pd.read_parquet(data_dir / "events.parquet")
    required = {"event_id", "user_id", "author_id", "timestamp", "split", "observed", "satisfaction"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events.parquet missing required columns: {sorted(missing)}")
    events = events.sort_values(["user_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)

    columns = ["event_id", "user_id", "author_id", "timestamp", "split"]
    click = events[columns].copy()
    click["feedback_type"] = CLICK
    click["feedback_order"] = 0
    click["is_click_target"] = 1

    observed = events.loc[events["observed"].eq(1), columns + ["satisfaction"]].copy()
    if not observed["satisfaction"].isin([0.0, 1.0]).all():
        raise ValueError("observed questionnaire events must have binary satisfaction labels")
    survey = observed.drop(columns="satisfaction")
    survey["feedback_type"] = observed["satisfaction"].map({1.0: SATISFIED, 0.0: DISSATISFIED}).astype("int8")
    survey["feedback_order"] = 1
    survey["is_click_target"] = 0

    feedback = pd.concat([click, survey], ignore_index=True)
    feedback = feedback.sort_values(
        ["user_id", "timestamp", "event_id", "feedback_order"], kind="stable"
    ).reset_index(drop=True)
    feedback.insert(0, "feedback_id", range(len(feedback)))
    feedback["feedback_type"] = feedback["feedback_type"].astype("int8")
    feedback["feedback_order"] = feedback["feedback_order"].astype("int8")
    feedback["is_click_target"] = feedback["is_click_target"].astype("int8")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feedback.to_parquet(output_file, index=False)

    source_q = events.loc[events["observed"].eq(1)]
    audit = {
        "source_events_file": str(data_dir / "events.parquet"),
        "feedback_events_file": str(output_file),
        "click_tokens": int(len(click)),
        "satisfied_tokens": int((survey["feedback_type"] == SATISFIED).sum()),
        "dissatisfied_tokens": int((survey["feedback_type"] == DISSATISFIED).sum()),
        "total_feedback_tokens": int(len(feedback)),
        "questionnaire_tokens": int(len(survey)),
        "questionnaire_users": int(source_q["user_id"].nunique()),
        "users_with_satisfied_feedback": int(source_q.loc[source_q["satisfaction"].eq(1), "user_id"].nunique()),
        "users_with_dissatisfied_feedback": int(source_q.loc[source_q["satisfaction"].eq(0), "user_id"].nunique()),
        "feedback_type_codes": TYPE_NAMES,
        "ordering": "(user_id, timestamp, event_id, feedback_order); CLICK=0, questionnaire=1",
        "click_target_rule": "Only CLICK tokens are train/valid/test ranking targets.",
        "split_counts": {
            split: int(((feedback["split"] == split) & feedback["is_click_target"].eq(1)).sum())
            for split in ("train", "valid", "test")
        },
    }
    audit_path = output_file.with_name("feedrec_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FeedRec/Table-2 multi-behavior tokens from processed SAQRec data.")
    parser.add_argument("--data_dir", required=True, help="Directory containing events.parquet from SAQRec/preprocess.py")
    parser.add_argument("--output_file", default=None, help="Defaults to <data_dir>/feedrec_events.parquet")
    args = parser.parse_args()
    print(json.dumps(build_multibehavior_events(args.data_dir, args.output_file), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
