"""Build a tiny event dataset and run SAQRec plus every baseline on CPU."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from saqrec.data import EventDataset, load_bundle


ROOT = Path(__file__).resolve().parent


def make_mock_raw(path: Path) -> None:
    rows, event_id = [], 0
    questionnaire = []
    # Every user has train/valid/test events and both answer polarities in train.
    for user in range(1, 7):
        for step in range(8):
            author = ((user + step) % 9) + 1
            live_id = f"l{event_id}"
            rows.append({"user_id": user, "author_id": author, "live_id": live_id,
                         "live_play_start_timestamp": f"2024-01-{(event_id % 27) + 1:02d}T00:00:00Z",
                         "live_play_end_timestamp": f"2024-01-{(event_id % 27) + 1:02d}T00:05:00Z"})
            if step == 1:
                # Same user re-enters the same live room.  The questionnaire
                # must still create exactly one observed satisfaction event.
                rows.append({"user_id": user, "author_id": author, "live_id": live_id,
                             "live_play_start_timestamp": f"2024-01-{(event_id % 27) + 1:02d}T00:10:00Z",
                             "live_play_end_timestamp": f"2024-01-{(event_id % 27) + 1:02d}T00:15:00Z"})
            if step == 1:
                questionnaire.append({"user_id": user, "author_id": author, "live_stream_id": live_id,
                                      "select_option": '["开播就推"]', "second_select_option": None})
            if step == 2:
                questionnaire.append({"user_id": user, "author_id": author, "live_stream_id": live_id,
                                      "select_option": '["打赏"]', "second_select_option": None})
            if step == 3:
                questionnaire.append({"user_id": user, "author_id": author, "live_stream_id": live_id,
                                      "select_option": '["不想再看"]', "second_select_option": None})
            event_id += 1
    pd.DataFrame(rows).to_csv(path / "live_interaction.csv", index=False)
    pd.DataFrame(questionnaire).to_csv(path / "live_questionnaire.csv", index=False)


def run(stage: str, data: Path, out: Path, extra=None) -> None:
    command = [sys.executable, str(ROOT / "run.py"), "--stage", stage, "--data_dir", str(data),
               "--work_dir", str(out), "--epochs", "2", "--batch_size", "8", "--dim", "16", "--cpu"]
    command.extend(extra or [])
    subprocess.run(command, check=True)
    if not (out / "best.pt").exists() or not (out / "test_metrics.json").exists():
        raise RuntimeError(f"{stage} did not create checkpoint and test metrics")


def main() -> None:
    work = ROOT / ".mock_run"
    if work.exists():
        shutil.rmtree(work)
    data = work / "data"
    raw = work / "raw"
    raw.mkdir(parents=True)
    make_mock_raw(raw)
    subprocess.run([sys.executable, str(ROOT / "preprocess.py"), "--data_dir", str(raw),
                    "--output_dir", str(data), "--min_interactions", "3"], check=True)
    subprocess.run([sys.executable, str(ROOT / "prepare_multibehavior_data.py"),
                    "--data_dir", str(data)], check=True)
    feedback = pd.read_parquet(data / "feedrec_events.parquet")
    survey = feedback[feedback["is_click_target"].eq(0)]
    for row in survey.itertuples(index=False):
        previous = feedback.iloc[row.feedback_id - 1]
        assert previous.is_click_target == 1
        assert previous.event_id == row.event_id
        assert previous.feedback_order == 0 and row.feedback_order == 1
    audit = json.loads((data / "audit.json").read_text())
    assert audit["questionnaire_option_column"] == "select_option"
    assert audit["positive_events_after_filter"] > 0
    assert audit["negative_events_after_filter"] > 0
    assert audit["mapped_option_counts"]["打赏"] > 0
    assert audit["questionnaire_author_mismatch_events"] == 0
    assert audit["questionnaire_duplicate_interaction_rows_removed"] == 6
    bundle = load_bundle(data)
    probe = EventDataset(bundle, "train", num_negs=5, seed=7)
    sample = probe[0]
    user_seen = bundle.seen_items[int(sample["uid"]) - 1]
    assert all(item == 0 or item not in user_seen for item in sample["neg"].tolist())
    base = work / "base"
    prop = work / "propensity"
    satis = work / "satisfaction"
    final = work / "saqrec"
    run("base", data, base)
    run("propensity", data, prop, ["--base_ckpt", str(base / "best.pt")])
    run("satisfaction", data, satis, ["--base_ckpt", str(base / "best.pt"), "--propensity_ckpt", str(prop / "best.pt")])
    run("saqrec", data, final, ["--base_ckpt", str(base / "best.pt"), "--satisfaction_ckpt", str(satis / "best.pt")])
    for model in ("Caser", "FMLPRec", "GRU4Rec", "HGN", "NARM", "SASRec",
                  "DFN", "DMT", "FeedRec", "FMLPRecM", "GRU4RecM", "SASRecM"):
        run("baseline", data, work / model.lower(),
            ["--model", model, "--rec_len", "5", "--num_blocks", "1"])
    print("mock validation passed")


if __name__ == "__main__":
    main()
