"""
Preprocess KuaiLive-M3 (KLM3) dataset into RecBole-CDR format.

Outputs:
  RecBole-CDR/dataset/klm3_live/klm3_live.inter   (target domain)
  RecBole-CDR/dataset/klm3_photo/klm3_photo.inter  (source domain)

Usage:
  python preprocess_klm3.py --data_dir /path/to/klm3_data
  python preprocess_klm3.py --data_dir /path/to/klm3_data --sample_ratio 0.01 --min_interactions 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from klm3 import KLM3Config, KLM3Dataset

RECBOLE_CDR_DIR = os.path.join(os.path.dirname(__file__), "RecBole-CDR")
DATASET_DIR = os.path.join(RECBOLE_CDR_DIR, "dataset")


def iterative_filter(df_live, df_photo, min_inter):
    """Remove users/items with fewer than min_inter interactions, iterating until stable."""
    prev_live, prev_photo = -1, -1
    round_num = 0
    while len(df_live) != prev_live or len(df_photo) != prev_photo:
        prev_live, prev_photo = len(df_live), len(df_photo)
        round_num += 1

        live_user_counts = df_live["user_id"].value_counts()
        photo_user_counts = df_photo["user_id"].value_counts()
        df_live = df_live[df_live["user_id"].isin(
            live_user_counts[live_user_counts >= min_inter].index)]
        df_photo = df_photo[df_photo["user_id"].isin(
            photo_user_counts[photo_user_counts >= min_inter].index)]

        live_item_counts = df_live["author_id"].value_counts()
        photo_item_counts = df_photo["photo_id"].value_counts()
        df_live = df_live[df_live["author_id"].isin(
            live_item_counts[live_item_counts >= min_inter].index)]
        df_photo = df_photo[df_photo["photo_id"].isin(
            photo_item_counts[photo_item_counts >= min_inter].index)]

        print(f"  [filter round {round_num}] live: {len(df_live):,} rows, "
              f"photo: {len(df_photo):,} rows")

    return df_live, df_photo


def write_inter(df, out_path, header):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, header=True)
    # Replace auto-generated pandas header with RecBole typed header
    with open(out_path, "r") as f:
        lines = f.readlines()
    lines[0] = header + "\n"
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"  Written: {out_path}  ({len(df):,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess KLM3 → RecBole-CDR format")
    parser.add_argument("--data_dir", required=True,
                        help="Path to KLM3 raw data directory")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Fraction of users to keep (stratified; 1.0 = full dataset)")
    parser.add_argument("--min_interactions", type=int, default=10,
                        help="Min interactions per user/item in each domain")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    print(f"\n=== KLM3 Preprocessing ===")
    print(f"data_dir        : {args.data_dir}")
    print(f"sample_ratio    : {args.sample_ratio}")
    print(f"min_interactions: {args.min_interactions}")
    print(f"output_dir      : {DATASET_DIR}\n")
    print(f"output_dir          : {DATASET_DIR}\n")

    # Load dataset
    config = KLM3Config(load_video=True)
    ds = KLM3Dataset(args.data_dir, config, verbose=True)

    # --- Live domain (target) ---
    print("\n[1/4] Loading live_interaction ...")
    live = ds.live_interaction[["user_id", "author_id", "live_play_start_timestamp"]].copy()
    print(f"  Raw rows: {len(live):,}")

    # Deduplicate (user, author) by keeping the latest play
    live = live.sort_values("live_play_start_timestamp", ascending=False)
    live = live.drop_duplicates(subset=["user_id", "author_id"], keep="first")
    print(f"  After dedup: {len(live):,} rows")

    # Convert timestamp to Unix epoch float
    live["timestamp"] = pd.to_datetime(
        live["live_play_start_timestamp"], errors="coerce"
    ).astype("int64") // int(1e9)
    live = live[live["timestamp"] > 0]  # drop NaT rows
    live = live[["user_id", "author_id", "timestamp"]]

    # --- Photo domain (source) ---
    # Use photo_play: has per-play timestamps enabling temporal split.
    # Positive sample: (leave_timestamp - enter_timestamp) / photo_duration >= 10%.
    print("\n[2/4] Loading photo_play + photo_meta ...")
    play = ds.photo_play[["user_id", "photo_id", "enter_timestamp", "leave_timestamp"]].copy()
    meta = ds.photo_meta[["photo_id", "author_id", "duration"]].copy()
    print(f"  photo_play raw rows: {len(play):,}")

    play = play.merge(meta, on="photo_id", how="inner")
    play = play[play["duration"] > 0]
    # enter/leave_timestamp are parsed as datetime; convert diff to ms integer
    play_duration_ms = (play["leave_timestamp"] - play["enter_timestamp"]).dt.total_seconds() * 1000
    play["watch_ratio"] = play_duration_ms / play["duration"]
    play = play[play["watch_ratio"] >= 0.1]
    print(f"  After >= 10% watch filter: {len(play):,} rows")

    # Use enter_timestamp (datetime) → Unix seconds for temporal split
    play["timestamp"] = play["enter_timestamp"].astype("int64") // int(1e9)
    play = play[play["timestamp"] > 0]
    play = play[["user_id", "photo_id", "timestamp"]]

    # Deduplicate (user, photo) keeping the latest play
    play = play.sort_values("timestamp", ascending=False)
    play = play.drop_duplicates(subset=["user_id", "photo_id"], keep="first")
    print(f"  After dedup: {len(play):,} rows")

    photo = play

    # --- Optional user subsampling ---
    if args.sample_ratio < 1.0:
        print(f"\n[3/4] Subsampling users at ratio {args.sample_ratio} ...")
        rng = np.random.default_rng(args.seed)
        all_users = list(set(live["user_id"].unique()) | set(photo["user_id"].unique()))
        n_sample = max(1, int(len(all_users) * args.sample_ratio))
        sampled_users = set(rng.choice(all_users, size=n_sample, replace=False))
        live = live[live["user_id"].isin(sampled_users)]
        photo = photo[photo["user_id"].isin(sampled_users)]
        print(f"  Kept {len(sampled_users):,} / {len(all_users):,} users")
        print(f"  live: {len(live):,} rows, photo: {len(photo):,} rows")
    else:
        print("\n[3/4] Skipping subsampling (sample_ratio=1.0)")

    # --- Iterative k-core filter ---
    print(f"\n[4/4] Applying {args.min_interactions}-core filter ...")
    live, photo = iterative_filter(live, photo, args.min_interactions)

    # --- Write output ---
    print("\n[Writing] RecBole-CDR .inter files ...")

    # Prefix photo_id with 'p_' to prevent collision with author_id across domains
    # (both are numeric starting from 1; plain author_id vs p_photo_id never collide).
    photo["photo_id"] = "p_" + photo["photo_id"].astype(str)

    live_out = os.path.join(DATASET_DIR, "klm3_live", "klm3_live.inter")
    write_inter(
        live[["user_id", "author_id", "timestamp"]],
        live_out,
        "user_id:token\tauthor_id:token\ttimestamp:float",
    )

    photo_out = os.path.join(DATASET_DIR, "klm3_photo", "klm3_photo.inter")
    write_inter(
        photo[["user_id", "photo_id", "timestamp"]],
        photo_out,
        "user_id:token\tphoto_id:token\ttimestamp:float",
    )

    # --- Write user link file ---
    # RecBole-CDR computes overlap by intersecting already-remapped integer IDs,
    # not raw token strings, so automatic detection fails. An explicit link file
    # that maps each shared user to itself lets the framework find the overlap.
    # _load_data runs before _rename_columns, so both source_user_field and
    # target_user_field equal the raw USER_ID_FIELD value ('user_id').
    # Since source and target share the same field name, two identical columns
    # would conflict; one column named 'user_id' satisfies both assertions.
    overlap_users = set(live["user_id"].unique()) & set(photo["user_id"].unique())
    link_out = os.path.join(DATASET_DIR, "klm3_user.link")
    link_df = pd.DataFrame({"user_id": sorted(overlap_users)})
    link_df.to_csv(link_out, sep="\t", index=False, header=True)
    with open(link_out, "r") as f:
        lines = f.readlines()
    lines[0] = "user_id:token\n"
    with open(link_out, "w") as f:
        f.writelines(lines)
    print(f"  Written: {link_out}  ({len(link_df):,} overlap users)")

    # --- Write streamer-video link file (for MGCCDR) ---
    # S-V graph = author (streamer, live-domain item) × photo (video, photo-domain item).
    # Source: photo_meta.author_id (the author who published each photo).
    # Filter to authors that survived the live-domain filter (so every row maps to a
    # valid target item) and to photos that survived the photo-domain filter. photo_id
    # gets the 'p_' prefix to match klm3_photo.inter; author_id stays bare to match
    # klm3_live.inter (whose item field is author_id).
    print("\n[Writing] MGCCDR streamer-video link file ...")
    sv = meta[["author_id", "photo_id"]].dropna().drop_duplicates()
    valid_authors = set(live["author_id"].unique())
    # photo["photo_id"] is already 'p_'-prefixed at this point; strip to compare with bare meta ids.
    valid_photos_bare = {str(p)[2:] for p in photo["photo_id"].astype(str)}
    sv = sv[sv["author_id"].isin(valid_authors)
            & sv["photo_id"].astype(str).isin(valid_photos_bare)]
    sv["photo_id"] = "p_" + sv["photo_id"].astype(str)
    sv_out = os.path.join(DATASET_DIR, "klm3_sv.link")
    sv.to_csv(sv_out, sep="\t", index=False, header=True)
    with open(sv_out, "r") as f:
        lines = f.readlines()
    lines[0] = "author_id:token\tphoto_id:token\n"
    with open(sv_out, "w") as f:
        f.writelines(lines)
    print(f"  Written: {sv_out}  ({len(sv):,} author-photo pairs, "
          f"{sv['author_id'].nunique():,} authors, {sv['photo_id'].nunique():,} photos)")

    # --- Summary ---
    live_users = live["user_id"].nunique()
    photo_users = photo["user_id"].nunique()
    print(f"\n=== Summary ===")
    print(f"Live  domain : {len(live):>10,} interactions | "
          f"{live_users:>8,} users | {live['author_id'].nunique():>8,} authors")
    print(f"Photo domain : {len(photo):>10,} interactions | "
          f"{photo_users:>8,} users | {photo['photo_id'].nunique():>8,} items")
    print(f"Overlap users: {len(overlap_users):>10,}")
    print(f"S-V pairs    : {len(sv):>10,}")
    print(f"\nDone. Run training with:")
    print(f"  cd RecBole-CDR && python run_recbole_cdr.py "
          f"--model CMF --config_files ../config/klm3.yaml")


if __name__ == "__main__":
    main()
