#!/usr/bin/env python
"""
gen_sv_link.py — Generate the MGCCDR streamer-video link file from existing artifacts.

Reads:
  - the already-preprocessed klm3_live.inter  (to get the surviving author_id set)
  - the already-preprocessed klm3_photo.inter (to get the surviving photo_id set)
  - KLM3 photo_meta.parquet                   (author_id x photo_id ownership)

Writes:
  RecBole-CDR/dataset/klm3_sv.link

This avoids re-running the full k-core filter in preprocess_klm3.py just to add
the S-V file. photo_id is written with the 'p_' prefix to match klm3_photo.inter;
author_id stays bare to match klm3_live.inter.

Usage:
  python gen_sv_link.py --data_dir /path/to/klm3_data
  python gen_sv_link.py --data_dir /path/to/klm3_data --dataset_dir RecBole-CDR/dataset
"""

import argparse
import os

import pandas as pd

DEFAULT_CDR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RecBole-CDR")


def read_inter_column(inter_path, col):
    """Read one column from a RecBole .inter file (tab-separated, first line is the
    RecBole-typed header, e.g. 'user_id:token\\tauthor_id:token\\ttimestamp:float')."""
    df = pd.read_csv(inter_path, sep="\t", dtype=str)
    # strip the ':type' suffix from column names
    df.columns = [c.split(":")[0] for c in df.columns]
    return df[col]


def main():
    parser = argparse.ArgumentParser(description="Generate klm3_sv.link for MGCCDR")
    parser.add_argument("--data_dir", required=True,
                        help="Path to KLM3 raw data (contains photo_meta.parquet)")
    parser.add_argument("--dataset_dir", default=os.path.join(DEFAULT_CDR_DIR, "dataset"),
                        help="RecBole-CDR dataset dir with klm3_live/klm3_photo .inter files")
    args = parser.parse_args()

    live_inter = os.path.join(args.dataset_dir, "klm3_live", "klm3_live.inter")
    photo_inter = os.path.join(args.dataset_dir, "klm3_photo", "klm3_photo.inter")
    for p in (live_inter, photo_inter):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run preprocess_klm3.py first to generate the .inter files.")

    print("[1/3] Reading surviving author_id set from klm3_live.inter ...")
    authors = read_inter_column(live_inter, "author_id")
    valid_authors = set(authors.dropna().astype(str).unique())
    print(f"      {len(valid_authors):,} unique authors")

    print("[2/3] Reading surviving photo_id set from klm3_photo.inter ...")
    # klm3_photo.inter stores photo_id already 'p_'-prefixed
    photos = read_inter_column(photo_inter, "photo_id")
    valid_photos_bare = {str(p)[2:] for p in photos.dropna().unique() if str(p).startswith("p_")}
    print(f"      {len(valid_photos_bare):,} unique photos")

    print("[3/3] Loading photo_meta.parquet and filtering ownership pairs ...")
    meta_path = os.path.join(args.data_dir, "photo_meta.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found.")
    meta = pd.read_parquet(meta_path, columns=["photo_id", "author_id"])
    sv = meta[["author_id", "photo_id"]].dropna().drop_duplicates()
    sv["photo_id"] = sv["photo_id"].astype(str)
    sv["author_id"] = sv["author_id"].astype(str)
    sv = sv[sv["author_id"].isin(valid_authors) & sv["photo_id"].isin(valid_photos_bare)]
    sv["photo_id"] = "p_" + sv["photo_id"]

    sv_out = os.path.join(args.dataset_dir, "klm3_sv.link")
    sv.to_csv(sv_out, sep="\t", index=False, header=True)
    with open(sv_out, "r") as f:
        lines = f.readlines()
    lines[0] = "author_id:token\tphoto_id:token\n"
    with open(sv_out, "w") as f:
        f.writelines(lines)

    print(f"\n=== Done ===")
    print(f"Written: {sv_out}")
    print(f"  {len(sv):,} author-photo pairs")
    print(f"  {sv['author_id'].nunique():,} authors")
    print(f"  {sv['photo_id'].nunique():,} photos")
    print(f"\nNow run: bash run_klm3_cdr.sh {args.data_dir} MGCCDR")


if __name__ == "__main__":
    main()
