#!/usr/bin/env python3
"""
parse_highlight_results.py — Parse highlight sweep results into a summary.

Reads every results_*.json under a checkpoint directory (recursively) and
reports, per model, all configs sorted by val mAP, with the best config's
val + test metrics highlighted.

Each results_*.json has the shape (written by train_highlight.py):
    {
      "model": "causal",
      "best_epoch": 37,
      "val":  {mAP, F1@50%, Kendall_tau_d0.0, ...},
      "test": {mAP, F1@50%, Kendall_tau_d0.0, ...},
      "args": {lr, weight_decay, d_model, ...}
    }

Usage
-----
    python parse_highlight_results.py highlight_ckpt
    python parse_highlight_results.py highlight_ckpt --csv summary.csv
    python parse_highlight_results.py highlight_ckpt --best-only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List


# Metric columns in display order. Names match the keys in results_*.json.
METRIC_COLS = [
    "mAP",
    "F1@50%",
    "Kendall_tau_d0.0",
    "Kendall_tau_d0.2",
    "Kendall_tau_d0.4",
    "Spearman_rho",
]
MODEL_ORDER = ["mlp", "gru", "causal", "hierarchical"]


def load_runs(ckpt_dir: str) -> List[Dict]:
    """Load every results_*.json under ckpt_dir (recursive)."""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "**", "results_*.json"),
                             recursive=True))
    if not paths:
        # Fallback: non-recursive (old single-level layout)
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "results_*.json")))

    runs = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  [warn] could not parse {p}: {e}", file=sys.stderr)
            continue
        a = d.get("args", {})
        runs.append({
            "model":     d.get("model"),
            "lr":        a.get("lr"),
            "wd":        a.get("weight_decay"),
            "use_stats": bool(a.get("use_stats", False)),
            "epoch":     d.get("best_epoch"),
            "val":       d.get("val", {}),
            "test":      d.get("test", {}),
            "path":      p,
        })
    return runs


def _g(d: Dict, k: str) -> float:
    """Safe float get; missing → nan."""
    v = d.get(k)
    return float(v) if v is not None else float("nan")


def _es_tag(r: Dict) -> str:
    """Tag a run's input variant: '' for embedding-only, '(es)' for emb+stats."""
    return " (es)" if r.get("use_stats") else ""


def fmt_row(rank, r, best_tag="") -> str:
    v, t = r["val"], r["test"]
    return (f"  {rank:<5}{str(r['lr']):<9}{str(r['wd']):<9}"
            f"{_g(v,'mAP'):<8.4f}  {_g(t,'mAP'):<8.4f}  "
            f"{_g(t,'F1@50%'):<8.4f}{_g(t,'Kendall_tau_d0.0'):<9.4f}"
            f"{_g(t,'Kendall_tau_d0.2'):<9.4f}{_g(t,'Kendall_tau_d0.4'):<9.4f}"
            f"{_g(t,'Spearman_rho'):<8.4f}{_es_tag(r)}{best_tag}")


def print_summary(runs: List[Dict], best_only: bool) -> None:
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        if r["model"]:
            by_model[r["model"]].append(r)

    models = [m for m in MODEL_ORDER if m in by_model]
    models += sorted(set(by_model) - set(MODEL_ORDER))   # any unexpected models last

    print(f"\n共 {len(runs)} 个成功 run, {len(by_model)} 个模型\n")

    for m in models:
        rs = sorted(by_model[m], key=lambda r: _g(r["val"], "mAP"), reverse=True)
        print("=" * 104)
        print(f"MODEL: {m}  ({len(rs)} runs)")
        print("=" * 104)
        if not rs:
            print("  (无结果)"); continue

        if not best_only:
            print(f"  {'rank':<5}{'lr':<9}{'wd':<9}{'val_mAP':<9}{'test_mAP':<10}"
                  f"{'F1@50':<9}{'Kτ(0)':<9}{'Kτ(.2)':<9}{'Kτ(.4)':<9}{'ρ':<8}")
            print("  " + "-" * 100)
            for i, r in enumerate(rs):
                print(fmt_row(i, r, best_tag="  ★" if i == 0 else ""))

        best = rs[0]
        print(f"\n  ★ BEST (val mAP): lr={best['lr']}  wd={best['wd']}  "
              f"(best_epoch={best['epoch']}){_es_tag(best)}")
        print(f"     val : " + "  ".join(f"{c}={_g(best['val'],c):.4f}" for c in METRIC_COLS))
        print(f"     test: " + "  ".join(f"{c}={_g(best['test'],c):.4f}" for c in METRIC_COLS))
        # val→test gap, useful for overfitting diagnosis
        print(f"     val→test mAP gap: {_g(best['val'],'mAP') - _g(best['test'],'mAP'):+.4f}")
        print()

    # Cross-model leaderboard by best test mAP
    print("=" * 104)
    print("LEADERBOARD (best config per model, by test mAP)")
    print("=" * 104)
    print(f"  {'model':<14}{'lr':<9}{'wd':<9}{'val_mAP':<9}{'test_mAP':<10}"
          f"{'F1@50':<9}{'Kτ(0)':<9}{'ρ':<8}")
    print("  " + "-" * 84)
    board = []
    for m in models:
        rs = sorted(by_model[m], key=lambda r: _g(r["val"], "mAP"), reverse=True)
        if rs:
            board.append(rs[0])
    board.sort(key=lambda r: _g(r["test"], "mAP"), reverse=True)
    for r in board:
        v, t = r["val"], r["test"]
        print(f"  {r['model']:<14}{str(r['lr']):<9}{str(r['wd']):<9}"
              f"{_g(v,'mAP'):<8.4f}  {_g(t,'mAP'):<8.4f}  "
              f"{_g(t,'F1@50%'):<8.4f}{_g(t,'Kendall_tau_d0.0'):<9.4f}{_g(t,'Spearman_rho'):<8.4f}"
              f"{_es_tag(r)}")
    print()


def write_csv(runs: List[Dict], csv_path: str) -> None:
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        if r["model"]:
            by_model[r["model"]].append(r)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "lr", "weight_decay", "use_stats", "best_epoch"]
        header += [f"val_{c}" for c in METRIC_COLS]
        header += [f"test_{c}" for c in METRIC_COLS]
        w.writerow(header)
        for m in MODEL_ORDER + sorted(set(by_model) - set(MODEL_ORDER)):
            for r in sorted(by_model.get(m, []), key=lambda r: _g(r["val"], "mAP"),
                            reverse=True):
                row = [m, r["lr"], r["wd"], int(r.get("use_stats", False)), r["epoch"]]
                row += [_g(r["val"], c) for c in METRIC_COLS]
                row += [_g(r["test"], c) for c in METRIC_COLS]
                w.writerow(row)
    print(f"\nCSV written to {csv_path}  ({sum(len(v) for v in by_model.values())} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description="Parse highlight sweep results.")
    p.add_argument("ckpt_dir", help="Directory holding per-run subdirs with results_*.json")
    p.add_argument("--csv", default="", help="Optional: write full grid to this CSV path")
    p.add_argument("--best-only", action="store_true",
                   help="Only print the best config per model (skip full grid)")
    args = p.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        print(f"[error] not a directory: {args.ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(args.ckpt_dir)
    if not runs:
        print(f"[error] no results_*.json found under {args.ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    print_summary(runs, best_only=args.best_only)
    if args.csv:
        write_csv(runs, args.csv)


if __name__ == "__main__":
    main()
