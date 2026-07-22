#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_ckpt.py — Evaluate a saved RecBole / RecBole-CDR checkpoint directly,
without retraining.

Skips training entirely: loads the checkpoint, rebuilds the dataset/dataloaders
from the config saved inside the checkpoint, restores model weights, and runs
the test-set evaluation. Prints valid + test metrics.

Usage:
    python evaluate_ckpt.py --ckpt <path/to/model.pth> [--split test|valid] [--gpu 0]

The script auto-detects whether the checkpoint is a CDR model (uses
recbole_cdr.quick_start.load_data_and_model) or a single-domain base RecBole
model (uses recbole.quick_start.load_data_and_model) based on whether the
config object is a CDRConfig.

Examples:
    python evaluate_ckpt.py --ckpt RecBole-CDR/saved/MGCCDR-Jul-12-2026_18-58-21.pth
    python evaluate_ckpt.py --ckpt RecBole-CDR/saved/CMF-Jul-09-2026_23-17-55.pth --split valid
    python evaluate_ckpt.py --ckpt RecBole/saved/BPR-xxx.pth --gpu 2
"""

import argparse
import os
import sys

# Make both packages importable regardless of cwd: recbole_cdr lives under
# RecBole-CDR/, base recbole under RecBole/. The repo isn't pip-installed, so we
# add both dirs to sys.path. (Run this script from the repo root.)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("RecBole-CDR", "RecBole"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved RecBole/RecBole-CDR checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--split", default="test", choices=["test", "valid"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--gpu", default="0", help="GPU id (default: 0)")
    parser.add_argument("--log", default=None,
                        help="Training log to append result to. If omitted, auto-find by ckpt "
                             "timestamp in log_tune/<model>/ (nearest match).")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    # Resolve to absolute path NOW, before any chdir below — the script cds into the
    # package dir (RecBole-CDR/ or RecBole/) so config['data_path']='dataset/' resolves,
    # which would break a relative ckpt path.
    args.ckpt = os.path.abspath(args.ckpt)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    # Import BOTH packages before any torch.load, so the unpickler can resolve
    # config objects (CDRConfig / Config) embedded in the checkpoint. The torch.load
    # weights_only shim in recbole/__init__.py is applied on this import too.
    import recbole  # noqa: F401  (applies torch.load shim)
    import recbole_cdr  # noqa: F401  (needed to unpickle CDRConfig)

    # Peek at the checkpoint to detect CDR vs single-domain. weights_only=False
    # because checkpoints pickle non-tensor objects (config, CDRConfig).
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config")
    if cfg is None:
        print("[ERROR] checkpoint has no 'config' key; cannot rebuild dataset.", file=sys.stderr)
        sys.exit(1)

    # Detect CDR config: CDRConfig has 'source_domain'/'target_domain' (recbole_cdr).
    is_cdr = cfg.__class__.__name__ == "CDRConfig" or "source_domain" in cfg
    print(f"[info] checkpoint config class: {cfg.__class__.__name__}  -> CDR: {is_cdr}")

    if is_cdr:
        from recbole_cdr.quick_start import load_data_and_model
        from recbole_cdr.utils import get_trainer
        pkg_dir = os.path.join(_REPO_ROOT, "RecBole-CDR")
    else:
        from recbole.quick_start import load_data_and_model
        from recbole.utils import get_trainer
        pkg_dir = os.path.join(_REPO_ROOT, "RecBole")

    # cd into the package dir so config['data_path']='dataset/' (a relative path)
    # resolves correctly — same cwd training used. RecBole looks for the dataset
    # at <cwd>/dataset/<name>, which lives under the package dir.
    os.chdir(pkg_dir)
    print(f"[info] cwd: {os.getcwd()}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.ckpt)
    model = model.to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    eval_data = test_data if args.split == "test" else valid_data

    print(f"\n{'=' * 60}")
    print(f"  Model   : {config['model']}")
    print(f"  Ckpt    : {args.ckpt}")
    print(f"  Split   : {args.split}")
    print(f"  Device  : {config['device']}")
    print(f"{'=' * 60}\n")

    # load_best_model=False because weights are already restored by load_data_and_model.
    result = trainer.evaluate(eval_data, load_best_model=False, show_progress=False)

    print(f"\n{'=' * 60}")
    print(f"  {args.split.upper()} result")
    print(f"{'=' * 60}")
    for k, v in result.items():
        print(f"  {k:<16} : {v}")
    print()

    # Append the test result to the matching run_tune.sh training log, so the
    # eval metrics live next to the training metrics that produced this checkpoint.
    # ckpt name: {Model}-{Mon-DD-YYYY}_{HH-MM-SS}.pth; log: *_{YYYYMMDD_HHMMSS}.log
    # NOTE: the ckpt timestamp is the best-epoch save time, the log timestamp is the
    # run-start time — they differ. --log gives an exact match; otherwise we find the
    # log whose timestamp is closest to (<=) the ckpt time.
    _append_to_training_log(args.ckpt, config['model'], result, args.split, args.log)
    return result


def _ckpt_time_parts(ckpt_name):
    """'MGCCDR-Jul-12-2026_18-58-21.pth' -> (year, month, day, hh, mm, ss)."""
    import re
    m = re.search(
        r'([A-Z][a-z]{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})', ckpt_name)
    if not m:
        return None
    mon_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
               'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    mon_s, day, year, hh, mm, ss = m.groups()
    if mon_s not in mon_map:
        return None
    return int(year), mon_map[mon_s], int(day), int(hh), int(mm), int(ss)


def _ckpt_time_to_log_time(ckpt_name):
    """'MGCCDR-Jul-12-2026_18-58-21.pth' -> '20260712_185324'."""
    parts = _ckpt_time_parts(ckpt_name)
    if parts is None:
        return None
    y, mo, d, hh, mm, ss = parts
    return f"{y:04d}{mo:02d}{d:02d}_{hh:02d}{mm:02d}{ss:02d}"


def _log_filename_to_parts(log_name):
    """'lr1e-3_wd1e-5_layer3_20260712_185324.log' -> (2026,7,12,18,53,24)."""
    import re
    m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', log_name)
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    return int(y), int(mo), int(d), int(hh), int(mm), int(ss)


def _to_epoch_sec(parts):
    import time
    if parts is None:
        return None
    y, mo, d, hh, mm, ss = parts
    try:
        return time.mktime((y, mo, d, hh, mm, ss, 0, 0, -1))
    except (OverflowError, ValueError):
        return None


def _append_to_training_log(ckpt_path, model_name, result, split, explicit_log):
    """Find the training log for this checkpoint and append the eval result line
    in RecBole's log format (so run_tune.sh's extract_test can parse it)."""
    ckpt_name = os.path.basename(ckpt_path)

    # 1. Explicit --log wins.
    if explicit_log:
        log_path = explicit_log if os.path.isabs(explicit_log) \
            else os.path.join(_REPO_ROOT, explicit_log)
        if not os.path.exists(log_path):
            print(f"[warn] --log not found: {log_path}; not appending.")
            return
    else:
        # 2. Auto-find: nearest log by timestamp in log_tune/<model>/.
        log_dir = os.path.join(_REPO_ROOT, "log_tune", model_name)
        if not os.path.isdir(log_dir):
            print(f"[warn] log dir not found: {log_dir}; not appending.")
            return
        ckpt_sec = _to_epoch_sec(_ckpt_time_parts(ckpt_name))
        candidates = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        if not candidates:
            print(f"[warn] no .log in {log_dir}; not appending.")
            return
        if ckpt_sec is None:
            # can't parse ckpt time — pick the most recently modified log
            matches = sorted(candidates,
                             key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
                             reverse=True)
        else:
            # prefer exact timestamp match; else nearest log started <= ckpt time
            exact = _ckpt_time_to_log_time(ckpt_name)
            exact_matches = [f for f in candidates if exact and f.endswith(f"_{exact}.log")]
            if exact_matches:
                matches = exact_matches
            else:
                # nearest by absolute time difference
                def _diff(f):
                    ls = _to_epoch_sec(_log_filename_to_parts(f))
                    return abs(ls - ckpt_sec) if ls else float('inf')
                matches = sorted(candidates, key=_diff)
        log_path = os.path.join(log_dir, matches[0])

    # RecBole format: 'test result: OrderedDict([(...)])' — run_tune.sh greps "test result"
    # then "grep -oP \"'ndcg@10', \\K[0-9.]+\"", so entries must be "'key', value" (comma-space),
    # matching str(OrderedDict([...])) exactly.
    items = ", ".join(f"'{k}', {v}" for k, v in result.items())
    line = f"test result (re-evaluated from ckpt {ckpt_name}): OrderedDict([{items}])\n"
    with open(log_path, "a") as f:
        f.write(line)
    print(f"[info] appended {split} result to: {log_path}")


if __name__ == "__main__":
    main()
