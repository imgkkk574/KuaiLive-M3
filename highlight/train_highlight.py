"""
train_highlight.py — Training & Evaluation for Highlight Prediction
====================================================================

Trains one-step-ahead highlight predictors on highlight_data/ produced by
preprocess_highlight.py. At position t, a model consumes segment t (and the
history allowed by its architecture) and predicts segment t+1.

Evaluation metrics (primary: AntPivot IJCAI-2022 convention;
                   secondary: KuaiHL ICME-2024 convention):
  Primary:
    mAP         — mean Average Precision (binary hl_binary as ground truth)
    F1@50%      — predict top-50% segments per room, F1 vs binary labels
  Secondary:
    Kendall τ   — at thresholds Δ ∈ {0.0, 0.2, 0.4}  (vs. hl_score)
    Spearman ρ  — rank correlation

Usage
-----
  # Causal Transformer (default)
  python train_highlight.py --data_dir highlight_data --model causal

  # Hierarchical causal adaptation
  python train_highlight.py --data_dir highlight_data --model hierarchical

  # Quick sanity check with MLP
  python train_highlight.py --data_dir highlight_data --model mlp --epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_highlight import build_dataloaders
from model_highlight import HighlightLoss, build_model


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_list:   List[np.ndarray],   # per-room predicted scores
    score_list:  List[np.ndarray],   # per-room true hl_scores (continuous)
    binary_list: List[np.ndarray],   # per-room true hl_binary (0/1)
    delta_list:  Tuple[float, ...] = (0.0, 0.2, 0.4),
    top_frac:    float = 0.5,
    hl_pct:      float = 70.0,       # percentile threshold for binary labels in mAP
) -> Dict[str, float]:
    """
    Compute all highlight evaluation metrics averaged over rooms.

    Parameters
    ----------
    pred_list   : list of per-room predicted scores (numpy arrays)
    score_list  : list of per-room true continuous scores
    binary_list : list of per-room true binary labels
    delta_list  : Kendall τ thresholds
    top_frac    : fraction predicted as highlight for F1 computation
    hl_pct      : percentile used to binarise continuous scores for mAP
                  (ignored if binary_list provided directly)

    Returns
    -------
    dict of metric name → float
    """
    mAPs:     List[float] = []
    f1s:      List[float] = []
    spearmans:List[float] = []
    kendalls: Dict[float, List[float]] = {d: [] for d in delta_list}

    for pred, true_cont, true_bin in zip(pred_list, score_list, binary_list):
        K = len(pred)
        if K < 2:
            continue

        # ── mAP ──────────────────────────────────────────────────────────────
        # Use provided binary labels (already at 70th-pct threshold per room)
        if true_bin.sum() > 0 and true_bin.sum() < K:
            mAPs.append(float(average_precision_score(true_bin, pred)))

        # ── F1@50% ───────────────────────────────────────────────────────────
        top_k   = max(1, int(K * top_frac))
        pred_hl = np.zeros(K, dtype=np.int32)
        pred_hl[np.argsort(pred)[-top_k:]] = 1
        tp = int((pred_hl & true_bin.astype(np.int32)).sum())
        p  = tp / (pred_hl.sum() + 1e-8)
        r  = tp / (true_bin.sum() + 1e-8)
        f1 = 2.0 * p * r / (p + r + 1e-8)
        f1s.append(f1)

        # ── Spearman ρ ────────────────────────────────────────────────────────
        rho, _ = spearmanr(pred, true_cont)
        if not np.isnan(rho):
            spearmans.append(float(rho))

        # ── Kendall τ at each Δ ───────────────────────────────────────────────
        for delta in delta_list:
            if delta == 0.0:
                tau, _ = kendalltau(pred, true_cont)
                if not np.isnan(tau):
                    kendalls[delta].append(float(tau))
            else:
                # Only pairs where |true_i - true_j| > delta
                pairs_p, pairs_t = [], []
                for i in range(K):
                    for j in range(i + 1, K):
                        if abs(true_cont[i] - true_cont[j]) > delta:
                            pairs_p.append(pred[i] - pred[j])
                            pairs_t.append(true_cont[i] - true_cont[j])
                if len(pairs_p) >= 2:
                    tau, _ = kendalltau(pairs_p, pairs_t)
                    if not np.isnan(tau):
                        kendalls[delta].append(float(tau))

    result: Dict[str, float] = {
        "mAP":          float(np.mean(mAPs))      if mAPs      else 0.0,
        "F1@50%":       float(np.mean(f1s))        if f1s       else 0.0,
        "Spearman_rho": float(np.mean(spearmans))  if spearmans else 0.0,
    }
    for delta in delta_list:
        vals = kendalls[delta]
        result[f"Kendall_tau_d{delta}"] = float(np.mean(vals)) if vals else 0.0

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader,
    device:     torch.device,
    split_name: str = "val",
) -> Dict[str, float]:
    """Run inference on loader, collect per-room predictions, compute metrics."""
    model.eval()
    pred_list:   List[np.ndarray] = []
    score_list:  List[np.ndarray] = []
    binary_list: List[np.ndarray] = []

    for batch in loader:
        emb    = batch["embeddings"].to(device)    # [B, T, 128]
        mask   = batch["pad_mask"].to(device)      # [B, T] bool
        scores = batch["hl_scores"]                # [B, T] cpu
        binary = batch["hl_binary"]                # [B, T] cpu
        lengths = batch["lengths"]                 # [B]
        stats  = batch.get("stats")                # [B, T, stat_dim] or None
        if stats is not None:
            stats = stats.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(emb, mask, stats)         # [B, T]

        pred_np   = pred.cpu().float().numpy()
        scores_np = scores.numpy()
        binary_np = binary.numpy()

        for b in range(emb.shape[0]):
            k = int(lengths[b])
            pred_list.append(pred_np[b, :k])
            score_list.append(scores_np[b, :k])
            binary_list.append(binary_np[b, :k].astype(np.int32))

    metrics = compute_metrics(pred_list, score_list, binary_list)
    model.train()
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n=== Highlight Prediction Training ===")
    print(f"  model      : {args.model}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  device     : {device}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch_size : {args.batch_size}")
    print("  objective  : segment t -> highlight score of segment t+1")
    print()

    # ── Guard: --use_stats only makes sense on top of the embedding input ──
    if args.use_stats and args.input_mode != "embedding":
        raise SystemExit("--use_stats requires --input_mode embedding "
                         "(stats are concatenated onto the embedding input)")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        seed=args.seed,
        input_mode=args.input_mode,
        use_stats=args.use_stats,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # Input dimension: 128 for segment embeddings, stats_feat_dim for stats mode.
    # Read from stats.json so model's seg_dim matches the actual feature width.
    with open(os.path.join(args.data_dir, "stats.json")) as f:
        _stats = json.load(f)
    input_dim = (_stats.get("stats_feat_dim", 10) if args.input_mode == "stats"
                 else _stats.get("segment_emb_dim", 128))
    stat_dim = _stats.get("stats_feat_dim", 10)
    print(f"  input_mode   : {args.input_mode}  (input_dim={input_dim})")
    if args.use_stats:
        print(f"  use_stats    : True  (stat_dim={stat_dim}, additive branch, gate-init=0)")

    # ── Model ────────────────────────────────────────────────────────────────
    model_kwargs = dict(
        seg_dim=input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len + 10,
        use_stats=args.use_stats,
        stat_dim=stat_dim,
    )
    if args.model == "causal":
        model_kwargs["n_layers"] = args.n_layers
    elif args.model == "hierarchical":
        model_kwargs["n_global"] = args.n_layers
        model_kwargs["window"]   = args.window
    elif args.model in ("gru",):
        model_kwargs = dict(seg_dim=input_dim, hidden=args.d_model,
                            n_layers=args.n_layers, dropout=args.dropout,
                            use_stats=args.use_stats, stat_dim=stat_dim)
    elif args.model == "mlp":
        model_kwargs = dict(seg_dim=input_dim, hidden=args.d_model, dropout=args.dropout,
                            use_stats=args.use_stats, stat_dim=stat_dim)

    model = build_model(args.model, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params  : {n_params:,}")

    # ── Optimiser ────────────────────────────────────────────────────────────
    loss_fn = HighlightLoss(
        lambda_point=args.lambda_point,
        lambda_pair=args.lambda_pair,
        lambda_border=args.lambda_border,
        pair_delta=args.pair_delta,
    )
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    best_ckpt  = os.path.join(args.out_dir, f"best_{args.model}.pt")
    log_path   = os.path.join(args.out_dir, f"log_{args.model}.jsonl")

    best_map     = -1.0
    no_improve   = 0
    history      = []

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>5} {'Loss':>9} {'mAP':>8} {'F1@50%':>8} "
          f"{'Kτ(Δ=0)':>10} {'ρ':>8}  time")
    print("─" * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            emb    = batch["embeddings"].to(device)  # [B, T, 128]
            mask   = batch["pad_mask"].to(device)    # [B, T]
            target = batch["hl_scores"].to(device)   # [B, T]
            stats  = batch.get("stats")              # [B, T, stat_dim] or None
            if stats is not None:
                stats = stats.to(device)

            optimiser.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(emb, mask, stats)       # [B, T]
                loss = loss_fn(pred, target, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_m = evaluate(model, val_loader, device, "val")
            elapsed = time.time() - t0

            print(f"{epoch:5d} {avg_loss:9.4f} {val_m['mAP']:8.4f} "
                  f"{val_m['F1@50%']:8.4f} "
                  f"{val_m['Kendall_tau_d0.0']:10.4f} "
                  f"{val_m['Spearman_rho']:8.4f}  {elapsed:.0f}s")

            log_row = {"epoch": epoch, "loss": avg_loss, **val_m}
            history.append(log_row)
            with open(log_path, "a") as f:
                f.write(json.dumps(log_row) + "\n")

            if val_m["mAP"] > best_map:
                best_map = val_m["mAP"]
                no_improve = 0
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "val_metrics": val_m}, best_ckpt)
            else:
                no_improve += args.eval_every
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {no_improve} epochs).")
                    break
        else:
            elapsed = time.time() - t0
            print(f"{epoch:5d} {avg_loss:9.4f} {'—':>8} {'—':>8} {'—':>10} {'—':>8}  {elapsed:.0f}s")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint from {best_ckpt} ...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Best checkpoint: epoch {ckpt['epoch']}, val mAP = {ckpt['val_metrics']['mAP']:.4f}")

    print("\n── Validation results (best checkpoint) ──")
    val_m = evaluate(model, val_loader, device, "val")
    _print_metrics(val_m)

    print("\n── Test results ──")
    test_m = evaluate(model, test_loader, device, "test")
    _print_metrics(test_m)

    # Save final results
    results = {
        "model":       args.model,
        "prediction_horizon": 1,
        "best_epoch":  ckpt["epoch"],
        "val":         val_m,
        "test":        test_m,
        "args":        vars(args),
    }
    results_path = os.path.join(args.out_dir, f"results_{args.model}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


def _print_metrics(m: Dict[str, float]) -> None:
    print(f"  mAP           : {m['mAP']:.4f}   (primary, AntPivot-style)")
    print(f"  F1@50%        : {m['F1@50%']:.4f}   (primary, AntPivot-style)")
    print(f"  Kendall τ(Δ=0): {m['Kendall_tau_d0.0']:.4f}   (secondary, KuaiHL-style)")
    print(f"  Kendall τ(Δ=.2): {m['Kendall_tau_d0.2']:.4f}")
    print(f"  Kendall τ(Δ=.4): {m['Kendall_tau_d0.4']:.4f}")
    print(f"  Spearman ρ    : {m['Spearman_rho']:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one-step-ahead highlight prediction model on KLM3"
    )

    # Data
    p.add_argument("--data_dir",    required=True,
                   help="Output directory of preprocess_highlight.py")
    p.add_argument("--max_seq_len", type=int, default=200,
                   help="Maximum next-segment predictions per room (0 = no limit)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_mode", default="embedding",
                   choices=["embedding", "stats"],
                   help="'embedding'=segment embeddings (default); "
                        "'stats'=leakage-free causal interaction statistics")
    p.add_argument("--use_stats", action="store_true", default=False,
                   help="Add the 10-dim segment-stat vector as an extra input "
                        "branch on top of the embedding (requires --input_mode "
                        "embedding). gate-init=0 → off is bit-equivalent to "
                        "the original embedding-only model.")

    # Model
    p.add_argument("--model",    default="causal",
                   choices=["causal", "hierarchical", "gru", "mlp"],
                   help="Model architecture (default: causal)")
    p.add_argument("--d_model",  type=int,   default=256)
    p.add_argument("--n_heads",  type=int,   default=4)
    p.add_argument("--n_layers", type=int,   default=2)
    p.add_argument("--window",   type=int,   default=5,
                   help="Local attention window size (hierarchical model only)")
    p.add_argument("--dropout",  type=float, default=0.1)

    # Loss
    p.add_argument("--lambda_point",  type=float, default=1.0)
    p.add_argument("--lambda_pair",   type=float, default=0.5)
    p.add_argument("--lambda_border", type=float, default=0.3)
    p.add_argument("--pair_delta",    type=float, default=0.2,
                   help="Min score difference for pairwise loss")

    # Training
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eval_every",   type=int,   default=1,
                   help="Evaluate on validation every N epochs")
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early stopping patience in epochs (0 = disabled)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="",
                   help="Device string (e.g. 'cuda:0', 'cpu'; auto-detect if empty)")

    # Output
    p.add_argument("--out_dir", default="highlight_ckpt",
                   help="Directory for checkpoints and result files")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
