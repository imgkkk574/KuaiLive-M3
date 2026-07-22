from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from saqrec.data import EventDataset, load_bundle, metric_at_ks
from saqrec.baselines import BASELINE_MODELS, build_baseline
from saqrec.multibehavior import MULTIBEHAVIOR_MODELS, build_multibehavior_model
from saqrec.multibehavior_data import MultiBehaviorDataset, load_multibehavior_bundle
from saqrec.models import BaseRec, PropensityModel, SAQRec, SatisfactionModel


CONSOLE_STDERR = sys.stderr


class TrainingStep(torch.nn.Module):
    """Expose every stage's loss through forward for DDP gradient syncing."""
    def __init__(self, model, stage: str, args, propensity=None) -> None:
        super().__init__()
        self.model = model
        self.stage = stage
        self.args = args
        self.propensity = propensity

    def forward(self, batch, epoch: int) -> torch.Tensor:
        if self.stage in {"base", "baseline"}:
            return self.model.loss(batch)
        if self.stage == "propensity":
            logits = self.model(batch["uid"], batch["pos"])
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["observed"])
        if self.stage == "satisfaction":
            with torch.no_grad():
                pro = self.propensity(batch["uid"], batch["pos"])
            return self.model.ips_loss(batch["uid"], batch["pos"], batch["satisfaction"], pro,
                                       self.args.propensity_clamp)
        return self.model.loss(batch, self.args.satisfaction_weight, self.args.correction_after, epoch)


def init_distributed(args) -> tuple[torch.device, bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required unless --cpu is specified")
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, distributed, rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


class Tee:
    """Write every run message to the terminal and a persistent log file."""
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class QuietStream:
    """File-free stdout/stderr sink for non-zero DDP ranks.

    Some managed training environments disallow opening ``/dev/null``.  A
    lightweight in-memory sink keeps auxiliary ranks quiet without relying on
    filesystem permissions.
    """
    def write(self, message: str) -> int:
        return len(message)

    def flush(self) -> None:
        return None


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)


def default_log_path(args, started_at: datetime) -> Path:
    """One readable, collision-resistant log name per experiment invocation."""
    name = args.model or args.stage
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    return Path(args.work_dir) / "logs" / f"{name}_lr{args.lr}_wd{args.weight_decay}_{timestamp}.log"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def seed_worker(worker_id: int) -> None:
    """Give each forked dataset worker an independent, reproducible sampler."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = random.Random(torch.initial_seed() % (2**32))


def save(model, path: Path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_base(path, n_users, n_items, dim, device):
    model = BaseRec(n_users, n_items, dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return model


def load_propensity(path, base, device):
    model = PropensityModel(base).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return model


def load_satisfaction(path, base, device):
    model = SatisfactionModel(base).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return model


@torch.no_grad()
def evaluate(model, dataset, device: torch.device, batch_size: int, ks: list[int]) -> dict:
    model.eval()
    ranks = []
    for sample in DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=device.type == "cuda"):
        batch = move(sample, device)
        candidates = torch.cat([batch["pos"].unsqueeze(1), batch["neg"]], dim=1)
        if isinstance(model, SAQRec):
            scores, _ = model.scores(batch, candidates)
        elif getattr(model, "uses_mixed_history", False):
            scores = model.score(batch["uid"], batch["mixed_his"], batch["mixed_type"], candidates)
        else:
            scores = model.score(batch["uid"], batch["rec_his"], candidates)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite ranking scores encountered during evaluation")
        rank = (scores > scores[:, :1]).sum(dim=1).add(1).detach().cpu().tolist()
        ranks.extend(rank)
    return metric_at_ks(ranks, ks)


def parse_ks(value: str) -> list[int]:
    tokens = value.strip().strip("[]").replace(" ", "").split(",")
    try:
        ks = sorted(set(int(token) for token in tokens if token))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks must look like 1,5,10,20 or [1,5,10,20]") from exc
    if not ks or ks[0] <= 0:
        raise argparse.ArgumentTypeError("--ks must contain positive integers")
    return ks


def train(args):
    device, distributed, rank, world_size = init_distributed(args)
    main_process = is_main_process(rank)
    out = Path(args.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "run_started", "time": args.started_at.isoformat(timespec="seconds"),
                      "stage": args.stage, "model": args.model, "work_dir": str(out),
                      "log_file": args.log_file, "ks": args.ks, "selection_metric": args.selection_metric,
                      "patience": args.patience}))
    set_seed(args.seed)
    print(json.dumps({"event": "distributed_ready", "distributed": distributed, "rank": rank,
                      "world_size": world_size, "device": str(device)}))
    print(json.dumps({"event": "loading_data", "data_dir": str(args.data_dir)}))
    is_multibehavior = args.stage == "baseline" and args.model in MULTIBEHAVIOR_MODELS
    if is_multibehavior:
        bundle = load_multibehavior_bundle(args.data_dir)
        train_set = MultiBehaviorDataset(bundle, "train", args.feedback_len, args.num_negs, args.seed)
        valid_set = MultiBehaviorDataset(bundle, "valid", args.feedback_len, 99, args.seed + 1)
        test_set = MultiBehaviorDataset(bundle, "test", args.feedback_len, 99, args.seed + 2)
        data_event_name = "feedback_tokens"
    else:
        bundle = load_bundle(args.data_dir)
        train_set = EventDataset(bundle, "train", args.rec_len, args.satis_len, args.dissatis_len,
                                 args.num_negs, args.seed, observed_only=args.stage == "satisfaction")
        valid_set = EventDataset(bundle, "valid", args.rec_len, args.satis_len, args.dissatis_len,
                                 99, args.seed + 1)
        test_set = EventDataset(bundle, "test", args.rec_len, args.satis_len, args.dissatis_len,
                                99, args.seed + 2)
        data_event_name = "events"
    print(json.dumps({"event": "data_loaded", data_event_name: int(len(bundle.uid)), "users": bundle.n_users - 1,
                      "authors": bundle.n_items - 1,
                      "split_sizes": {key: int(len(value)) for key, value in bundle.split_indices.items()},
                      "multibehavior": is_multibehavior}))
    generator = torch.Generator().manual_seed(args.seed + rank)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed, drop_last=False) if distributed else None
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
    )
    print(json.dumps({"event": "datasets_ready", "train_examples": len(train_set),
                      "train_examples_per_rank": len(train_sampler) if train_sampler else len(train_set),
                      "valid_examples": len(valid_set), "test_examples": len(test_set), "device": str(device)}))

    base = None
    if args.stage == "base":
        print(json.dumps({"event": "initializing_model", "model": "BaseRec"}))
        model = BaseRec(bundle.n_users, bundle.n_items, args.dim, args.dropout).to(device)
    elif args.stage == "baseline":
        print(json.dumps({"event": "initializing_model", "model": args.model}))
        if is_multibehavior:
            model = build_multibehavior_model(args.model, bundle.n_users, bundle.n_items, args.dim,
                                               args.feedback_len, args.dropout, args.num_heads,
                                               args.num_blocks, args.disentangle_weight,
                                               args.num_experts).to(device)
        else:
            model = build_baseline(args.model, bundle.n_users, bundle.n_items, args.dim, args.rec_len,
                                   args.dropout, args.num_heads, args.num_blocks).to(device)
    else:
        if not args.base_ckpt:
            raise ValueError(f"--base_ckpt is required for stage={args.stage}")
        print(json.dumps({"event": "loading_base_checkpoint", "path": args.base_ckpt}))
        base = load_base(args.base_ckpt, bundle.n_users, bundle.n_items, args.dim, device)
        print(json.dumps({"event": "base_checkpoint_loaded", "path": args.base_ckpt}))
        if args.stage == "propensity":
            print(json.dumps({"event": "initializing_model", "model": "PropensityModel"}))
            model = PropensityModel(base, dropout=args.dropout).to(device)
        elif args.stage == "satisfaction":
            if not args.propensity_ckpt:
                raise ValueError("--propensity_ckpt is required for satisfaction")
            print(json.dumps({"event": "loading_propensity_checkpoint", "path": args.propensity_ckpt}))
            propensity = load_propensity(args.propensity_ckpt, base, device).eval()
            for parameter in propensity.parameters():
                parameter.requires_grad_(False)
            print(json.dumps({"event": "propensity_checkpoint_loaded", "path": args.propensity_ckpt}))
            print(json.dumps({"event": "initializing_model", "model": "SatisfactionModel"}))
            model = SatisfactionModel(base, dropout=args.dropout).to(device)
        elif args.stage == "saqrec":
            if not args.satisfaction_ckpt:
                raise ValueError("--satisfaction_ckpt is required for saqrec")
            print(json.dumps({"event": "loading_satisfaction_checkpoint", "path": args.satisfaction_ckpt}))
            teacher = load_satisfaction(args.satisfaction_ckpt, base, device)
            print(json.dumps({"event": "satisfaction_checkpoint_loaded", "path": args.satisfaction_ckpt}))
            print(json.dumps({"event": "initializing_model", "model": "SAQRec"}))
            model = SAQRec(bundle.n_users, bundle.n_items, teacher, args.dim, args.dropout, args.num_interest).to(device)
            model.initialize_from_base(base)
        else:
            raise ValueError(f"unknown stage {args.stage}")

    train_step = TrainingStep(model, args.stage, args, propensity).to(device) if args.stage == "satisfaction" else \
        TrainingStep(model, args.stage, args).to(device)
    if distributed:
        train_step = DistributedDataParallel(train_step, device_ids=[device.index] if device.type == "cuda" else None,
                                            output_device=device.index if device.type == "cuda" else None)
    optimizer = torch.optim.Adam((p for p in train_step.parameters() if p.requires_grad), lr=args.lr,
                                 weight_decay=args.weight_decay)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({"event": "model_ready", "trainable_parameters": parameter_count,
                      "epochs": args.epochs, "batch_size_per_rank": args.batch_size,
                      "effective_global_batch_size": args.batch_size * world_size, "lr": args.lr,
                      "weight_decay": args.weight_decay}))
    best, best_metric, stale_epochs = None, -float("inf"), 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print(json.dumps({"event": "epoch_started", "epoch": epoch + 1, "total_epochs": args.epochs}))
        train_step.train()
        losses = []
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch",
                        dynamic_ncols=True, leave=False, disable=args.no_tqdm or not main_process, file=CONSOLE_STDERR)
        for step, raw in enumerate(progress, start=1):
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(batch, epoch)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"non-finite training loss at epoch={epoch + 1}, step={step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(train_step.parameters(), 5.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at epoch={epoch + 1}, step={step}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}", refresh=False)
        should_stop = False
        if main_process:
            evaluator = model if args.stage in {"base", "baseline", "saqrec"} else base
            metrics = evaluate(evaluator, valid_set, device, args.eval_batch_size, args.ks)
            record = {"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics}
            print(json.dumps({"split": "valid", **record}))
            selection_score = metrics[args.selection_metric]
            if not math.isfinite(selection_score):
                raise FloatingPointError(
                    f"non-finite validation {args.selection_metric} at epoch={epoch + 1}"
                )
            if selection_score > best_metric:
                best_metric = selection_score
                best = record
                stale_epochs = 0
                save(model, out / "best.pt", {"stage": args.stage, "model": args.model,
                                                "n_users": bundle.n_users, "n_items": bundle.n_items,
                                                "dim": args.dim})
                print(json.dumps({"event": "best_checkpoint_saved", "epoch": epoch + 1,
                                  "selection_metric": args.selection_metric,
                                  "score": best_metric, "path": str(out / "best.pt")}))
            else:
                stale_epochs += 1
                should_stop = args.patience > 0 and stale_epochs >= args.patience
                if should_stop:
                    print(json.dumps({"event": "early_stop", "epoch": epoch + 1,
                                      "selection_metric": args.selection_metric, "patience": args.patience}))
        if distributed:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break
    if main_process:
        (out / "metrics.json").write_text(json.dumps(best, indent=2))
        print(json.dumps({"event": "testing_best_checkpoint", "path": str(out / "best.pt")}))
        checkpoint = torch.load(out / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        evaluator = model if args.stage in {"base", "baseline", "saqrec"} else base
        test_metrics = evaluate(evaluator, test_set, device, args.eval_batch_size, args.ks)
        (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        print(json.dumps({"split": "test", **test_metrics}))
        print(json.dumps({"event": "run_completed", "best_epoch": best["epoch"],
                          "metrics_file": str(out / "metrics.json"),
                          "test_metrics_file": str(out / "test_metrics.json")}))
    else:
        best, test_metrics = None, None
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return best, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one stage of KLM3-SA QRec.")
    parser.add_argument("--stage", choices=["base", "baseline", "propensity", "satisfaction", "saqrec"], required=True)
    parser.add_argument("--model", choices=sorted({**BASELINE_MODELS, **MULTIBEHAVIOR_MODELS}),
                        help="Required with --stage baseline.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--log_file", help="Optional path for the persistent run log (default: <work_dir>/train.log).")
    parser.add_argument("--base_ckpt")
    parser.add_argument("--propensity_ckpt")
    parser.add_argument("--satisfaction_ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after this many non-improving validation epochs; 0 disables it.")
    parser.add_argument("--selection_metric", default="mrr",
                        help="Validation metric used for checkpoints and early stopping.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers; use 8-16 on the KLM3 server.")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_negs", type=int, default=2)
    parser.add_argument("--rec_len", type=int, default=50)
    parser.add_argument("--feedback_len", type=int, default=100,
                        help="Mixed CLICK/SATISFIED/DISSATISFIED history length for FeedRec and *_M models.")
    parser.add_argument("--satis_len", type=int, default=20)
    parser.add_argument("--dissatis_len", type=int, default=10)
    parser.add_argument("--num_interest", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=2, help="SASRec attention heads.")
    parser.add_argument("--num_blocks", type=int, default=2, help="SASRec Transformer blocks.")
    parser.add_argument("--disentangle_weight", type=float, default=0.1,
                        help="FeedRec positive/negative click-interest disentangling coefficient.")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="DMT MMoE expert count; other models ignore this value.")
    parser.add_argument("--propensity_clamp", type=float, default=0.05)
    parser.add_argument("--satisfaction_weight", type=float, default=0.01)
    parser.add_argument("--correction_after", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,5,10,20"),
                        help="Cutoffs, e.g. --ks '1,5,10,20' or --ks '[1,5,10,20]'.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true",
                        help="Disable per-epoch training progress bars.")
    args = parser.parse_args()
    if args.stage == "baseline" and not args.model:
        parser.error("--model is required when --stage baseline")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    available_metrics = set(metric_at_ks([1], args.ks))
    if args.selection_metric not in available_metrics:
        parser.error(f"--selection_metric must be one of {sorted(available_metrics)}")
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.main_process = is_main_process(args.rank)
    args.started_at = datetime.now()
    if not args.log_file:
        args.log_file = str(default_log_path(args, args.started_at))
    if args.main_process:
        configure_logging(Path(args.log_file))
    else:
        sys.stdout = QuietStream()
        sys.stderr = QuietStream()
    print(json.dumps({"event": "main_entered", "time": args.started_at.isoformat(timespec="seconds"),
                      "stage": args.stage, "model": args.model, "log_file": args.log_file,
                      "rank": args.rank, "world_size": args.world_size}))
    print(json.dumps({"event": "arguments_validated", "data_dir": args.data_dir,
                      "work_dir": args.work_dir, "epochs": args.epochs, "batch_size": args.batch_size,
                      "lr": args.lr, "weight_decay": args.weight_decay}))
    train(args)


if __name__ == "__main__":
    main()
