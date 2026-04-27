"""Training entrypoint.

Usage:
    python -m src.train --cfg configs/default.yaml
    python -m src.train --cfg configs/default.yaml train.epochs=10 optim.lr=0.05
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataloaders
from .models import build_model
from .utils import AverageMeter, accuracy, load_config, set_seed


# --------------------------------------------------------------------------- #
# Optim / scheduler factories                                                 #
# --------------------------------------------------------------------------- #


def _build_optimizer(params, optim_cfg: Mapping) -> torch.optim.Optimizer:
    name = str(optim_cfg["name"]).lower()
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(optim_cfg.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Mapping,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(sched_cfg.get("name", "none")).lower()
    if name in ("none", "", None):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 30)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unknown scheduler {name!r}")


# --------------------------------------------------------------------------- #
# Train / validate one epoch                                                  #
# --------------------------------------------------------------------------- #


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    grad_clip: float,
    log_interval: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    pbar = tqdm(
        loader,
        desc=f"train {epoch:03d}",
        leave=False,
        dynamic_ncols=True,
    )
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        top1 = accuracy(logits.detach(), targets, topk=(1,))[0]
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)

        if log_interval and (step + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{top1_meter.avg:.2f}")

    return {"loss": loss_meter.avg, "acc": top1_meter.avg}


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    desc: str = "val",
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        # Guard top-5 for tasks with <5 classes.
        topk = (1, 5) if logits.size(1) >= 5 else (1,)
        accs = accuracy(logits, targets, topk=topk)
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(accs[0], batch_size)
        if len(accs) > 1:
            top5_meter.update(accs[1], batch_size)

    return {
        "loss": loss_meter.avg,
        "acc": top1_meter.avg,
        "acc_top5": top5_meter.avg,
    }


# --------------------------------------------------------------------------- #
# Checkpoint I/O                                                              #
# --------------------------------------------------------------------------- #


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "config": config,
    }
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional dotted overrides, e.g. train.epochs=5 optim.lr=0.05",
    )
    args = parser.parse_args()

    cfg = load_config(args.cfg, args.overrides)

    set_seed(int(cfg.experiment.seed))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config alongside checkpoints for reproducibility.
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    train_loader, val_loader, num_classes = build_dataloaders(cfg.data)

    model = build_model(cfg.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model.parameters(), cfg.optim)
    scheduler = _build_scheduler(optimizer, cfg.scheduler, int(cfg.train.epochs))

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict] = []
    best_metric = float("-inf")
    best_metric_name = str(cfg.train.get("save_best_metric", "val_acc"))

    print(f"[train] device={device} num_classes={num_classes}")
    print(f"[train] train batches/epoch={len(train_loader)}  val batches={len(val_loader)}")

    epochs = int(cfg.train.epochs)
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_stats = _run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=use_amp,
            grad_clip=float(cfg.train.get("grad_clip", 0.0) or 0.0),
            log_interval=int(cfg.experiment.get("log_interval", 0) or 0),
            epoch=epoch,
        )
        val_stats = _run_eval_epoch(
            model, val_loader, criterion, device, use_amp=use_amp
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_acc_top5": val_stats["acc_top5"],
            "elapsed_s": elapsed,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs} | lr {lr:.4g} | "
            f"train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.2f} | "
            f"val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.2f} | "
            f"{elapsed:.1f}s"
        )

        # Persist history as JSONL for easy plotting later.
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Checkpointing.
        current_metric = record.get(best_metric_name, val_stats["acc"])
        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            config=cfg.to_dict(),
        )
        if current_metric > best_metric:
            best_metric = current_metric
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                config=cfg.to_dict(),
            )
            print(f"  -> new best {best_metric_name}={best_metric:.4f}, saved best.pt")

    print(f"[train] done. best {best_metric_name}={best_metric:.4f}")


if __name__ == "__main__":
    main()
