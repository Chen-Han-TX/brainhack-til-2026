"""Evaluation entrypoint.

Usage:
    python -m src.evaluate --cfg configs/default.yaml
    python -m src.evaluate --cfg configs/default.yaml --ckpt runs/exp/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import build_dataloaders
from .models import build_model
from .utils import AverageMeter, accuracy, load_config, set_seed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> dict[str, float]:
    """Run a full pass over ``loader`` and return aggregate metrics."""
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="eval", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

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
        "samples": loss_meter.count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint .pt (overrides cfg.eval.checkpoint).",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional dotted overrides applied to the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.cfg, args.overrides)
    set_seed(int(cfg.experiment.seed))

    ckpt_path = Path(args.ckpt or cfg.eval.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Use the validation loader for evaluation; eval batch size from config.
    eval_batch = int(cfg.eval.get("batch_size", cfg.data.batch_size))
    _, val_loader, num_classes = build_dataloaders(cfg.data, eval_batch_size=eval_batch)

    model = build_model(cfg.model, num_classes).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[eval] WARNING unexpected keys: {unexpected}")

    criterion = nn.CrossEntropyLoss()
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"

    metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
    print(f"[eval] checkpoint={ckpt_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
