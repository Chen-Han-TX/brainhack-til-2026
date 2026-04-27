"""Single-image inference entrypoint.

Usage:
    python -m src.inference --ckpt runs/resnet18_cifar10/best.pt --image path/to/img.jpg

The checkpoint must have been produced by ``src.train`` — the resolved training
config is read from the checkpoint's ``config`` field, so the eval transform
and model architecture are reconstructed exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from .models.registry import build_model


def _build_eval_transform(data_cfg: dict) -> T.Compose:
    """Mirror src.data.datasets._build_eval_transform without importing it
    (datasets.py pulls torchvision.datasets, which we don't need at inference)."""
    image_size = int(data_cfg["image_size"])
    resize_size = max(image_size, int(image_size * 1.15))
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image inference.")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pt file.")
    parser.add_argument("--image", required=True, type=str, help="Path to input image.")
    parser.add_argument("--topk", type=int, default=5, help="How many top predictions to print.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cpu")

    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload["config"]

    transform = _build_eval_transform(cfg["data"])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    num_classes = int(cfg["data"]["num_classes"])
    model = build_model(cfg["model"], num_classes).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        topk = min(int(args.topk), probs.numel())
        scores, indices = probs.topk(topk)

    print(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "image": str(image_path),
                "top_k": [
                    {"class_index": int(i), "probability": float(s)}
                    for i, s in zip(indices.tolist(), scores.tolist())
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
