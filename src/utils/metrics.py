"""Lightweight training metric helpers."""

from __future__ import annotations

from typing import Sequence

import torch


class AverageMeter:
    """Tracks the running average of a scalar metric (e.g. loss, accuracy)."""

    __slots__ = ("sum", "count", "avg")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> list[float]:
    """Compute top-k accuracy (in percent) for the specified k values.

    Args:
        logits:  ``(batch, num_classes)`` raw model outputs.
        targets: ``(batch,)`` integer class labels.
        topk:    iterable of k values, e.g. ``(1, 5)``.

    Returns:
        A list of accuracies (percent) aligned with ``topk``.
    """
    if logits.numel() == 0:
        return [0.0 for _ in topk]

    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, batch)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results: list[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0)
        results.append((correct_k * (100.0 / batch_size)).item())
    return results
