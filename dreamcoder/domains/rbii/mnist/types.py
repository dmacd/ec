from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import torch
from torch import nn

from dreamcoder.type import baseType


# Explicit input / output types for DreamCoder tasks in this domain.
tmnist_state = baseType("mnist_state")
tmnist_pred = baseType("mnist_pred")


@dataclass
class MNISTPrediction:
    """
    Prediction value produced by synthesized predictor programs.

    kind:
      - "dist": use `dist` (length-10 probabilities)
      - "label": use `label` (integer class id)

    model is optional metadata carried so short transformers (e.g. one-step
    backprop edits) can continue updating a predictor lineage.
    """

    kind: str
    dist: Optional[torch.Tensor] = None
    label: Optional[int] = None
    model: Optional[nn.Module] = None


@runtime_checkable
class MNISTEvalState(Protocol):
    @property
    def timestep(self) -> int:
        ...

    def current_x(self) -> torch.Tensor:
        ...

    def context_id(self) -> str:
        ...

    def x_at(self, idx: int) -> torch.Tensor:
        ...

    def label_at(self, idx: int) -> int:
        ...

    def program_at(self, k: int) -> Callable[["MNISTEvalState"], MNISTPrediction]:
        ...


def label_to_distribution(
    label: int,
    eps: float = 1e-3,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    label_i = int(label)
    if label_i < 0 or label_i >= num_classes:
        raise ValueError(f"label out of range: {label_i}")

    if eps <= 0.0:
        probs = torch.zeros(num_classes, dtype=torch.float32, device=device)
        probs[label_i] = 1.0
        return probs

    off = eps / float(num_classes - 1)
    probs = torch.full((num_classes,), off, dtype=torch.float32, device=device)
    probs[label_i] = 1.0 - eps
    return probs


def normalize_distribution(
    dist: torch.Tensor,
    min_prob: float = 1e-8,
    num_classes: int = 10,
) -> torch.Tensor:
    flat = dist.detach().float().view(-1)
    if flat.numel() != num_classes:
        raise ValueError(f"expected distribution with {num_classes} classes, got {flat.numel()}")

    # Clamp to avoid -inf log-loss and renormalize.
    clamped = torch.clamp(flat, min=min_prob)
    total = torch.sum(clamped)
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise ValueError("distribution is non-finite or sums to zero")
    return clamped / total


def prediction_to_distribution(
    pred: MNISTPrediction,
    eps: float = 1e-3,
    num_classes: int = 10,
) -> torch.Tensor:
    if pred.kind == "dist":
        if pred.dist is None:
            raise ValueError("prediction kind='dist' missing dist")
        return normalize_distribution(pred.dist, num_classes=num_classes)

    if pred.kind == "label":
        if pred.label is None:
            raise ValueError("prediction kind='label' missing label")
        return label_to_distribution(pred.label, eps=eps, num_classes=num_classes)

    raise ValueError(f"unknown prediction kind: {pred.kind}")


def coerce_prediction(value: object) -> MNISTPrediction:
    if isinstance(value, MNISTPrediction):
        return value

    if isinstance(value, int):
        return MNISTPrediction(kind="label", label=int(value))

    if isinstance(value, torch.Tensor):
        return MNISTPrediction(kind="dist", dist=value)

    if isinstance(value, Sequence) and len(value) == 10:
        return MNISTPrediction(kind="dist", dist=torch.tensor(value, dtype=torch.float32))

    raise ValueError(f"cannot coerce value into MNISTPrediction: {type(value).__name__}")


def argmax_label(dist: torch.Tensor) -> int:
    return int(torch.argmax(dist).item())


def safe_log2_prob(prob: torch.Tensor, min_prob: float = 1e-12) -> float:
    clamped = torch.clamp(prob.detach().float(), min=min_prob)
    return float(torch.log2(clamped).item())
