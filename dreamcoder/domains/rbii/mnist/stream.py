from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torchvision.datasets import MNIST


@dataclass(frozen=True)
class StreamExample:
    x: torch.Tensor
    y: int
    context: str


def _indices_for_labels(targets: torch.Tensor, labels: Iterable[int]) -> List[int]:
    label_set = set(int(v) for v in labels)
    return [int(i) for i, y in enumerate(targets.tolist()) if int(y) in label_set]


def build_split_mnist_return_stream(
    data_dir: str = "data/mnist",
    download: bool = True,
    train: bool = True,
    per_context: int = 150,
    schedule: Sequence[str] = ("A", "B", "A", "B"),
    seed: int = 0,
    context_to_labels: Dict[str, Tuple[int, ...]] | None = None,
) -> List[StreamExample]:
    if context_to_labels is None:
        context_to_labels = {
            "A": (0, 1, 2, 3, 4),
            "B": (5, 6, 7, 8, 9),
        }

    try:
        ds = MNIST(root=data_dir, train=train, download=download)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load/download MNIST dataset. "
            "Check network access or pre-populate data/mnist."
        ) from exc

    data = ds.data.float().view(len(ds.data), -1) / 255.0
    targets = ds.targets.long()

    rng = random.Random(int(seed))

    pools: Dict[str, List[int]] = {}
    for context, labels in context_to_labels.items():
        idxs = _indices_for_labels(targets, labels)
        if not idxs:
            raise ValueError(f"no samples found for context {context} labels={labels}")
        rng.shuffle(idxs)
        pools[context] = idxs

    pointers = {context: 0 for context in pools}

    stream: List[StreamExample] = []
    for context in schedule:
        if context not in pools:
            raise ValueError(f"schedule references unknown context: {context}")

        idxs = pools[context]
        for _ in range(int(per_context)):
            ptr = pointers[context]
            if ptr >= len(idxs):
                rng.shuffle(idxs)
                ptr = 0
            idx = idxs[ptr]
            pointers[context] = ptr + 1

            x = data[idx].detach().clone().cpu()
            y = int(targets[idx].item())
            stream.append(StreamExample(x=x, y=y, context=context))

    return stream


def build_synthetic_return_stream(
    per_context: int = 150,
    schedule: Sequence[str] = ("A", "B", "A", "B"),
    seed: int = 0,
    context_to_labels: Dict[str, Tuple[int, ...]] | None = None,
) -> List[StreamExample]:
    """
    Offline smoke-test stream with MNIST-like shape (784 dims, 10 labels).
    """
    if context_to_labels is None:
        context_to_labels = {
            "A": (0, 1, 2, 3, 4),
            "B": (5, 6, 7, 8, 9),
        }

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    label_prototypes = {
        label: torch.randn((784,), generator=g, dtype=torch.float32) * 0.5
        for label in range(10)
    }

    rng = random.Random(int(seed))
    stream: List[StreamExample] = []

    for context in schedule:
        labels = context_to_labels.get(context)
        if labels is None:
            raise ValueError(f"schedule references unknown context: {context}")

        for _ in range(int(per_context)):
            y = int(rng.choice(labels))
            noise = torch.randn((784,), generator=g, dtype=torch.float32) * 0.35
            x = torch.clamp((label_prototypes[y] + noise) / 2.0 + 0.5, 0.0, 1.0)
            stream.append(StreamExample(x=x, y=y, context=context))

    return stream
