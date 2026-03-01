from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SimpleMNISTMLP(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)


def build_fixed_mlp(hidden_size: int = 64, seed: int = 0, device: str = "cpu") -> SimpleMNISTMLP:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    model = SimpleMNISTMLP(hidden_size=hidden_size)
    # Deterministic initialization from the local generator.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p, generator=g) * 0.02)
    return model.to(device)


def clone_model(model: nn.Module, device: Optional[str] = None) -> nn.Module:
    cloned = copy.deepcopy(model)
    if device is not None:
        return cloned.to(device)
    return cloned


def _prep_x(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.detach().float().view(1, -1).to(device)


def predict_distribution(model: nn.Module, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(_prep_x(x, device))
        probs = torch.softmax(logits, dim=-1).view(-1)
    return probs.detach().cpu()


def one_step_sgd(
    model: nn.Module,
    x: torch.Tensor,
    y: int,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    candidate = clone_model(model, device=device)
    candidate.train()

    opt = torch.optim.SGD(candidate.parameters(), lr=float(learning_rate))
    opt.zero_grad(set_to_none=True)

    logits = candidate(_prep_x(x, device))
    target = torch.tensor([int(y)], dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, target)

    if not torch.isfinite(loss):
        raise ValueError("non-finite loss in one_step_sgd")

    loss.backward()
    opt.step()

    return candidate.to("cpu")
