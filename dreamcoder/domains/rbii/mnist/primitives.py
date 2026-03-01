from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.type import arrow, tint

from .model import build_fixed_mlp, clone_model, one_step_sgd, predict_distribution
from .types import MNISTEvalState, MNISTPrediction, tmnist_pred, tmnist_state


def _uniform_pred(state: MNISTEvalState) -> MNISTPrediction:
    _ = state
    return MNISTPrediction(kind="dist", dist=torch.full((10,), 0.1, dtype=torch.float32))


def _label_pred(label: int):
    def inner(state: MNISTEvalState) -> MNISTPrediction:
        _ = state
        return MNISTPrediction(kind="label", label=int(label) % 10)

    return inner


def _prev_label_pred(k: int):
    def inner(state: MNISTEvalState) -> MNISTPrediction:
        idx = (state.timestep - 1) - int(k)
        y = state.label_at(idx)
        return MNISTPrediction(kind="label", label=int(y))

    return inner


def _get_historical_program_abs(k: int):
    def inner(state: MNISTEvalState) -> MNISTPrediction:
        fn = state.program_at(int(k))
        return fn(state)

    return inner


def _fresh_mlp_predict(template_model, device: str):
    def inner(state: MNISTEvalState) -> MNISTPrediction:
        model = clone_model(template_model, device=device).to("cpu")
        dist = predict_distribution(model, state.current_x(), device=device)
        return MNISTPrediction(kind="dist", dist=dist, model=model)

    return inner


def _backprop_prev(program_idx: int, learning_rate: float, device: str):
    """
    Short backprop functor:
      load historical program `program_idx`, run one SGD step using (x_{t-1}, y_{t-1}),
      then predict on x_t with the updated model.
    """

    def inner(state: MNISTEvalState) -> MNISTPrediction:
        base_fn = state.program_at(int(program_idx))
        base_pred = base_fn(state)

        if base_pred.model is None:
            return base_pred

        if state.timestep <= 0:
            dist0 = predict_distribution(base_pred.model, state.current_x(), device=device)
            return MNISTPrediction(kind="dist", dist=dist0, model=base_pred.model)

        x_prev = state.x_at(state.timestep - 1)
        y_prev = state.label_at(state.timestep - 1)

        updated = one_step_sgd(
            base_pred.model,
            x_prev,
            y_prev,
            learning_rate=learning_rate,
            device=device,
        )
        dist = predict_distribution(updated, state.current_x(), device=device)
        return MNISTPrediction(kind="dist", dist=dist, model=updated)

    return inner


@dataclass(frozen=True)
class MNISTPrimitiveConfig:
    max_int: int = 16
    hidden_size: int = 64
    model_seed: int = 0
    learning_rate: float = 1e-3
    device: str = "cpu"
    log_variable: float = 0.0


def _primitive(name: str, tp, value) -> Primitive:
    if name in Primitive.GLOBALS:
        return Primitive.GLOBALS[name]
    return Primitive(name, tp, value)


def make_mnist_rbii_grammar(cfg: MNISTPrimitiveConfig = MNISTPrimitiveConfig()) -> Grammar:
    prims: List[Primitive] = []

    # Shared integer constants used for labels, lookbacks, and program ids.
    for i in range(cfg.max_int + 1):
        prims.append(_primitive(str(i), tint, i))

    template_model = build_fixed_mlp(
        hidden_size=cfg.hidden_size,
        seed=cfg.model_seed,
        device=cfg.device,
    ).to("cpu")

    prims.append(
        _primitive(
            "uniform_pred",
            arrow(tmnist_state, tmnist_pred),
            _uniform_pred,
        )
    )
    prims.append(
        _primitive(
            "label_pred",
            arrow(tint, tmnist_state, tmnist_pred),
            _label_pred,
        )
    )
    prims.append(
        _primitive(
            "prev_label_pred",
            arrow(tint, tmnist_state, tmnist_pred),
            _prev_label_pred,
        )
    )
    prims.append(
        _primitive(
            "get_historical_program",
            arrow(tint, tmnist_state, tmnist_pred),
            _get_historical_program_abs,
        )
    )
    prims.append(
        _primitive(
            "fresh_mlp_pred",
            arrow(tmnist_state, tmnist_pred),
            _fresh_mlp_predict(template_model, device=cfg.device),
        )
    )
    prims.append(
        _primitive(
            "backprop_prev",
            arrow(tint, tmnist_state, tmnist_pred),
            lambda k: _backprop_prev(k, learning_rate=cfg.learning_rate, device=cfg.device),
        )
    )

    g = Grammar.uniform(prims)
    g.logVariable = float(cfg.log_variable)
    return g
