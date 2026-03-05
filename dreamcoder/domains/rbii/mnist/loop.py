from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from dreamcoder.enumeration import enumerateForTasks
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.utilities import eprint

from .likelihood import MNISTLogLossLikelihoodModel
from .metrics import OnlineMNISTMetrics
from .state import MNISTState, MNISTStateView
from .types import (
    MNISTEvalState,
    MNISTPrediction,
    argmax_label,
    coerce_prediction,
    prediction_to_distribution,
    safe_log2_prob,
    tmnist_pred,
    tmnist_state,
)


@dataclass
class Predictor:
    program: Program
    fn: Callable[[MNISTEvalState], MNISTPrediction]
    source: str = "enumerated"
    program_id: Optional[int] = None
    duplicate_candidate: bool = False
    cum_loss_bits: float = 0.0


@dataclass
class MNISTRBIIConfig:
    pool_target_size: int = 4
    validation_window: int = 64
    min_time: int = 32

    enum_timeout_s: float = 1.0
    eval_timeout_s: Optional[float] = 0.05
    upper_bound: float = 40.0
    budget_increment: float = 1.5
    max_frontier: int = 12

    label_smoothing_eps: float = 1e-3
    evict_max_bits: float = 3.5
    weight_temperature: float = 1.0

    verbose: bool = True
    event_log_dir: str = os.path.join("experimentOutputs", "rbii_mnist_events")
    event_log_name: str = "rbii_mnist"
    log_candidate_events: bool = True
    allow_duplicates: bool = True


class MNISTRBIILoop:
    def __init__(self, grammar, state: MNISTState, cfg: MNISTRBIIConfig = MNISTRBIIConfig()):
        self.g = grammar
        self.state = state
        self.cfg = cfg

        self.pool: List[Predictor] = []
        self._seen_program_strs: Set[str] = set()
        self._seen_candidate_program_strs: Set[str] = set()
        self._candidate_events: int = 0
        self._duplicate_candidate_events: int = 0
        self._duplicate_readds: int = 0

        self.metrics = OnlineMNISTMetrics()

        self.event_log_path: Optional[str] = None
        self.metrics_path: Optional[str] = None
        self._event_fp = None
        self._init_event_log()

    def _init_event_log(self) -> None:
        if not self.cfg.event_log_dir:
            return
        os.makedirs(self.cfg.event_log_dir, exist_ok=True)

        self.event_log_path = os.path.join(self.cfg.event_log_dir, f"{self.cfg.event_log_name}.jsonl")
        self.metrics_path = os.path.join(self.cfg.event_log_dir, f"{self.cfg.event_log_name}_metrics.json")
        self._event_fp = open(self.event_log_path, "w", encoding="utf-8")

        self._log_event("run_start", timestep=self.state.time(), config=asdict(self.cfg))

    def _log_event(self, event: str, **payload) -> None:
        if self._event_fp is None:
            return
        row = {
            "event": event,
            "wall_time_s": time.time(),
            **payload,
        }
        self._event_fp.write(json.dumps(row, sort_keys=True) + "\n")
        self._event_fp.flush()

    def close(self) -> None:
        if self._event_fp is not None:
            self._log_event(
                "run_stats",
                timestep=self.state.time(),
                candidate_events=self._candidate_events,
                duplicate_candidate_events=self._duplicate_candidate_events,
                duplicate_readds=self._duplicate_readds,
                unique_program_texts=len(self._seen_program_strs),
                seen_candidate_texts=len(self._seen_candidate_program_strs),
            )
            self._log_event("run_end", timestep=self.state.time())
            self._event_fp.close()
            self._event_fp = None

        if self.metrics_path:
            with open(self.metrics_path, "w", encoding="utf-8") as fh:
                json.dump(self.metrics.summary(), fh, indent=2, sort_keys=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def duplicate_stats(self) -> Dict[str, int]:
        return {
            "candidate_events": self._candidate_events,
            "duplicate_candidate_events": self._duplicate_candidate_events,
            "duplicate_readds": self._duplicate_readds,
            "unique_program_texts": len(self._seen_program_strs),
            "seen_candidate_texts": len(self._seen_candidate_program_strs),
        }

    def _predictor_key(self, p: Program) -> str:
        return str(p)

    def _evaluate_predictor(
        self,
        predictor: Predictor,
        view: MNISTStateView,
    ) -> Optional[torch.Tensor]:
        try:
            pred_value = predictor.fn(view)
            pred = coerce_prediction(pred_value)
            dist = prediction_to_distribution(pred, eps=self.cfg.label_smoothing_eps)
            return dist
        except Exception:
            return None

    def _weights_for(self, predictors: List[Predictor]) -> torch.Tensor:
        if not predictors:
            return torch.empty((0,), dtype=torch.float32)

        losses = torch.tensor([p.cum_loss_bits for p in predictors], dtype=torch.float32)
        tau = max(float(self.cfg.weight_temperature), 1e-6)
        raw = torch.exp(-losses / tau)

        s = torch.sum(raw)
        if not torch.isfinite(s) or float(s) <= 0.0:
            return torch.full((len(predictors),), 1.0 / len(predictors), dtype=torch.float32)
        return raw / s

    def _predict_mixture(
        self,
        view: MNISTStateView,
    ) -> Tuple[torch.Tensor, List[Tuple[Predictor, torch.Tensor]], torch.Tensor]:
        if not self.pool:
            return torch.full((10,), 0.1, dtype=torch.float32), [], torch.empty((0,), dtype=torch.float32)

        valid: List[Tuple[Predictor, torch.Tensor]] = []
        for pp in self.pool:
            dist = self._evaluate_predictor(pp, view)
            if dist is None:
                continue
            valid.append((pp, dist))

        if not valid:
            return torch.full((10,), 0.1, dtype=torch.float32), [], torch.empty((0,), dtype=torch.float32)

        active_predictors = [pp for pp, _ in valid]
        w = self._weights_for(active_predictors)

        mixture = torch.zeros((10,), dtype=torch.float32)
        for wi, (_, dist) in zip(w, valid):
            mixture += wi * dist

        mixture = mixture / torch.sum(mixture)
        return mixture, valid, w

    def observe_and_update(self, x: torch.Tensor, y: int, context: str) -> None:
        t = self.state.time()
        view = self.state.view_for_prediction(current_x=x, current_context=context)

        mixture, valid, w = self._predict_mixture(view)
        pred_label = argmax_label(mixture)
        bits_mix = -safe_log2_prob(mixture[int(y)])

        self.metrics.update(context=context, y_true=int(y), dist=mixture)

        used_program_id = None
        used_program = None
        if valid:
            used_program_id = valid[0][0].program_id
            used_program = str(valid[0][0].program)
            used_pred_label = argmax_label(valid[0][1])
            self._log_event(
                "predict_used",
                timestep=t,
                program_id=used_program_id,
                program=used_program,
                duplicate_candidate=bool(valid[0][0].duplicate_candidate),
                predicted=str(int(used_pred_label)),
                observed=str(int(y)),
            )

        self._log_event(
            "predict",
            timestep=t,
            context=str(context),
            y_true=int(y),
            y_hat=int(pred_label),
            logloss_bits=float(bits_mix),
            pool_size=len(self.pool),
            used_program_id=used_program_id,
            used_program=used_program,
        )

        dist_by_program = {id(pp): dist for pp, dist in valid}

        survivors: List[Predictor] = []
        evicted = 0
        for pp in self.pool:
            dist = dist_by_program.get(id(pp))
            if dist is None:
                evicted += 1
                self._log_event(
                    "evicted",
                    timestep=t,
                    program_id=pp.program_id,
                    program=str(pp.program),
                    duplicate_candidate=bool(pp.duplicate_candidate),
                    reason="invalid_prediction",
                )
                continue

            bits_i = -safe_log2_prob(dist[int(y)])
            pp.cum_loss_bits += float(bits_i)

            if float(bits_i) <= float(self.cfg.evict_max_bits):
                survivors.append(pp)
            else:
                evicted += 1
                self._log_event(
                    "evicted",
                    timestep=t,
                    program_id=pp.program_id,
                    program=str(pp.program),
                    duplicate_candidate=bool(pp.duplicate_candidate),
                    reason="high_logloss",
                    logloss_bits=float(bits_i),
                )

        self.pool = survivors

        self.state.observe(x=x, y=int(y), context=str(context))
        self._log_event(
            "observe",
            timestep=t,
            context=str(context),
            observed=str(int(y)),
            predicted=str(int(pred_label)),
            y_true=int(y),
            y_hat=int(pred_label),
            logloss_bits=float(bits_mix),
            pool_survivors=len(self.pool),
            evicted=evicted,
        )

        if self.cfg.verbose:
            eprint(
                f"[mnist t={t:04d}] ctx={context} y_hat={pred_label} y={int(y)} "
                f"bits={bits_mix:.4f} pool={len(self.pool)}"
            )

        self._refill_pool_if_needed(current_index=t)

    def _make_window_task(self, current_index: int) -> Optional[Task]:
        w = int(self.cfg.validation_window)
        start = max(int(self.cfg.min_time), int(current_index) - w + 1)
        if start > current_index:
            return None

        examples = [
            ((self.state.view_for_history_index(i),), int(self.state.y_history[i]))
            for i in range(start, current_index + 1)
        ]

        request = arrow(tmnist_state, tmnist_pred)
        return Task(
            name=f"rbii_mnist_window_{start}_{current_index}",
            request=request,
            examples=examples,
            cache=False,
        )

    def _refill_pool_if_needed(self, current_index: int) -> None:
        if len(self.pool) >= int(self.cfg.pool_target_size):
            return

        if current_index < int(self.cfg.min_time):
            return

        task = self._make_window_task(current_index)
        if task is None:
            return

        need = int(self.cfg.pool_target_size) - len(self.pool)

        lm = MNISTLogLossLikelihoodModel(
            timeout=self.cfg.eval_timeout_s,
            label_smoothing_eps=self.cfg.label_smoothing_eps,
        )

        frontiers, _, _ = enumerateForTasks(
            self.g,
            [task],
            lm,
            timeout=float(self.cfg.enum_timeout_s),
            lowerBound=0.0,
            upperBound=float(self.cfg.upper_bound),
            budgetIncrement=float(self.cfg.budget_increment),
            maximumFrontiers={task: int(self.cfg.max_frontier)},
            verbose=False,
            testing=False,
            elapsedTime=0.0,
            CPUs=1,
        )

        frontier = frontiers[task]
        if frontier.empty:
            if self.cfg.verbose:
                eprint(f"  [mnist refill] no solutions for {task.name}")
            return

        ranked = frontier.normalize().entries

        added = 0
        for entry in ranked:
            if added >= need:
                break

            p = entry.program
            k = self._predictor_key(p)
            duplicate_candidate = k in self._seen_candidate_program_strs
            self._candidate_events += 1
            if duplicate_candidate:
                self._duplicate_candidate_events += 1
            self._seen_candidate_program_strs.add(k)

            if self.cfg.log_candidate_events:
                self._log_event(
                    "candidate",
                    timestep=current_index,
                    program=str(p),
                    duplicate_candidate=duplicate_candidate,
                    seen_in_pool_history=(k in self._seen_program_strs),
                    log_likelihood=float(entry.logLikelihood),
                    log_prior=float(entry.logPrior),
                )

            if k in self._seen_program_strs:
                if self.cfg.allow_duplicates:
                    self._duplicate_readds += 1
                    self._log_event(
                        "candidate_readded",
                        timestep=current_index,
                        program=str(p),
                        duplicate_candidate=duplicate_candidate,
                        reason="seen_in_pool_history",
                    )
                    if self.cfg.verbose:
                        eprint(f"  [mnist refill] re-adding duplicate program {p}")
                else:
                    self._log_event(
                        "candidate_rejected",
                        timestep=current_index,
                        program=str(p),
                        duplicate_candidate=duplicate_candidate,
                        reason="seen_in_pool_history",
                    )
                    continue

            try:
                fn = p.evaluate([])
            except Exception:
                self._log_event(
                    "candidate_rejected",
                    timestep=current_index,
                    program=str(p),
                    duplicate_candidate=duplicate_candidate,
                    reason="compile_failed",
                )
                continue

            program_id = self.state.add_best_program(p, birth_timestep=current_index)
            self.pool.append(
                Predictor(
                    program=p,
                    fn=fn,
                    source="enumerated",
                    program_id=program_id,
                    duplicate_candidate=duplicate_candidate,
                    cum_loss_bits=0.0,
                )
            )
            self._seen_program_strs.add(k)
            added += 1

            self._log_event(
                "enter",
                timestep=current_index,
                program_id=program_id,
                program=str(p),
                duplicate_candidate=duplicate_candidate,
            )

            if self.cfg.verbose:
                eprint(f"  [mnist refill] added id={program_id} program={p}")

        if self.cfg.verbose and added == 0:
            eprint("  [mnist refill] solutions existed but all were duplicates/unusable")
