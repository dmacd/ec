# dreamcoder/domains/rbii/rbii_loop.py
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Set

from dreamcoder.enumeration import enumerateForTasks
from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow, tcharacter
from dreamcoder.utilities import eprint

from .rbii_state import RBIIState, RBIIStateView
from .rbii_types import RBIIEvalState, trbii_state


# allow duplicate programs to re-enter the pool
ALLOW_DUPLICATES = True

@dataclass
class Predictor:
    program: Program
    fn: Callable[[RBIIEvalState], str]
    # (optional) bookkeeping:
    source: str = "enumerated"
    program_id: Optional[int] = None  # absolute id in state.best_programs if stored
    duplicate_candidate: bool = False


@dataclass
class RBIIConfig:
    pool_target_size: int = 3
    validation_window: int = 6
    # Start training/evaluating only at t >= min_time (needs enough lookback)
    min_time: int = 3

    # Enumeration controls
    enum_timeout_s: float = 0.5
    eval_timeout_s: Optional[float] = 0.02
    upper_bound: float = 30.0
    budget_increment: float = 1.5
    max_frontier: int = 10

    verbose: bool = True
    event_log_dir: Optional[str] = None
    event_log_name: str = "rbii_events"
    log_candidate_events: bool = True


class RBIILoop:
    """
    Minimal RBII loop:
      - maintain a pool of predictors
      - evict predictors that fail on the newest observation
      - refill the pool by enumerating predictors that perfectly fit a short
        finite validation window of recent observations.
    """

    def __init__(self, grammar, state: RBIIState, cfg: RBIIConfig = RBIIConfig()):
        self.g = grammar
        self.state = state
        self.cfg = cfg
        self.pool: List[Predictor] = []
        self._seen_program_strs: Set[str] = set()
        self._seen_candidate_program_strs: Set[str] = set()
        self.event_log_path: Optional[str] = None
        self._event_fp = None
        self._init_event_log()

    def _init_event_log(self) -> None:
        if not self.cfg.event_log_dir:
            return
        os.makedirs(self.cfg.event_log_dir, exist_ok=True)
        self.event_log_path = os.path.join(
            self.cfg.event_log_dir, f"{self.cfg.event_log_name}.jsonl"
        )
        self._event_fp = open(self.event_log_path, "w", encoding="utf-8")
        self._log_event(
            "run_start",
            timestep=self.state.time(),
            config=asdict(self.cfg),
        )
        # If state was seeded before loop construction (warmup), emit those
        # observations so visualizations can include the full sequence.
        for t, symbol in enumerate(self.state.obs_history):
            self._log_event(
                "observe",
                timestep=t,
                observed=symbol,
                predicted=None,
                used_program_id=None,
                used_program=None,
                warmup=True,
            )

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
        if self._event_fp is None:
            return
        self._log_event("run_end", timestep=self.state.time())
        self._event_fp.close()
        self._event_fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _predictor_key(self, p: Program) -> str:
        # String form is stable enough for minimal de-dup.
        return str(p)

    def _pool_summary(self) -> str:
        if not self.pool:
            return "[]"
        return "[" + ", ".join(str(pp.program) for pp in self.pool) + "]"

    def _predict_with_view(self, view: RBIIStateView) -> Optional[str]:
        if not self.pool:
            return None
        try:
            return self.pool[0].fn(view)
        except Exception:
            return None

    def predict_next(self) -> Optional[str]:
        """
        Predict next symbol at time t = len(obs_history).
        Returns None if pool empty or errors.
        """
        t = self.state.time()
        view = self.state.view_for_timestep(t)
        return self._predict_with_view(view)

    def observe_and_update(self, symbol: str) -> None:
        """
        Step the RBII system by observing the true symbol at current time t.
        This:
          1) appends obs
          2) evicts pool predictors that fail on this new observation
          3) refills pool if below target size
        """
        t = self.state.time()  # index being observed now
        pre_view = self.state.view_for_timestep(t)

        # TODO: ick - fix this if we use multiple preds for
        #  prediction
        used_predictor = self.pool[0] if self.pool else None
        pred_before = self._predict_with_view(pre_view)
        if used_predictor is not None:
            self._log_event(
                "predict_used",
                timestep=t,
                program_id=used_predictor.program_id,
                program=str(used_predictor.program),
                duplicate_candidate=used_predictor.duplicate_candidate,
                predicted=pred_before,
                observed=symbol,
            )

        # Evict failing predictors (test on the just-observed index t)
        survivors: List[Predictor] = []
        evicted = 0
        for pp in self.pool:
            try:
                yhat = pp.fn(pre_view)
            except Exception:
                yhat = None
            if yhat == symbol:
                survivors.append(pp)
            else:
                evicted += 1
                self._log_event(
                    "evicted",
                    timestep=t,
                    program_id=pp.program_id,
                    program=str(pp.program),
                    duplicate_candidate=pp.duplicate_candidate,
                    predicted=yhat,
                    observed=symbol,
                )
        self.pool = survivors

        # Record the newly observed symbol after all pre-observation checks.
        self.state.observe(symbol)
        self._log_event(
            "observe",
            timestep=t,
            observed=symbol,
            predicted=pred_before,
            used_program_id=(used_predictor.program_id if used_predictor else None),
            used_program=(str(used_predictor.program) if used_predictor else None),
        )

        if self.cfg.verbose:
            eprint(
                f"[t={t:03d}] pred={pred_before!r}  obs={symbol!r}  "
                f"pool_survivors={len(self.pool)} evicted={evicted}"
            )

        # Refill if needed
        self._refill_pool_if_needed(current_index=t)

    def _make_window_task(self, current_index: int) -> Optional[Task]:
        """
        Build a Task for indices in the validation window:
          examples: (state_at_t,) -> obs[t]
        """
        W = self.cfg.validation_window
        start = max(self.cfg.min_time, current_index - W + 1)
        if start > current_index:
            return None

        examples = [
            ((self.state.view_for_timestep(i),), self.state.obs_history[i])
            for i in range(start, current_index + 1)
        ]
        request = arrow(trbii_state, tcharacter)
        return Task(
            name=f"rbii_window_{start}_{current_index}",
            request=request,
            examples=examples,
            cache=False,
        )

    def _refill_pool_if_needed(self, current_index: int) -> None:
        if len(self.pool) >= self.cfg.pool_target_size:
            return

        # Don't enumerate until we have enough history to make useful lookbacks.
        if current_index < self.cfg.min_time:
            return

        task = self._make_window_task(current_index)
        if task is None:
            return

        need = self.cfg.pool_target_size - len(self.pool)

        lm = AllOrNothingLikelihoodModel(timeout=self.cfg.eval_timeout_s)

        frontiers, _, _ = enumerateForTasks(
            self.g,
            [task],
            lm,
            timeout=self.cfg.enum_timeout_s,
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
                eprint(f"  refill: no solutions for {task.name}")
            return

        # Normalize sorts by posterior descending.
        ranked = frontier.normalize().entries

        added = 0
        for entry in ranked:
            if added >= need:
                break
            p = entry.program
            k = self._predictor_key(p)
            duplicate_candidate = k in self._seen_candidate_program_strs
            self._seen_candidate_program_strs.add(k)

            if self.cfg.log_candidate_events:
                self._log_event(
                    "candidate",
                    timestep=current_index,
                    program=str(p),
                    duplicate_candidate=duplicate_candidate,
                    seen_in_pool_history=(k in self._seen_program_strs),
                )

            # TODO: disable this de-duplication to allow multiple programs
            #  with same string form but different semantics
            if k in self._seen_program_strs:
                if ALLOW_DUPLICATES:
                  eprint(f"  refill: re-adding duplicate program {p}")
                else:
                  self._log_event(
                      "candidate_rejected",
                      timestep=current_index,
                      program=str(p),
                      duplicate_candidate=duplicate_candidate,
                      reason="seen_in_pool_history",
                  )
                  continue

            # Compile predictor function (rbii_state -> char)
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

            # Store into state as a "best program" (absolute index for retrieval)
            program_id = self.state.add_best_program(
                p, birth_timestep=current_index
            )

            self.pool.append(
                Predictor(
                    program=p,
                    fn=fn,
                    source="enumerated",
                    program_id=program_id,
                    duplicate_candidate=duplicate_candidate,
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
                eprint(f"  refill: added program_id={program_id}  prog={p}")

        if self.cfg.verbose and added == 0:
            eprint("  refill: solutions existed but all were duplicates/unusable")
