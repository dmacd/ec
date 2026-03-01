# dreamcoder/domains/rbii/rbii_loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Set

from dreamcoder.enumeration import enumerateForTasks
from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow, tcharacter
from dreamcoder.utilities import eprint

from .rbii_state import RBIIState, RBIIStateView
from .rbii_types import RBIIEvalState, trbii_state


@dataclass
class Predictor:
    program: Program
    fn: Callable[[RBIIEvalState], str]
    # (optional) bookkeeping:
    source: str = "enumerated"
    program_id: Optional[int] = None  # absolute id in state.best_programs if stored


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
        pred_before = self._predict_with_view(pre_view)

        # Evict failing predictors (test on the just-observed index t)
        survivors: List[Predictor] = []
        evicted: List[Predictor] = []
        for pp in self.pool:
            try:
                yhat = pp.fn(pre_view)
            except Exception:
                yhat = None
            if yhat == symbol:
                survivors.append(pp)
            else:
                evicted.append(pp)
        self.pool = survivors

        # Record the newly observed symbol after all pre-observation checks.
        self.state.observe(symbol)

        if self.cfg.verbose:
            eprint(
                f"[t={t:03d}] pred={pred_before!r}  obs={symbol!r}  "
                f"pool_survivors={len(self.pool)} evicted={len(evicted)}"
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
            # TODO: disable this de-duplication to allow multiple programs
            #  with same string form but different semantics
            if k in self._seen_program_strs:
                continue

            # Compile predictor function (rbii_state -> char)
            # TODO: why?
            try:
                fn = p.evaluate([])
            except Exception:
                continue

            # Store into state as a "best program" (absolute index for retrieval)
            program_id = self.state.add_best_program(
                p, birth_timestep=current_index
            )

            self.pool.append(Predictor(program=p, fn=fn, source="enumerated", program_id=program_id))
            self._seen_program_strs.add(k)
            added += 1

            if self.cfg.verbose:
                eprint(f"  refill: added program_id={program_id}  prog={p}")

        if self.cfg.verbose and added == 0:
            eprint("  refill: solutions existed but all were duplicates/unusable")
