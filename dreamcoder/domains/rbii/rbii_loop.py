# dreamcoder/domains/rbii/rbii_loop.py
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Set

from dreamcoder.enumeration import (
    EnumerationDebugHook,
    NOOP_ENUMERATION_DEBUG_HOOK,
    enumerateForTasks,
    solveForTask_bottom,
)
from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow, tcharacter
from dreamcoder.utilities import eprint

from .rbii_state import RBIIState, RBIIStateView
from .rbii_types import RBIIEvalState, trbii_state


# allow duplicate programs to re-enter the pool
ALLOW_DUPLICATES = True


class CollectingLikelihoodModel:
    """
    Wrapper that records every program scored by the underlying likelihood model.
    """

    def __init__(self, inner):
        self.inner = inner
        self.tried_programs: List[str] = []
        self.tried_program_objects: List[Program] = []

    def score(self, program, task):
        self.tried_programs.append(str(program))
        self.tried_program_objects.append(program)
        return self.inner.score(program, task)

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
    enum_solver: str = "python"  # "python" or "bottom"
    enum_cpus: int = 1
    enum_bottom_compile_me: bool = False

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

    def __init__(
        self,
        grammar,
        state: RBIIState,
        cfg: RBIIConfig = RBIIConfig(),
        enumeration_debug_hooks_factory: Optional[
            Callable[[int, Task], EnumerationDebugHook]
        ] = None,
    ):
        self.g = grammar
        self.state = state
        self.cfg = cfg
        self._enumeration_debug_hooks_factory = enumeration_debug_hooks_factory
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

    def _write_failed_enumeration_programs(
        self,
        current_index: int,
        task_name: str,
        request,
        tried_programs: List[str],
        tried_program_objects: List[Program],
        total_number_of_programs: int,
    ) -> Optional[str]:
        if not self.cfg.event_log_dir:
            return None

        os.makedirs(self.cfg.event_log_dir, exist_ok=True)
        path = os.path.join(
            self.cfg.event_log_dir,
            f"enumeration_t{current_index:04d}.json",
        )
        tried_program_records = []
        for program_str, program_obj in zip(tried_programs, tried_program_objects):
            description_length = None
            try:
                description_length = -float(self.g.logLikelihood(request, program_obj))
            except Exception:
                description_length = None
            tried_program_records.append(
                {
                    "program": program_str,
                    "description_length": description_length,
                }
            )

        payload = {
            "timestep": current_index,
            "task_name": task_name,
            "num_programs_scored": len(tried_programs),
            "num_programs_enumerated": int(total_number_of_programs),
            "programs_tried": tried_programs,
            "programs_tried_with_description_length": tried_program_records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        return path

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

        tried_programs: List[str] = []
        tried_program_objects: List[Program] = []
        num_programs_scored = 0

        if self.cfg.enum_solver == "bottom":
            frontiers, _, total_number_of_programs = solveForTask_bottom(
                g=self.g,
                tasks=[task],
                lowerBound=0.0,
                upperBound=float(self.cfg.upper_bound),
                budgetIncrement=float(self.cfg.budget_increment),
                timeout=self.cfg.enum_timeout_s,
                CPUs=max(1, int(self.cfg.enum_cpus)),
                likelihoodModel=None,
                evaluationTimeout=self.cfg.eval_timeout_s,
                maximumFrontiers={task: int(self.cfg.max_frontier)},
                testing=False,
                compile_me=bool(self.cfg.enum_bottom_compile_me),
            )
        else:
            lm = CollectingLikelihoodModel(
                AllOrNothingLikelihoodModel(timeout=self.cfg.eval_timeout_s)
            )
            enumeration_debug_hook = NOOP_ENUMERATION_DEBUG_HOOK
            if self._enumeration_debug_hooks_factory is not None:
                enumeration_debug_hook = self._enumeration_debug_hooks_factory(
                    current_index, task
                )

            frontiers, _, total_number_of_programs = enumerateForTasks(
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
                enumeration_debug_hook=enumeration_debug_hook,
            )
            tried_programs = lm.tried_programs
            tried_program_objects = lm.tried_program_objects
            num_programs_scored = len(lm.tried_programs)

        frontier = frontiers[task]

        if frontier.empty:
            programs_path = self._write_failed_enumeration_programs(
                current_index=current_index,
                task_name=task.name,
                request=task.request,
                tried_programs=tried_programs,
                tried_program_objects=tried_program_objects,
                total_number_of_programs=total_number_of_programs,
            )
            self._log_event(
                "enumeration_no_solution",
                timestep=current_index,
                task_name=task.name,
                num_programs_scored=num_programs_scored,
                num_programs_enumerated=int(total_number_of_programs),
                programs_file=programs_path,
            )
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
