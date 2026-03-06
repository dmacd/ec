from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, Sequence, Set, Tuple

from dreamcoder.enumeration import (
    EnumerationDebugHook,
    NOOP_ENUMERATION_DEBUG_HOOK,
    solveForTask_bottom,
)
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow, tcharacter
from dreamcoder.utilities import eprint

from .rbii_loss import CategoricalLogLossModel, RBIIWindowLossModel
from .rbii_state import RBIIState
from .rbii_types import RBIIEvalState, trbii_state


@dataclass
class PoolPredictor:
    program: Program
    fn: Callable[[RBIIEvalState], Any]
    weight: float = 1.0
    source: str = "enumerated"
    program_id: Optional[int] = None
    active_id: Optional[int] = None
    duplicate_candidate: bool = False


@dataclass
class CandidateProposal:
    program: Program
    witness_bits: float
    fn: Optional[Callable[[RBIIEvalState], Any]] = None
    source: str = "enumerated"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightedAdmission:
    proposal: CandidateProposal
    initial_weight: float
    reason: str = "accepted"


@dataclass
class FreezeDecision:
    should_freeze: bool
    predictor: Optional[PoolPredictor] = None
    reason: str = "policy"


@dataclass
class PredictionSnapshot:
    symbol: Optional[str]
    incumbent: Optional[PoolPredictor]
    distribution: Optional[Dict[str, float]]


@dataclass
class ExploreContext:
    timestep: int
    grammar: Any
    task: Task
    state: RBIIState
    pool: Sequence[PoolPredictor]
    cfg: "RBIIConfigV2"
    surprise_score: float = 0.0


@dataclass
class CandidateContext:
    timestep: int
    grammar: Any
    task: Task
    state: RBIIState
    pool: Sequence[PoolPredictor]
    cfg: "RBIIConfigV2"


@dataclass
class FreezeContext:
    timestep: int
    state: RBIIState
    pool: Sequence[PoolPredictor]
    incumbent: Optional[PoolPredictor]
    incumbent_run_length: int
    cfg: "RBIIConfigV2"


class EnumerationController(Protocol):
    def propose_batch(self, ctx: ExploreContext) -> List[CandidateProposal]:
        """
        Produce candidate predictor proposals for the current timestep.
        """
        ...


class EnumerationSchedule(Protocol):
    def batch_size(self, ctx: ExploreContext) -> int:
        """
        Return how many proposals should be surfaced at this timestep.
        """
        ...


class CandidateWeightPolicy(Protocol):
    def admit_and_weight(
        self,
        ctx: CandidateContext,
        candidates: Sequence[CandidateProposal],
    ) -> List[WeightedAdmission]:
        """
        Score/weight candidates on the current window.
        """
        ...


class FreezePolicy(Protocol):
    def should_freeze(self, ctx: FreezeContext) -> FreezeDecision:
        """
        Decide whether the current incumbent should be frozen.
        """
        ...


@dataclass
class RBIIConfigV2:
    pool_target_size: int = 3
    validation_window: int = 6
    min_time: int = 3

    enum_timeout_s: float = 0.5
    eval_timeout_s: Optional[float] = 0.02
    upper_bound: float = 30.0
    budget_increment: float = 1.5
    max_frontier: int = 10
    enum_cpus: int = 1
    enum_bottom_compile_me: bool = False

    exploration_min_batch: int = 1
    exploration_surplus_batch: int = 0

    # Required explicit alphabet for categorical log-loss accounting.
    alphabet: Tuple[str, ...] = ()
    deterministic_smoothing_eps: float = 1e-3
    min_probability: float = 1e-12

    # Candidate passes compression-cost gate when:
    #   compression_gain + compression_gain_slack_bits > witness_bits
    # Larger values are more permissive.
    compression_gain_slack_bits: float = 0.0

    verbose: bool = True
    event_log_dir: Optional[str] = None
    event_log_name: str = "rbii_events"

    alphabet_set: FrozenSet[str] = field(init=False, repr=False)
    baseline_bits_per_symbol: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.alphabet, tuple):
            raise TypeError("RBIIConfigV2.alphabet must be a tuple[str, ...].")
        if not self.alphabet:
            raise ValueError("RBIIConfigV2.alphabet must be explicitly provided.")

        seen = set()
        for symbol in self.alphabet:
            if not isinstance(symbol, str) or len(symbol) != 1:
                raise ValueError(
                    "RBIIConfigV2.alphabet entries must be single-character strings."
                )
            if symbol in seen:
                raise ValueError("RBIIConfigV2.alphabet must not contain duplicates.")
            seen.add(symbol)

        self.alphabet_set = frozenset(self.alphabet)
        self.baseline_bits_per_symbol = (
            0.0 if len(self.alphabet) <= 1 else math.log2(float(len(self.alphabet)))
        )


class BottomSolverEnumerationController:
    """
    Bottom-solver-backed proposal generator.

    This adapter is intentionally narrow and policy-agnostic:
    it only proposes programs and carries basic complexity witnesses.
    Selection and weight assignment stay in CandidateWeightPolicy.
    """

    def __init__(
        self,
        schedule: Optional[EnumerationSchedule] = None,
        enumeration_debug_hooks_factory: Optional[
            Callable[[int, Task], EnumerationDebugHook]
        ] = None,
    ):
        self._schedule = schedule or ConstantEnumerationSchedule(batch_size_value=16)
        self._enumeration_debug_hooks_factory = enumeration_debug_hooks_factory

    def propose_batch(self, ctx: ExploreContext) -> List[CandidateProposal]:
        hook = NOOP_ENUMERATION_DEBUG_HOOK
        if self._enumeration_debug_hooks_factory is not None:
            hook = self._enumeration_debug_hooks_factory(ctx.timestep, ctx.task)

        frontiers, _, _total = solveForTask_bottom(
            g=ctx.grammar,
            tasks=[ctx.task],
            lowerBound=0.0,
            upperBound=float(ctx.cfg.upper_bound),
            budgetIncrement=float(ctx.cfg.budget_increment),
            timeout=ctx.cfg.enum_timeout_s,
            CPUs=max(1, int(ctx.cfg.enum_cpus)),
            likelihoodModel=None,
            evaluationTimeout=ctx.cfg.eval_timeout_s,
            maximumFrontiers={ctx.task: int(ctx.cfg.max_frontier)},
            testing=False,
            compile_me=bool(ctx.cfg.enum_bottom_compile_me),
            enumeration_debug_hook=hook,
        )

        frontier = frontiers[ctx.task]
        if frontier.empty:
            return []

        ranked = frontier.normalize().entries
        out: List[CandidateProposal] = []
        for entry in ranked:
            if entry.logPrior is None:
                raise ValueError("Bottom-solver candidate missing logPrior; witness_bits required.")
            witness_bits = float(-entry.logPrior)
            out.append(
                CandidateProposal(
                    program=entry.program,
                    source="bottom_enumerator",
                    witness_bits=witness_bits,
                    metadata={
                        "log_likelihood": float(entry.logLikelihood),
                        "log_prior": float(entry.logPrior),
                    },
                )
            )

        batch_size = max(0, int(self._schedule.batch_size(ctx)))
        if batch_size == 0:
            return []
        return out[:batch_size]


@dataclass
class ConstantEnumerationSchedule:
    batch_size_value: int = 16

    def batch_size(self, _ctx: ExploreContext) -> int:
        return max(0, int(self.batch_size_value))


@dataclass
class SurpriseAdaptiveEnumerationSchedule:
    min_batch: int = 1
    surprise_batch: int = 8
    surprise_threshold: float = 0.5

    def batch_size(self, ctx: ExploreContext) -> int:
        if float(ctx.surprise_score) >= float(self.surprise_threshold):
            return max(0, int(self.surprise_batch))
        return max(0, int(self.min_batch))


class MDLDetectabilityPolicy:
    """
    Minimal MDL-style policy scaffold.

    Current version keeps the policy simple:
    - estimates candidate loss on the current validation window
    - computes MDL score = loss + witness_bits
    - filters out candidates whose compression gain does not exceed bit-cost
    - assigns a comparable insertion weight to every scored candidate
      via Z * 2^{-MDL score}
    """

    def __init__(self, loss_model: RBIIWindowLossModel):
        self.loss_model = loss_model

    def admit_and_weight(
        self,
        ctx: CandidateContext,
        candidates: Sequence[CandidateProposal],
    ) -> List[WeightedAdmission]:
        if not candidates:
            return []

        current_mass = sum(max(pp.weight, 0.0) for pp in ctx.pool)
        if current_mass <= 0.0:
            current_mass = 1.0

        start = max(int(ctx.cfg.min_time), int(ctx.timestep) - int(ctx.cfg.validation_window) + 1)
        if start > int(ctx.timestep):
            return []
        baseline_bits = self.loss_model.baseline_bits(
            state=ctx.state,
            cfg=ctx.cfg,
            start_timestep=start,
            end_timestep=int(ctx.timestep),
        )

        ranked: List[tuple[float, CandidateProposal]] = []
        for cand in candidates:
            loss = self._window_loss_bits(ctx, cand)
            if loss is None:
                continue

            k_hat = float(cand.witness_bits)
            if k_hat < 0.0:
                raise ValueError(f"witness_bits must be non-negative; got {k_hat}.")
            compression_gain = baseline_bits - float(loss)
            effective_gain = compression_gain + float(ctx.cfg.compression_gain_slack_bits)
            if effective_gain <= k_hat:
                continue
            score = float(loss + k_hat)
            ranked.append((score, cand))

        ranked.sort(key=lambda x: x[0])

        admissions: List[WeightedAdmission] = []
        for score, cand in ranked:
            initial_weight = math.pow(2.0, -score) * current_mass
            admissions.append(
                WeightedAdmission(
                    proposal=cand,
                    initial_weight=initial_weight,
                    reason=f"mdl_score={score:.4f}",
                )
            )

        return admissions

    def _window_loss_bits(
        self,
        ctx: CandidateContext,
        cand: CandidateProposal,
    ) -> Optional[float]:
        fn = cand.fn
        if fn is None:
            try:
                fn = cand.program.evaluate([])
            except Exception:
                return None

        start = max(int(ctx.cfg.min_time), int(ctx.timestep) - int(ctx.cfg.validation_window) + 1)
        if start > int(ctx.timestep):
            return None

        loss = 0.0
        for i in range(start, int(ctx.timestep) + 1):
            view = ctx.state.view_for_timestep(i)
            try:
                yhat = fn(view)
            except Exception:
                return None

            y = ctx.state.obs_history[i]
            bits = self.loss_model.loss_bits(
                prediction=yhat,
                observed=y,
                state=ctx.state,
                cfg=ctx.cfg,
                timestep=i,
            )
            if bits is None:
                return None
            loss += float(bits)
        return loss


class AlwaysFreezeIncumbentPolicy:
    """
    Analysis-mode freeze policy equivalent to "freeze incumbent each step".
    """

    def should_freeze(self, ctx: FreezeContext) -> FreezeDecision:
        if ctx.incumbent is None:
            return FreezeDecision(False, None, reason="empty_pool")
        return FreezeDecision(True, ctx.incumbent, reason="always_freeze_incumbent")


class RBIILoopV2:
    """
    Policy-factored RBII scaffold.

    This class wires together three replaceable policy surfaces:
    - EnumerationController: scheduling and proposal generation
    - CandidateWeightPolicy: candidate admission + insertion weight
    - FreezePolicy: freeze decision for incumbents
    """

    def __init__(
        self,
        grammar: Any,
        state: RBIIState,
        cfg: Optional[RBIIConfigV2] = None,
        enumerator: Optional[EnumerationController] = None,
        candidate_policy: Optional[CandidateWeightPolicy] = None,
        freeze_policy: Optional[FreezePolicy] = None,
        loss_model: Optional[RBIIWindowLossModel] = None,
    ):
        self.g = grammar
        self.state = state
        if cfg is None:
            raise ValueError("RBIILoopV2 requires an explicit RBIIConfigV2 with alphabet.")
        self.cfg = cfg
        self.loss_model = loss_model or CategoricalLogLossModel()

        default_schedule = SurpriseAdaptiveEnumerationSchedule(
            min_batch=max(0, int(self.cfg.exploration_min_batch)),
            surprise_batch=max(
                int(self.cfg.exploration_min_batch),
                int(self.cfg.exploration_min_batch) + int(self.cfg.exploration_surplus_batch),
            ),
            surprise_threshold=0.5,
        )
        self.enumerator = enumerator or BottomSolverEnumerationController(
            schedule=default_schedule
        )
        self.candidate_policy = candidate_policy or MDLDetectabilityPolicy(
            loss_model=self.loss_model
        )
        self.freeze_policy = freeze_policy or AlwaysFreezeIncumbentPolicy()

        self.pool: List[PoolPredictor] = []
        self._frozen_program_id_by_key: Dict[str, int] = {}
        for i, p in enumerate(self.state.best_programs):
            k = str(p)
            if k not in self._frozen_program_id_by_key:
                self._frozen_program_id_by_key[k] = i
        self._incumbent_key: Optional[str] = None
        self._incumbent_run_length: int = 0
        self._last_step_surprise: float = 0.0
        self._next_active_id: int = 1
        self._seen_active_program_keys: Set[str] = set()
        self.event_log_path: Optional[str] = None
        self._event_fp = None
        self._init_event_log()

    def predict_next(self) -> Optional[str]:
        t = self.state.time()
        view = self.state.view_for_timestep(t)
        return self._prediction_snapshot(view).symbol

    def observe_and_update(self, symbol: str) -> None:
        t = self.state.time()
        view = self.state.view_for_timestep(t)
        snapshot = self._prediction_snapshot(view)
        pred_before = snapshot.symbol
        self._last_step_surprise = 0.0 if pred_before == symbol else 1.0
        logloss_bits = None
        if snapshot.distribution is not None:
            logloss_bits = self.loss_model.loss_bits(
                prediction=snapshot.distribution,
                observed=symbol,
                state=self.state,
                cfg=self.cfg,
                timestep=t,
            )

        # Online weight update on current observation.
        survivors: List[PoolPredictor] = []
        exits: List[Tuple[PoolPredictor, str]] = []
        for pp in self.pool:
            try:
                yhat = pp.fn(view)
            except Exception:
                exits.append((pp, "predict_error"))
                continue

            bits = self.loss_model.loss_bits(
                prediction=yhat,
                observed=symbol,
                state=self.state,
                cfg=self.cfg,
                timestep=t,
            )
            if bits is None:
                exits.append((pp, "invalid_prediction"))
                continue
            pp.weight *= math.pow(2.0, -float(bits))

            survivors.append(pp)
        self.pool = survivors

        self.state.observe(symbol)
        self._emit_observe(t, symbol, pred_before, logloss_bits, snapshot.incumbent)
        for pp, reason in exits:
            self._emit_pool_exit(t, pp, reason)

        if self.cfg.verbose:
            eprint(
                f"[v2 t={t:03d}] pred={pred_before!r} obs={symbol!r} "
                f"pool={len(self.pool)}"
            )

        if t >= int(self.cfg.min_time):
            self._explore_select_and_freeze(timestep=t)

    def _explore_select_and_freeze(self, timestep: int) -> None:
        task = self._make_window_task(timestep)
        if task is None:
            return

        explore_ctx = ExploreContext(
            timestep=timestep,
            grammar=self.g,
            task=task,
            state=self.state,
            pool=list(self.pool),
            cfg=self.cfg,
            surprise_score=float(self._last_step_surprise),
        )
        proposals = self.enumerator.propose_batch(explore_ctx)

        candidate_ctx = CandidateContext(
            timestep=timestep,
            grammar=self.g,
            task=task,
            state=self.state,
            pool=list(self.pool),
            cfg=self.cfg,
        )
        admissions = self.candidate_policy.admit_and_weight(
            candidate_ctx, proposals
        )

        self._rerank_pool_and_candidates(admissions, timestep=timestep)
        self._update_incumbent()
        self._apply_freeze_policy(timestep)

    def _make_window_task(self, current_index: int) -> Optional[Task]:
        start = max(int(self.cfg.min_time), int(current_index) - int(self.cfg.validation_window) + 1)
        if start > current_index:
            return None

        examples = [
            ((self.state.view_for_timestep(i),), self.state.obs_history[i])
            for i in range(start, current_index + 1)
        ]
        return Task(
            name=f"rbii_v2_window_{start}_{current_index}",
            request=arrow(trbii_state, tcharacter),
            examples=examples,
            cache=False,
        )

    def _predict_with_pool(self, view: RBIIEvalState) -> Optional[str]:
        return self._prediction_snapshot(view).symbol

    def _rerank_pool_and_candidates(
        self,
        admissions: Sequence[WeightedAdmission],
        timestep: int,
    ) -> None:
        target = int(self.cfg.pool_target_size)
        if target <= 0:
            self._commit_reranked_pool([], timestep)
            return

        ranked_pool = sorted(self.pool, key=lambda p: p.weight, reverse=True)
        ranked_admissions = sorted(admissions, key=lambda a: a.initial_weight, reverse=True)
        existing_pool_keys: Set[str] = {str(pp.program) for pp in self.pool}

        competition: List[tuple[str, float, Optional[PoolPredictor], Optional[WeightedAdmission]]] = []
        for pp in ranked_pool:
            competition.append(("pool", float(pp.weight), pp, None))
        for adm in ranked_admissions:
            competition.append(("candidate", float(adm.initial_weight), None, adm))

        competition.sort(key=lambda item: item[1], reverse=True)

        new_pool: List[PoolPredictor] = []
        selected_program_keys: Set[str] = set()
        for kind, weight, pp, adm in competition:
            if len(new_pool) >= target:
                break

            if kind == "pool":
                if pp is None:
                    continue
                pool_key = str(pp.program)
                if pool_key in selected_program_keys:
                    continue
                new_pool.append(pp)
                selected_program_keys.add(pool_key)
                continue

            if adm is None:
                continue
            cand = adm.proposal
            cand_key = str(cand.program)
            if cand_key in existing_pool_keys:
                continue
            if cand_key in selected_program_keys:
                continue
            fn = cand.fn
            if fn is None:
                try:
                    fn = cand.program.evaluate([])
                except Exception:
                    continue

            new_pool.append(
                PoolPredictor(
                    program=cand.program,
                    fn=fn,
                    weight=weight,
                    source=cand.source,
                    program_id=self._frozen_program_id_by_key.get(cand_key),
                    active_id=self._alloc_active_id(),
                    duplicate_candidate=False,
                )
            )
            selected_program_keys.add(cand_key)

        self._commit_reranked_pool(new_pool, timestep)

    def _current_incumbent(self) -> Optional[PoolPredictor]:
        if not self.pool:
            return None
        return max(self.pool, key=lambda p: p.weight)

    def _update_incumbent(self) -> None:
        incumbent = self._current_incumbent()
        if incumbent is None:
            self._incumbent_key = None
            self._incumbent_run_length = 0
            return

        incumbent_key = str(incumbent.program)
        if incumbent_key == self._incumbent_key:
            self._incumbent_run_length += 1
            return

        self._incumbent_key = incumbent_key
        self._incumbent_run_length = 1

    def _apply_freeze_policy(self, timestep: int) -> None:
        incumbent = self._current_incumbent()
        ctx = FreezeContext(
            timestep=timestep,
            state=self.state,
            pool=list(self.pool),
            incumbent=incumbent,
            incumbent_run_length=int(self._incumbent_run_length),
            cfg=self.cfg,
        )
        decision = self.freeze_policy.should_freeze(ctx)
        if not decision.should_freeze or decision.predictor is None:
            return
        frozen_key = str(decision.predictor.program)
        added = False
        if frozen_key in self._frozen_program_id_by_key:
            pid = self._frozen_program_id_by_key[frozen_key]
        else:
            pid = self.state.add_best_program(
                decision.predictor.program,
                birth_timestep=timestep,
            )
            self._frozen_program_id_by_key[frozen_key] = pid
            added = True

        for pp in self.pool:
            if str(pp.program) == frozen_key:
                pp.program_id = pid
        if added:
            self._emit_freeze(timestep, decision.predictor, pid, decision.reason)

    def _prediction_snapshot(self, view: RBIIEvalState) -> PredictionSnapshot:
        if not self.pool:
            return PredictionSnapshot(None, None, None)

        weighted_predictions: List[Tuple[float, Any]] = []
        valid_predictors: List[Tuple[float, PoolPredictor]] = []
        for pp in self.pool:
            try:
                yhat = pp.fn(view)
            except Exception:
                continue
            weight = max(pp.weight, 0.0)
            weighted_predictions.append((weight, yhat))
            if weight > 0.0:
                valid_predictors.append((weight, pp))

        dist = self.loss_model.mixture_distribution(
            weighted_predictions=weighted_predictions,
            state=self.state,
            cfg=self.cfg,
            timestep=view.timestep,
        )
        symbol = None if dist is None else max(dist.items(), key=lambda kv: kv[1])[0]
        incumbent = None
        if valid_predictors:
            incumbent = max(valid_predictors, key=lambda item: item[0])[1]
        return PredictionSnapshot(symbol, incumbent, dist)

    def _alloc_active_id(self) -> int:
        active_id = self._next_active_id
        self._next_active_id += 1
        return active_id

    def _commit_reranked_pool(self, new_pool: Sequence[PoolPredictor], timestep: int) -> None:
        old_ids = {pp.active_id for pp in self.pool if pp.active_id is not None}
        new_ids = {pp.active_id for pp in new_pool if pp.active_id is not None}

        for pp in self.pool:
            if pp.active_id is None or pp.active_id in new_ids:
                continue
            self._emit_pool_exit(timestep, pp, "reranked_out")

        self.pool = list(new_pool)

        for pp in self.pool:
            if pp.active_id is None or pp.active_id in old_ids:
                continue
            key = str(pp.program)
            pp.duplicate_candidate = key in self._seen_active_program_keys
            self._seen_active_program_keys.add(key)
            self._emit_pool_enter(timestep + 1, pp, "rerank_selected")

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
            loop="v2",
            config=self._event_config(),
        )
        for t, symbol in enumerate(self.state.obs_history):
            self._emit_observe(t, symbol, None, None, None, warmup=True)

    def _event_config(self) -> Dict[str, Any]:
        return {
            "pool_target_size": int(self.cfg.pool_target_size),
            "validation_window": int(self.cfg.validation_window),
            "min_time": int(self.cfg.min_time),
            "alphabet": list(self.cfg.alphabet),
            "enum_timeout_s": float(self.cfg.enum_timeout_s),
            "upper_bound": float(self.cfg.upper_bound),
            "max_frontier": int(self.cfg.max_frontier),
            "enum_cpus": int(self.cfg.enum_cpus),
        }

    def _log_event(self, event: str, **payload: Any) -> None:
        if self._event_fp is None:
            return
        self._event_fp.write(
            json.dumps({"event": event, "wall_time_s": time.time(), **payload}, sort_keys=True)
            + "\n"
        )
        self._event_fp.flush()

    def _emit_observe(
        self,
        timestep: int,
        observed: str,
        predicted: Optional[str],
        logloss_bits: Optional[float],
        incumbent: Optional[PoolPredictor],
        warmup: bool = False,
    ) -> None:
        self._log_event(
            "observe",
            timestep=timestep,
            observed=observed,
            predicted=predicted,
            logloss_bits=logloss_bits,
            active_id=(None if incumbent is None else incumbent.active_id),
            program_id=(None if incumbent is None else incumbent.program_id),
            program=(None if incumbent is None else str(incumbent.program)),
            warmup=bool(warmup),
        )

    def _emit_pool_enter(self, timestep: int, predictor: PoolPredictor, reason: str) -> None:
        self._log_event(
            "pool_enter",
            timestep=timestep,
            active_id=predictor.active_id,
            program_id=predictor.program_id,
            program=str(predictor.program),
            source=predictor.source,
            weight=float(predictor.weight),
            duplicate_candidate=bool(predictor.duplicate_candidate),
            reason=reason,
        )

    def _emit_pool_exit(self, timestep: int, predictor: PoolPredictor, reason: str) -> None:
        self._log_event(
            "pool_exit",
            timestep=timestep,
            active_id=predictor.active_id,
            program_id=predictor.program_id,
            program=str(predictor.program),
            weight=float(predictor.weight),
            duplicate_candidate=bool(predictor.duplicate_candidate),
            reason=reason,
        )

    def _emit_freeze(
        self,
        timestep: int,
        predictor: PoolPredictor,
        program_id: int,
        reason: str,
    ) -> None:
        self._log_event(
            "freeze",
            timestep=timestep,
            active_id=predictor.active_id,
            program_id=program_id,
            program=str(predictor.program),
            weight=float(predictor.weight),
            duplicate_candidate=bool(predictor.duplicate_candidate),
            reason=reason,
            incumbent_run_length=int(self._incumbent_run_length),
        )

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
