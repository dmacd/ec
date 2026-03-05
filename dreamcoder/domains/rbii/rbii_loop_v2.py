from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set

from dreamcoder.enumeration import (
    EnumerationDebugHook,
    NOOP_ENUMERATION_DEBUG_HOOK,
    solveForTask_bottom,
)
from dreamcoder.program import Program
from dreamcoder.task import Task
from dreamcoder.type import arrow, tcharacter
from dreamcoder.utilities import eprint

from .rbii_state import RBIIState
from .rbii_types import RBIIEvalState, trbii_state


@dataclass
class PoolPredictor:
    program: Program
    fn: Callable[[RBIIEvalState], str]
    weight: float = 1.0
    source: str = "enumerated"
    program_id: Optional[int] = None
    duplicate_candidate: bool = False


@dataclass
class CandidateProposal:
    program: Program
    witness_bits: float
    fn: Optional[Callable[[RBIIEvalState], str]] = None
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
    candidate_buffer: Sequence[CandidateProposal]
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
    candidate_buffer_cap: int = 64

    mispredict_penalty: float = 0.1

    verbose: bool = True


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
    - assigns a comparable insertion weight to every scored candidate
      via Z * 2^{-MDL score}
    """

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

        ranked: List[tuple[float, CandidateProposal]] = []
        for cand in candidates:
            loss = self._window_loss_bits(ctx, cand)
            if loss is None:
                continue

            k_hat = float(cand.witness_bits)
            if k_hat < 0.0:
                raise ValueError(f"witness_bits must be non-negative; got {k_hat}.")
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
            loss += 0.0 if yhat == y else 1.0
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
    ):
        self.g = grammar
        self.state = state
        self.cfg = cfg or RBIIConfigV2()

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
        self.candidate_policy = candidate_policy or MDLDetectabilityPolicy()
        self.freeze_policy = freeze_policy or AlwaysFreezeIncumbentPolicy()

        self.pool: List[PoolPredictor] = []
        self.candidate_buffer: List[CandidateProposal] = []
        self._frozen_program_id_by_key: Dict[str, int] = {}
        for i, p in enumerate(self.state.best_programs):
            k = str(p)
            if k not in self._frozen_program_id_by_key:
                self._frozen_program_id_by_key[k] = i
        self._incumbent_key: Optional[str] = None
        self._incumbent_run_length: int = 0
        self._last_step_surprise: float = 0.0

    def predict_next(self) -> Optional[str]:
        t = self.state.time()
        view = self.state.view_for_timestep(t)
        return self._predict_with_pool(view)

    def observe_and_update(self, symbol: str) -> None:
        t = self.state.time()
        view = self.state.view_for_timestep(t)
        pred_before = self._predict_with_pool(view)
        self._last_step_surprise = 0.0 if pred_before == symbol else 1.0

        # Online weight update on current observation.
        survivors: List[PoolPredictor] = []
        for pp in self.pool:
            try:
                yhat = pp.fn(view)
            except Exception:
                continue

            if yhat == symbol:
                pp.weight *= 1.0
            else:
                pp.weight *= float(self.cfg.mispredict_penalty)

            survivors.append(pp)
        self.pool = survivors

        self.state.observe(symbol)

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

        if self.cfg.candidate_buffer_cap > 0:
            self.candidate_buffer.extend(proposals)
            if len(self.candidate_buffer) > int(self.cfg.candidate_buffer_cap):
                self.candidate_buffer = self.candidate_buffer[-int(self.cfg.candidate_buffer_cap) :]

        candidate_ctx = CandidateContext(
            timestep=timestep,
            grammar=self.g,
            task=task,
            state=self.state,
            pool=list(self.pool),
            candidate_buffer=list(self.candidate_buffer),
            cfg=self.cfg,
        )
        admissions = self.candidate_policy.admit_and_weight(
            candidate_ctx, list(self.candidate_buffer)
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
        if not self.pool:
            return None

        votes: Dict[str, float] = {}
        for pp in self.pool:
            try:
                yhat = pp.fn(view)
            except Exception:
                continue
            votes[yhat] = votes.get(yhat, 0.0) + max(pp.weight, 0.0)

        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def _rerank_pool_and_candidates(
        self,
        admissions: Sequence[WeightedAdmission],
        timestep: int,
    ) -> None:
        _ = timestep
        target = int(self.cfg.pool_target_size)
        if target <= 0:
            self.pool = []
            self.candidate_buffer = []
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
        selected_candidate_ids: Set[int] = set()
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
                    program_id=None,
                    duplicate_candidate=False,
                )
            )
            selected_candidate_ids.add(id(cand))
            selected_program_keys.add(cand_key)

        self.pool = new_pool
        if selected_candidate_ids:
            self.candidate_buffer = [
                c for c in self.candidate_buffer if id(c) not in selected_candidate_ids
            ]

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
        if frozen_key in self._frozen_program_id_by_key:
            pid = self._frozen_program_id_by_key[frozen_key]
        else:
            pid = self.state.add_best_program(
                decision.predictor.program,
                birth_timestep=timestep,
            )
            self._frozen_program_id_by_key[frozen_key] = pid

        for pp in self.pool:
            if str(pp.program) == frozen_key:
                pp.program_id = pid
