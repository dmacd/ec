import math
from argparse import Namespace
from pathlib import Path

import pytest


def _seed_state(seq: str):
    from dreamcoder.domains.rbii.rbii_state import RBIIState

    s = RBIIState()
    for ch in seq:
        s.observe(ch)
    return s


class _FakeProgram:
    def __init__(self, name: str, value):
        self.name = name
        self._value = value

    def __str__(self):
        return self.name

    def evaluate(self, _environment):
        return self._value


class _StaticEnumerator:
    def __init__(self, proposals):
        self.proposals = list(proposals)

    def propose_batch(self, _ctx):
        return list(self.proposals)


class _PassThroughCandidatePolicy:
    def __init__(self, weight: float = 1.0):
        self.weight = float(weight)

    def admit_and_weight(self, _ctx, candidates):
        from dreamcoder.domains.rbii.rbii_loop_v2 import WeightedAdmission

        return [
            WeightedAdmission(proposal=cand, initial_weight=self.weight, reason="test")
            for cand in candidates
        ]


class _NeverFreezePolicy:
    def should_freeze(self, _ctx):
        from dreamcoder.domains.rbii.rbii_loop_v2 import FreezeDecision

        return FreezeDecision(False, None, reason="test_never")


class _AlwaysFreezePolicy:
    def should_freeze(self, ctx):
        from dreamcoder.domains.rbii.rbii_loop_v2 import FreezeDecision

        return FreezeDecision(
            should_freeze=(ctx.incumbent is not None),
            predictor=ctx.incumbent,
            reason="test_always",
        )


def test_rbii_config_v2_requires_canonical_alphabet_tuple():
    from dreamcoder.domains.rbii.rbii_loop_v2 import RBIIConfigV2

    with pytest.raises(TypeError):
        RBIIConfigV2(alphabet=["a", "b"])

    with pytest.raises(ValueError):
        RBIIConfigV2(alphabet=())

    with pytest.raises(ValueError):
        RBIIConfigV2(alphabet=("a", "a"))

    cfg = RBIIConfigV2(alphabet=("a", "b", "c"), verbose=False)
    assert cfg.alphabet == ("a", "b", "c")
    assert cfg.alphabet_set == frozenset({"a", "b", "c"})
    assert cfg.baseline_bits_per_symbol == pytest.approx(math.log2(3.0))


def test_categorical_log_loss_model_handles_nonbinary_symbol_and_distribution_outputs():
    from dreamcoder.domains.rbii.rbii_loop_v2 import RBIIConfigV2
    from dreamcoder.domains.rbii.rbii_loss import CategoricalLogLossModel

    cfg = RBIIConfigV2(
        alphabet=("a", "b", "c"),
        deterministic_smoothing_eps=0.3,
        verbose=False,
    )
    state = _seed_state("")
    model = CategoricalLogLossModel()

    baseline = model.baseline_bits(
        state=state,
        cfg=cfg,
        start_timestep=0,
        end_timestep=1,
    )
    assert baseline == pytest.approx(2.0 * math.log2(3.0))

    correct_bits = model.loss_bits(
        prediction="a",
        observed="a",
        state=state,
        cfg=cfg,
        timestep=0,
    )
    wrong_bits = model.loss_bits(
        prediction="a",
        observed="b",
        state=state,
        cfg=cfg,
        timestep=0,
    )
    dict_bits = model.loss_bits(
        prediction={"a": 2.0, "b": 1.0, "c": 1.0},
        observed="a",
        state=state,
        cfg=cfg,
        timestep=0,
    )
    seq_bits = model.loss_bits(
        prediction=(3.0, 1.0, 0.0),
        observed="a",
        state=state,
        cfg=cfg,
        timestep=0,
    )

    assert correct_bits == pytest.approx(-math.log2(0.7))
    assert wrong_bits == pytest.approx(-math.log2(0.15))
    assert dict_bits == pytest.approx(1.0)
    assert seq_bits == pytest.approx(-math.log2(0.75))


def test_categorical_log_loss_model_mixture_predict_symbol_uses_weighted_distribution_mass():
    from dreamcoder.domains.rbii.rbii_loop_v2 import RBIIConfigV2
    from dreamcoder.domains.rbii.rbii_loss import CategoricalLogLossModel

    cfg = RBIIConfigV2(alphabet=("a", "b"), verbose=False)
    model = CategoricalLogLossModel()
    state = _seed_state("")

    pred = model.mixture_predict_symbol(
        weighted_predictions=[
            (1.0, {"a": 0.2, "b": 0.8}),
            (2.0, (0.9, 0.1)),
        ],
        state=state,
        cfg=cfg,
        timestep=0,
    )

    assert pred == "a"


def test_mdl_policy_respects_compression_gain_slack():
    from dreamcoder.domains.rbii.rbii_loop_v2 import (
        CandidateContext,
        CandidateProposal,
        MDLDetectabilityPolicy,
        RBIIConfigV2,
    )
    from dreamcoder.domains.rbii.rbii_loss import CategoricalLogLossModel

    state = _seed_state("a")
    program = _FakeProgram("always_a", lambda _view: "a")
    proposal = CandidateProposal(
        program=program,
        witness_bits=1.65,
        fn=program.evaluate([]),
    )
    loss_model = CategoricalLogLossModel()

    strict_cfg = RBIIConfigV2(
        alphabet=("a", "b", "c"),
        min_time=0,
        validation_window=1,
        compression_gain_slack_bits=0.0,
        verbose=False,
    )
    loose_cfg = RBIIConfigV2(
        alphabet=("a", "b", "c"),
        min_time=0,
        validation_window=1,
        compression_gain_slack_bits=0.1,
        verbose=False,
    )

    strict_ctx = CandidateContext(
        timestep=0,
        grammar=None,
        task=None,
        state=state,
        pool=[],
        cfg=strict_cfg,
    )
    loose_ctx = CandidateContext(
        timestep=0,
        grammar=None,
        task=None,
        state=state,
        pool=[],
        cfg=loose_cfg,
    )

    policy = MDLDetectabilityPolicy(loss_model=loss_model)
    assert policy.admit_and_weight(strict_ctx, [proposal]) == []
    assert len(policy.admit_and_weight(loose_ctx, [proposal])) == 1


def test_rerank_skips_candidate_duplicate_already_in_pool():
    from dreamcoder.domains.rbii.rbii_loop_v2 import (
        CandidateProposal,
        PoolPredictor,
        RBIILoopV2,
        RBIIConfigV2,
        WeightedAdmission,
    )

    cfg = RBIIConfigV2(alphabet=("a", "b"), pool_target_size=2, verbose=False)
    loop = RBIILoopV2(
        grammar=None,
        state=_seed_state(""),
        cfg=cfg,
        enumerator=_StaticEnumerator([]),
        candidate_policy=_PassThroughCandidatePolicy(),
        freeze_policy=_NeverFreezePolicy(),
    )

    p1 = _FakeProgram("p1", lambda _view: "a")
    p2 = _FakeProgram("p2", lambda _view: "b")
    loop.pool = [
        PoolPredictor(program=p1, fn=p1.evaluate([]), weight=1.0),
    ]

    admissions = [
        WeightedAdmission(
            proposal=CandidateProposal(program=p1, witness_bits=0.0, fn=p1.evaluate([])),
            initial_weight=10.0,
        ),
        WeightedAdmission(
            proposal=CandidateProposal(program=p2, witness_bits=0.0, fn=p2.evaluate([])),
            initial_weight=0.5,
        ),
    ]

    loop._rerank_pool_and_candidates(admissions, timestep=0)

    assert [str(pp.program) for pp in loop.pool] == ["p1", "p2"]


def test_v2_pool_admission_does_not_freeze_without_freeze_policy():
    from dreamcoder.domains.rbii.rbii_loop_v2 import CandidateProposal, RBIILoopV2, RBIIConfigV2

    state = _seed_state("")
    cfg = RBIIConfigV2(
        alphabet=("a", "b"),
        min_time=0,
        validation_window=1,
        pool_target_size=1,
        verbose=False,
    )
    program = _FakeProgram("always_a", lambda _view: "a")
    proposal = CandidateProposal(
        program=program,
        witness_bits=0.0,
        fn=program.evaluate([]),
    )
    loop = RBIILoopV2(
        grammar=None,
        state=state,
        cfg=cfg,
        enumerator=_StaticEnumerator([proposal]),
        candidate_policy=_PassThroughCandidatePolicy(weight=1.0),
        freeze_policy=_NeverFreezePolicy(),
    )

    loop.observe_and_update("a")

    assert len(loop.pool) == 1
    assert str(loop.pool[0].program) == "always_a"
    assert state.best_programs == []


def test_v2_freeze_policy_is_only_path_into_best_programs():
    from dreamcoder.domains.rbii.rbii_loop_v2 import CandidateProposal, RBIILoopV2, RBIIConfigV2

    state = _seed_state("")
    cfg = RBIIConfigV2(
        alphabet=("a", "b"),
        min_time=0,
        validation_window=1,
        pool_target_size=1,
        verbose=False,
    )
    program = _FakeProgram("always_a", lambda _view: "a")
    proposal = CandidateProposal(
        program=program,
        witness_bits=0.0,
        fn=program.evaluate([]),
    )
    loop = RBIILoopV2(
        grammar=None,
        state=state,
        cfg=cfg,
        enumerator=_StaticEnumerator([proposal]),
        candidate_policy=_PassThroughCandidatePolicy(weight=1.0),
        freeze_policy=_AlwaysFreezePolicy(),
    )

    loop.observe_and_update("a")
    assert len(state.best_programs) == 1
    assert str(state.best_programs[0]) == "always_a"
    assert loop.pool[0].program_id == 0

    loop.observe_and_update("a")
    assert len(state.best_programs) == 1
    assert loop.pool[0].program_id == 0


def test_v2_bottom_solver_smoke():
    from dreamcoder.domains.rbii.rbii_loop_v2 import RBIILoopV2, RBIIConfigV2
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar

    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    state = _seed_state("aaa")
    cfg = RBIIConfigV2(
        alphabet=("a", "b"),
        pool_target_size=1,
        validation_window=1,
        min_time=3,
        enum_timeout_s=1.0,
        eval_timeout_s=0.02,
        upper_bound=12.0,
        budget_increment=1.5,
        max_frontier=5,
        enum_cpus=1,
        enum_bottom_compile_me=False,
        verbose=False,
    )

    loop = RBIILoopV2(grammar=grammar, state=state, cfg=cfg)
    loop.observe_and_update("a")

    assert state.time() == 4


def test_v2_event_log_and_viz_smoke(tmp_path: Path):
    from dreamcoder.domains.rbii.rbii_loop_v2 import CandidateProposal, RBIILoopV2, RBIIConfigV2
    from dreamcoder.domains.rbii.rbii_viz_graph import _build_svg_for_log

    state = _seed_state("a")
    cfg = RBIIConfigV2(
        alphabet=("a", "b"),
        min_time=0,
        validation_window=1,
        pool_target_size=1,
        verbose=False,
        event_log_dir=str(tmp_path),
        event_log_name="loop_v2_smoke",
    )
    program = _FakeProgram("always_a", lambda _view: "a")
    proposal = CandidateProposal(
        program=program,
        witness_bits=0.0,
        fn=program.evaluate([]),
    )
    loop = RBIILoopV2(
        grammar=None,
        state=state,
        cfg=cfg,
        enumerator=_StaticEnumerator([proposal]),
        candidate_policy=_PassThroughCandidatePolicy(weight=1.0),
        freeze_policy=_AlwaysFreezePolicy(),
    )

    loop.observe_and_update("a")
    loop.observe_and_update("a")
    loop.close()

    log_path = tmp_path / "loop_v2_smoke.jsonl"
    rows = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows

    import json

    events = [json.loads(row) for row in rows]
    assert events[0]["event"] == "run_start"
    assert any(ev["event"] == "pool_enter" for ev in events)
    assert any(ev["event"] == "freeze" for ev in events)
    assert events[-1]["event"] == "run_end"

    observe_rows = [ev for ev in events if ev["event"] == "observe" and not ev.get("warmup")]
    assert observe_rows
    assert any(isinstance(ev.get("logloss_bits"), float) for ev in observe_rows)
    assert any(isinstance(ev.get("active_id"), int) for ev in observe_rows)

    svg = _build_svg_for_log(
        log_path,
        Namespace(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
            logs=[],
            format="svg",
            show_timestep_labels=False,
            label_mode="program",
            max_program_label_len=84,
            show_program_map=False,
            row_step=22.0,
            lane_step=34.0,
            connector_len=68.0,
            min_box_margin=5.0,
            font_family="Helvetica,Arial,sans-serif",
            code_font_family="Menlo,Consolas,Monaco,'Courier New',monospace",
        ),
    )

    assert "frozen store" in svg
    assert "@1" in svg
    assert "[#0]" in svg


def test_v2_viz_staggers_same_timestep_episode_joins():
    from dreamcoder.domains.rbii.rbii_viz_graph import ProgramEpisode, _assign_lanes, _layout_episode_boxes

    episodes = [
        ProgramEpisode(active_id=1, program_id=None, program_text="p1", start_t=8, end_t=8, duplicate_candidate=False),
        ProgramEpisode(active_id=2, program_id=None, program_text="p2", start_t=8, end_t=8, duplicate_candidate=False),
        ProgramEpisode(active_id=3, program_id=None, program_text="p3", start_t=8, end_t=8, duplicate_candidate=False),
    ]
    _assign_lanes(episodes)
    layouts = _layout_episode_boxes(
        episodes=episodes,
        labels=["@1 p1", "@2 p2", "@3 p3"],
        y_by_t={8: 100.0},
        bracket_x0=100.0,
        lane_step=34.0,
        connector_len=68.0,
        label_gap=10.0,
        min_box_margin=5.0,
        font_size=13.0,
        pad_x=9.0,
        pad_y=6.0,
    )

    assert len(layouts) == 3
    assert len({round(item.y1, 2) for item in layouts}) == 3
    assert len({round(item.router_x, 2) for item in layouts}) == 3


def test_v2_viz_uses_distinct_router_columns_even_when_lane_reused():
    from dreamcoder.domains.rbii.rbii_viz_graph import ProgramEpisode, _assign_lanes, _layout_episode_boxes

    episodes = [
        ProgramEpisode(active_id=1, program_id=None, program_text="p1", start_t=1, end_t=1, duplicate_candidate=False),
        ProgramEpisode(active_id=2, program_id=None, program_text="p2", start_t=3, end_t=3, duplicate_candidate=False),
        ProgramEpisode(active_id=3, program_id=None, program_text="p3", start_t=5, end_t=5, duplicate_candidate=False),
    ]
    _assign_lanes(episodes)
    assert {ep.lane for ep in episodes} == {0}

    layouts = _layout_episode_boxes(
        episodes=episodes,
        labels=["@1 p1", "@2 p2", "@3 p3"],
        y_by_t={1: 60.0, 3: 100.0, 5: 140.0},
        bracket_x0=100.0,
        lane_step=34.0,
        connector_len=68.0,
        label_gap=10.0,
        min_box_margin=5.0,
        font_size=13.0,
        pad_x=9.0,
        pad_y=6.0,
    )

    assert len(layouts) == 3
    assert len({round(item.router_x, 2) for item in layouts}) == 3
