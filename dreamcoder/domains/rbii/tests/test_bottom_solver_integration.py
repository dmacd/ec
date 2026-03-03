import pytest


def _seed_state(seq: str):
    from dreamcoder.domains.rbii.rbii_state import RBIIState

    s = RBIIState()
    for ch in seq:
        s.observe(ch)
    return s


def test_bottom_solver_runs_on_rbii_task():
    """
    Integration test: actually executes solveForTask_bottom (compile_me=False)
    on an RBII state-view task.
    """
    from dreamcoder.enumeration import solveForTask_bottom
    from dreamcoder.frontier import Frontier
    from dreamcoder.task import Task
    from dreamcoder.type import arrow, tcharacter
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
    from dreamcoder.domains.rbii.rbii_types import trbii_state
    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    state = _seed_state("aaa")
    timestep = state.time()  # 3
    view = state.view_for_timestep(timestep)
    task = Task(
        name="rbii_bottom_integration",
        request=arrow(trbii_state, tcharacter),
        examples=[((view,), "a")],
        cache=False,
    )

    frontiers, search_times, total = solveForTask_bottom(
        g=grammar,
        tasks=[task],
        lowerBound=0.0,
        upperBound=12.0,
        budgetIncrement=1.5,
        timeout=1.0,
        CPUs=1,
        likelihoodModel=None,
        evaluationTimeout=0.02,
        maximumFrontiers={task: 5},
        testing=False,
        compile_me=False,
    )

    assert task in frontiers
    assert isinstance(frontiers[task], Frontier)
    assert task in search_times
    assert isinstance(total, int)
    assert total > 0


def test_bottom_solver_parallel_smoke_or_skip():
    """
    Optional parallel smoke test: runs bottom solver with CPUs=2 when
    multiprocessing semaphores are available.
    """
    from dreamcoder.enumeration import solveForTask_bottom
    from dreamcoder.task import Task
    from dreamcoder.type import arrow, tcharacter
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
    from dreamcoder.domains.rbii.rbii_types import trbii_state
    from dreamcoder import utilities as dc_utils

    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    state = _seed_state("aaa")
    timestep = state.time()
    view = state.view_for_timestep(timestep)
    task = Task(
        name="rbii_bottom_parallel_smoke",
        request=arrow(trbii_state, tcharacter),
        examples=[((view,), "a")],
        cache=False,
    )

    try:
        _frontiers, _search_times, total = solveForTask_bottom(
            g=grammar,
            tasks=[task],
            lowerBound=0.0,
            upperBound=12.0,
            budgetIncrement=1.5,
            timeout=1.0,
            CPUs=2,
            likelihoodModel=None,
            evaluationTimeout=0.02,
            maximumFrontiers={task: 5},
            testing=False,
            compile_me=False,
        )
    except PermissionError:
        dc_utils.PARALLELMAPDATA = None
        dc_utils.PARALLELBASESEED = None
        pytest.skip("multiprocessing semaphores unavailable in this environment")

    assert total > 0


def test_succ_char_primitive_value_is_picklable():
    """
    Regression test for multiprocessing: succ_char primitive value must be
    picklable and preserve behavior after roundtrip.
    """
    import pickle
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar

    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    succ_primitive = None
    for _lp, _tp, p in grammar.productions:
        if p.name == "succ_char":
            succ_primitive = p
            break

    assert succ_primitive is not None
    payload = pickle.dumps(succ_primitive.value)
    recovered = pickle.loads(payload)
    assert recovered("a") == "b"
    assert recovered("b") == "b"


def test_bottom_solver_parallel_handles_alternating_ab_window_or_skip():
    """
    Parallel bottom solver on the previously failing alternating_ab window
    should not raise MaybeEncodingError from succ_char pickling.
    """
    import multiprocessing.pool as mp_pool
    from dreamcoder.enumeration import solveForTask_bottom
    from dreamcoder.task import Task
    from dreamcoder.type import arrow, tcharacter
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
    from dreamcoder.domains.rbii.rbii_types import trbii_state
    from dreamcoder import utilities as dc_utils

    # Former failure case from rbii_test: history "aba", predict "b".
    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    state = _seed_state("aba")
    view = state.view_for_timestep(3)
    task = Task(
        name="rbii_bottom_parallel_pickling_error",
        request=arrow(trbii_state, tcharacter),
        examples=[((view,), "b")],
        cache=False,
    )

    try:
        _frontiers, _search_times, total = solveForTask_bottom(
            g=grammar,
            tasks=[task],
            lowerBound=0.0,
            upperBound=12.0,
            budgetIncrement=1.5,
            timeout=1.0,
            CPUs=2,
            likelihoodModel=None,
            evaluationTimeout=0.02,
            maximumFrontiers={task: 10},
            testing=False,
            compile_me=False,
        )
    except PermissionError:
        dc_utils.PARALLELMAPDATA = None
        dc_utils.PARALLELBASESEED = None
        pytest.skip("multiprocessing semaphores unavailable in this environment")
    except mp_pool.MaybeEncodingError as e:
        pytest.fail(f"Unexpected multiprocessing encoding failure: {e}")

    assert total > 0


def test_rbii_loop_bottom_mode_runs_without_debug_hooks(tmp_path):
    """
    Integration test: run one actual RBII update in bottom mode and ensure
    debug hooks are not touched.
    """
    from dreamcoder.domains.rbii.rbii_loop import RBIIConfig, RBIILoop
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar
    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    state = _seed_state("aaa")  # warmup to min_time

    cfg = RBIIConfig(
        pool_target_size=1,
        validation_window=1,
        min_time=3,
        enum_timeout_s=1.0,
        eval_timeout_s=0.02,
        upper_bound=12.0,
        budget_increment=1.5,
        max_frontier=5,
        enum_solver="bottom",
        enum_cpus=1,
        enum_bottom_compile_me=False,
        verbose=False,
        event_log_dir=str(tmp_path),
        event_log_name="rbii_bottom_mode",
        log_candidate_events=True,
    )

    def fail_if_called(_current_index, _task):
        raise AssertionError("debug hook factory should not be used in bottom mode")

    loop = RBIILoop(
        grammar=grammar,
        state=state,
        cfg=cfg,
        enumeration_debug_hooks_factory=fail_if_called,
    )
    loop.observe_and_update("a")
    loop.close()

    assert state.time() == 4


def test_rbii_loop_bottom_parallel_survives_second_refill_or_skip(tmp_path):
    """
    Regression: with bottom+parallel, second refill used to fail when Task
    examples captured RBIIState containing non-picklable compiled lambdas.
    """
    import multiprocessing.pool as mp_pool
    from dreamcoder import utilities as dc_utils
    from dreamcoder.domains.rbii.rbii_loop import RBIIConfig, RBIILoop
    from dreamcoder.domains.rbii.rbii_primitives import RBIIPrimitiveConfig, make_rbii_grammar

    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet="ab", max_int=2, log_variable=0.0)
    )
    state = _seed_state("aba")
    cfg = RBIIConfig(
        pool_target_size=3,
        validation_window=2,
        min_time=3,
        enum_timeout_s=1.0,
        eval_timeout_s=0.02,
        upper_bound=12.0,
        budget_increment=1.5,
        max_frontier=10,
        enum_solver="bottom",
        enum_cpus=2,
        enum_bottom_compile_me=False,
        verbose=False,
        event_log_dir=str(tmp_path),
        event_log_name="rbii_bottom_parallel_second_refill",
        log_candidate_events=True,
    )

    loop = RBIILoop(grammar=grammar, state=state, cfg=cfg)
    try:
        # First update populates pool and state.best_programs at t=3.
        loop.observe_and_update("b")
        # Second update triggers another refill at t=4 with existing programs.
        loop.observe_and_update("a")
    except PermissionError:
        dc_utils.PARALLELMAPDATA = None
        dc_utils.PARALLELBASESEED = None
        pytest.skip("multiprocessing semaphores unavailable in this environment")
    except mp_pool.MaybeEncodingError as e:
        pytest.fail(f"Unexpected multiprocessing encoding failure: {e}")
    finally:
        loop.close()

    assert state.time() == 5
