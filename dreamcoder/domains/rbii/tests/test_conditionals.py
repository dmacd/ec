from typing import Tuple

from dreamcoder.domains.rbii.rbii_primitives import (
    RBIIPrimitiveConfig,
    make_rbii_grammar,
)
from dreamcoder.domains.rbii.rbii_state import RBIIState
from dreamcoder.domains.rbii.rbii_types import trbii_state
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.type import arrow, tcharacter


def _inner_parse_rbii_program(
    source: str, alphabet: str = "abcde", max_int: int = 6
) -> Tuple[Program, Grammar]:
    g = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet=alphabet, max_int=max_int, log_variable=0.0)
    )
    return Program.parse(source), g


def _parse_rbii_program(
    source: str, alphabet: str = "abcde", max_int: int = 6) -> Program:
    program, g = _inner_parse_rbii_program(source, alphabet, max_int)
    return program

def _seed_state(seq: str) -> RBIIState:
    state = RBIIState()
    for ch in seq:
        state.observe(ch)
    return state


def test_lazy_if_can_select_function_valued_branch_before_applying_view():
    program = _parse_rbii_program(
        "(lambda ((if (eq_char (get_historical_obs 0 $0) a) "
        "(get_historical_obs 1) "
        "(get_historical_obs 0)) "
        "$0))"
    )
    fn = program.evaluate([])

    assert fn(_seed_state("aba").view_for_timestep(3)) == "b"
    assert fn(_seed_state("abb").view_for_timestep(3)) == "b"


def test_lazy_if_does_not_force_false_branch_when_true_branch_is_selected():
    program, g = _inner_parse_rbii_program(
        "(lambda ((if (eq_char (get_historical_obs 0 $0) a) "
        "(get_historical_obs 0) "
        "(get_historical_obs 1)) "
        "$0))"
    )
    fn = program.evaluate([])

    # ll = g.logLikelihood(arrow(trbii_state, tcharacter), program)
    # print(f"log likelihood of program {program}: {ll}")

    assert fn(_seed_state("a").view_for_timestep(1)) == "a"


def test_lazy_if_does_not_force_true_branch_when_false_branch_is_selected():
    program = _parse_rbii_program(
        "(lambda ((if (eq_char (get_historical_obs 0 $0) a) "
        "(get_historical_obs 1) "
        "(get_historical_obs 0)) "
        "$0))"
    )
    fn = program.evaluate([])

    assert fn(_seed_state("b").view_for_timestep(1)) == "b"


def test_lazy_if_selected_branch_can_read_visible_programs_from_view():
    program = _parse_rbii_program(
        "(lambda ((if (eq_char (get_historical_obs 0 $0) a) "
        "(get_historical_program 0) "
        "(get_historical_obs 0)) "
        "$0))"
    )
    stored_program = _parse_rbii_program("(lambda b)")

    state = _seed_state("a")
    state.add_best_program(stored_program, birth_timestep=0)

    fn = program.evaluate([])
    assert fn(state.view_for_timestep(1)) == "b"


def test_pairs_cycle_reference_conditional_program_predicts_full_sequence():
    program = _parse_rbii_program(
        "(lambda "
        "(if (eq_char (get_historical_obs 0 $0) (get_historical_obs 1 $0)) "
        "(if (eq_char (get_historical_obs 0 $0) e) "
        "a "
        "(succ_char (get_historical_obs 0 $0))) "
        "(get_historical_obs 0 $0)))",
        alphabet="abcde",
        max_int=6,
    )
    fn = program.evaluate([])

    seq = "aabbccddee" * 4
    state = _seed_state(seq[:2])

    for t in range(2, len(seq)):
        assert fn(state.view_for_timestep(t)) == seq[t]
        state.observe(seq[t])
