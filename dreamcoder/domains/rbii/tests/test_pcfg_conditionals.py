import math

from dreamcoder.domains.rbii.rbii_primitives import (
    RBIIPrimitiveConfig,
    make_rbii_grammar,
)
from dreamcoder.domains.rbii.rbii_types import trbii_state
from dreamcoder.grammar import PCFG
from dreamcoder.program import NamedHole, Primitive, Program
from dreamcoder.type import Context, arrow, tbool, tint, tcharacter
from dreamcoder.utilities import NEGATIVEINFINITY, lse


def _make_rbii_pcfg(alphabet: str = "ab", max_int: int = 2):
    grammar = make_rbii_grammar(
        RBIIPrimitiveConfig(alphabet=alphabet, max_int=max_int, log_variable=0.0)
    )
    request = arrow(trbii_state, tcharacter)
    pcfg = PCFG.from_grammar(grammar, request)
    return grammar, pcfg, pcfg.number_rules()


def _manual_pcfg(productions):
    return PCFG(productions, 0, 0)


def test_pcfg_instantiates_if_over_arrow_types():
    _grammar, pcfg, _ = _make_rbii_pcfg()

    higher_order_if_rules = [
        arguments
        for _lp, constructor, arguments in pcfg.productions[pcfg.start_symbol]
        if getattr(constructor, "name", None) == "if"
        and len(arguments) == 4
        and arguments[1][0] == 1
        and arguments[2][0] == 1
    ]

    assert higher_order_if_rules


def test_pcfg_if_rules_require_stateful_condition_children():
    _grammar, pcfg, _ = _make_rbii_pcfg()

    if_rules = [
        arguments
        for _lp, constructor, arguments in pcfg.productions[pcfg.start_symbol]
        if getattr(constructor, "name", None) == "if"
    ]

    assert if_rules
    assert all(arguments[0][1][-1] is True for arguments in if_rules)


def test_build_candidates_instantiates_if_over_arrow_types():
    grammar, _pcfg, _ = _make_rbii_pcfg()

    if_candidates = [
        tp
        for _l, tp, primitive, _new_context in grammar.buildCandidates(
            tint, Context.EMPTY, [trbii_state], normalize=False, returnTable=False
        )
        if getattr(primitive, "name", None) == "if"
    ]

    assert any(len(tp.functionArguments()) == 4 for tp in if_candidates)


def test_build_candidates_orders_polymorphic_if_instantiations_deterministically():
    grammar, _pcfg, _ = _make_rbii_pcfg(alphabet="abc", max_int=3)

    if_candidates = [
        str(tp)
        for _l, tp, primitive, _new_context in grammar.buildCandidates(
            trbii_state, Context.EMPTY, [tbool, trbii_state], normalize=False, returnTable=False
        )
        if getattr(primitive, "name", None) == "if"
    ]

    assert if_candidates == sorted(if_candidates)


def test_pcfg_log_probability_distinguishes_higher_order_and_oversaturated_if():
    _grammar, _pcfg, pcfg = _make_rbii_pcfg()

    higher_order = Program.parse(
        "(lambda "
        "(if "
        "(not (eq_char (get_historical_obs 0 $0) (get_historical_obs 0 $0))) "
        "(lambda (get_historical_obs 1 $0)) "
        "(lambda (get_historical_obs 0 $0)) "
        "$0))"
    )
    oversaturated = Program.parse(
        "(lambda "
        "(((if "
        "(not (eq_char (get_historical_obs 0 $0) (get_historical_obs 0 $0))) "
        "(lambda (get_historical_obs 1 $0)) "
        "(lambda (get_historical_obs 0 $0))) "
        "$0) "
        "$0))"
    )

    higher_order_lp = pcfg.log_probability(higher_order)
    oversaturated_lp = pcfg.log_probability(oversaturated)

    assert math.isfinite(higher_order_lp)
    assert oversaturated_lp == NEGATIVEINFINITY


def test_pcfg_rejects_constant_condition_if_under_stateful_split():
    _grammar, _pcfg, pcfg = _make_rbii_pcfg(alphabet="abc", max_int=3)

    constant_condition = Program.parse(
        "(lambda "
        "(if "
        "(eq_char a a) "
        "b "
        "c))"
    )

    assert pcfg.log_probability(constant_condition) == NEGATIVEINFINITY


def test_grammar_log_likelihood_backtracks_across_higher_order_if_instantiations():
    grammar, _pcfg, _ = _make_rbii_pcfg(alphabet="abc", max_int=3)

    program = Program.parse(
        "(lambda "
        "(if "
        "(eq_char a a) "
        "(lambda c) "
        "(lambda (get_historical_obs 0 $0)) "
        "$0))"
    )

    log_likelihood = grammar.logLikelihood(arrow(trbii_state, tcharacter), program)

    assert math.isfinite(log_likelihood)


def test_quantized_enumeration_uses_fourth_argument_nonterminal():
    constructor = Primitive("qenum_f4", tint, None)
    a1 = Primitive("qenum_a1", tint, None)
    a2 = Primitive("qenum_a2", tint, None)
    a3 = Primitive("qenum_a3", tint, None)
    a4 = Primitive("qenum_a4", tint, None)
    pcfg = _manual_pcfg(
        [
            [(0.0, constructor, [(0, 1), (0, 2), (0, 3), (0, 4)])],
            [(0.0, a1, [])],
            [(0.0, a2, [])],
            [(0.0, a3, [])],
            [(0.0, a4, [])],
        ]
    )

    program = next(pcfg.quantized_enumeration(skeletons=[NamedHole(0)]))

    assert str(program) == "(qenum_f4 qenum_a1 qenum_a2 qenum_a3 qenum_a4)"


def test_quantized_enumeration_uses_fifth_argument_nonterminal_and_lambda_count():
    constructor = Primitive("qenum_f5", tint, None)
    a1 = Primitive("qenum_b1", tint, None)
    a2 = Primitive("qenum_b2", tint, None)
    a3 = Primitive("qenum_b3", tint, None)
    a4 = Primitive("qenum_b4", tint, None)
    a5 = Primitive("qenum_b5", tint, None)
    pcfg = _manual_pcfg(
        [
            [(0.0, constructor, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)])],
            [(0.0, a1, [])],
            [(0.0, a2, [])],
            [(0.0, a3, [])],
            [(0.0, a4, [])],
            [(0.0, a5, [])],
        ]
    )

    program = next(pcfg.quantized_enumeration(skeletons=[NamedHole(0)]))

    assert str(program) == "(qenum_f5 qenum_b1 qenum_b2 qenum_b3 qenum_b4 (lambda qenum_b5))"


def test_lse_all_negative_infinity_returns_negative_infinity():
    assert lse([NEGATIVEINFINITY, NEGATIVEINFINITY]) == NEGATIVEINFINITY
    assert lse(NEGATIVEINFINITY, NEGATIVEINFINITY) == NEGATIVEINFINITY
