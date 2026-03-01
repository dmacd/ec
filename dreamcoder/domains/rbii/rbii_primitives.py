# dreamcoder/domains/rbii/rbii_primitives.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.type import (
  arrow,
  t0,
  tbool,
  tcharacter,
  tint,
)

from .rbii_types import RBIIEvalState, trbii_state


# ---- Primitive implementations (Python values) ----

def _get_historical_obs(k: int):
  """
  Returns a function (state:rbii_state) -> char that reads the kth previous
  observation relative to state.timestep.

  Semantics:
    get_historical_obs(k)(state) = state.obs_at(state.timestep - 1 - k)

  So:
    k=0 -> "most recent observed char before time t"
    k=1 -> "char 2 steps back", etc.
  """

  def inner(state: RBIIEvalState) -> str:
    idx = (state.timestep - 1) - k
    return state.obs_at(idx)

  return inner



def _get_historical_program_abs(k: int):
  """
  Returns a function (state:rbii_state) -> char that delegates to the k-th
  stored "best" program by absolute index.

  Semantics:
    get_historical_program(k)(state) = state.program_at(k)(state)

  This keeps program references stable as the list grows.
  """

  def inner(state: RBIIEvalState) -> str:
    fn = state.program_at(k)
    return fn(state)

  return inner


def _succ_char(alphabet: Sequence[str]):
  succ: Dict[str, str] = {}
  for i, c in enumerate(alphabet):
    if i + 1 < len(alphabet):
      succ[c] = alphabet[i + 1]
    else:
      succ[c] = alphabet[i]  # clamp at end

  def f(c: str) -> str:
    if not (isinstance(c, str) and len(c) == 1):
      raise ValueError(f"expected char, got {c!r}")
    if c not in succ:
      # If outside the alphabet, just leave it unchanged for robustness.
      return c
    return succ[c]

  return f


def _eq_char(a: str):
  return lambda b: a == b


def _triple_eq(a: str):
  return lambda b: (lambda c: (a == b and b == c))


def _and(a: bool):
  return lambda b: bool(a and b)


def _if(c: bool):
  # polymorphic if: bool -> t0 -> t0 -> t0
  return lambda x: (lambda y: x if c else y)


# ---- Grammar builder ----

@dataclass(frozen=True)
class RBIIPrimitiveConfig:
  alphabet: str = "abcde"
  max_int: int = 6
  log_variable: float = 0.0


def _primitive(name: str, tp, value) -> Primitive:
  """
  Avoid creating duplicate Primitive names across multiple runs in the same
  interpreter session. Primitive.GLOBALS stores the canonical object.
  """
  if name in Primitive.GLOBALS:
    p = Primitive.GLOBALS[name]
    # We assume you're not trying to redefine primitives mid-run.
    return p
  return Primitive(name, tp, value)


def make_rbii_grammar(
    cfg: RBIIPrimitiveConfig = RBIIPrimitiveConfig()) -> Grammar:
  """
  Minimal grammar for predicting characters from history.

  Request type we target in the RBII tests:
    rbii_state -> char

  Core ideas:
    - get_historical_obs(k) returns (rbii_state -> char)
    - get_historical_program(k) returns (rbii_state -> char) by absolute id
    - triple_eq + if + succ_char allow easy discovery of run-length-3 patterns
    - get_historical_obs(1) solves simple alternation (ababab...)
    - get_historical_obs(0) solves constant sequences after warmup (aaaa...)
  """
  alphabet = list(cfg.alphabet)

  prims: List[Primitive] = []

  # int constants 0..max_int
  for i in range(cfg.max_int + 1):
    prims.append(_primitive(str(i), tint, i))

  # historical observation lookup: int -> rbii_state -> char
  prims.append(
    _primitive(
      "get_historical_obs",
      arrow(tint, trbii_state, tcharacter),
      _get_historical_obs,
    )
  )

  # stored program lookup (absolute): int -> rbii_state -> char
  prims.append(
    _primitive(
      "get_historical_program",
      arrow(tint, trbii_state, tcharacter),
      _get_historical_program_abs,
    )
  )

  # char logic
  prims.append(
    _primitive("eq_char", arrow(tcharacter, tcharacter, tbool), _eq_char))
  prims.append(
    _primitive("triple_eq", arrow(tcharacter, tcharacter, tcharacter, tbool),
               _triple_eq))
  prims.append(_primitive("and", arrow(tbool, tbool, tbool), _and))

  # successor (for aaabbbccc... pattern)
  prims.append(_primitive("succ_char", arrow(tcharacter, tcharacter),
                          _succ_char(alphabet)))

  # polymorphic if
  prims.append(_primitive("if", arrow(tbool, t0, t0, t0), _if))

  # Uniform prior over primitives; variable usage weight set by logVariable.
  g = Grammar.uniform(prims)
  g.logVariable = float(cfg.log_variable)
  return g
