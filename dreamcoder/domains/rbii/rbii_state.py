# dreamcoder/domains/rbii/rbii_state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from dreamcoder.program import Program


@dataclass
class RBIIState:
    """
    State = (obs_history, best_predictors_history).

    obs_history:
      - list of observed characters (Python str of len==1), indexed by time.

    best_programs:
      - list of DreamCoder Program objects that we have accepted as "best"
        predictors at some point in the run.

    compiled_programs:
      - cached Python callables corresponding to best_programs[i].evaluate([]).
        These should be functions of type (int -> char) for our test domain.
    """
    obs_history: List[str] = field(default_factory=list)
    best_programs: List[Program] = field(default_factory=list)
    compiled_programs: List[Callable[[int], str]] = field(default_factory=list)

    def observe(self, symbol: str) -> None:
        assert isinstance(symbol, str) and len(symbol) == 1, symbol
        self.obs_history.append(symbol)

    def time(self) -> int:
        """Current 'next index' to be predicted."""
        return len(self.obs_history)

    def add_best_program(self, program: Program) -> int:
        """
        Adds a program to best_programs and caches its evaluate([]) function.
        Returns the absolute program index used by get_historical_program(k).
        """
        fn = program.evaluate([])  # expects (int -> char) in this domain
        self.best_programs.append(program)
        self.compiled_programs.append(fn)
        return len(self.best_programs) - 1


# Global state handle used by primitives (simple + minimal for prototype).
_GLOBAL_STATE: Optional[RBIIState] = None


def set_global_state(state: RBIIState) -> None:
    global _GLOBAL_STATE
    _GLOBAL_STATE = state


def get_global_state() -> RBIIState:
    assert _GLOBAL_STATE is not None, (
        "RBII global state is not set. Call set_global_state(state) before "
        "evaluating programs that use RBII primitives."
    )
    return _GLOBAL_STATE