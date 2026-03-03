from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from dreamcoder.program import Program

from .rbii_types import RBIIEvalState


@dataclass(frozen=True)
class RBIIStateView:
    """
    Immutable causal view used as a task input.

    - timestep: prediction index this view represents.
    - obs_cutoff: only observations [0, obs_cutoff) are visible.
    - program_cutoff: only programs [0, program_cutoff) are visible.
    """

    _state: "RBIIState"
    timestep: int
    obs_cutoff: int
    program_cutoff: int

    def obs_at(self, idx: int) -> str:
        if idx < 0 or idx >= self.obs_cutoff:
            raise IndexError(
                f"obs idx out of range for view: t={self.timestep} idx={idx} "
                f"obs_cutoff={self.obs_cutoff}"
            )
        return self._state.obs_history[idx]

    def program_at(self, k: int) -> Callable[[RBIIEvalState], str]:
        if k < 0 or k >= self.program_cutoff:
            raise IndexError(
                f"program idx out of range for view: t={self.timestep} k={k} "
                f"program_cutoff={self.program_cutoff}"
            )
        return self._state.compiled_programs[k]


@dataclass
class RBIIState:
    """
    Mutable RBII runtime state.

    obs_history:
      - list of observed characters (str of len==1), indexed by time.

    best_programs / compiled_programs:
      - accepted predictor programs and compiled callables of type
        (rbii_state -> char).

    program_birth_timestep:
      - program_birth_timestep[k] is the timestep index observed just before
        program k was added. Program k becomes visible for timestep > birth.
    """

    obs_history: List[str] = field(default_factory=list)
    best_programs: List[Program] = field(default_factory=list)
    compiled_programs: List[Callable[[RBIIEvalState], str]] = field(
        default_factory=list
    )
    program_birth_timestep: List[int] = field(default_factory=list)

    def __getstate__(self):
        """
        Make state picklable for multiprocessing transport.
        Compiled callables are runtime cache only and are not serializable.
        """
        return {
            "obs_history": list(self.obs_history),
            "best_programs": list(self.best_programs),
            "program_birth_timestep": list(self.program_birth_timestep),
        }

    def __setstate__(self, state):
        self.obs_history = list(state.get("obs_history", []))
        self.best_programs = list(state.get("best_programs", []))
        self.program_birth_timestep = list(state.get("program_birth_timestep", []))
        # Rebuild runtime cache of compiled callables.
        self.compiled_programs = [p.evaluate([]) for p in self.best_programs]

    def observe(self, symbol: str) -> None:
        assert isinstance(symbol, str) and len(symbol) == 1, symbol
        self.obs_history.append(symbol)

    def time(self) -> int:
        """Current 'next index' to be predicted."""
        return len(self.obs_history)

    def add_best_program(self, program: Program, birth_timestep: int) -> int:
        """
        Adds a program and caches its compiled callable.
        Returns the absolute program index.
        """
        fn = program.evaluate([])  # expects (rbii_state -> char) in this domain
        self.best_programs.append(program)
        self.compiled_programs.append(fn)
        self.program_birth_timestep.append(int(birth_timestep))
        return len(self.best_programs) - 1

    def _program_cutoff_for_timestep(self, timestep: int) -> int:
        """
        Number of stored programs visible when predicting at `timestep`.
        Programs born at timestep t are only visible for timesteps > t.
        """
        return sum(1 for born in self.program_birth_timestep if born < timestep)

    def view_for_timestep(self, timestep: int) -> RBIIStateView:
        """
        Return a causal view for predicting obs[timestep].
        """
        if timestep < 0 or timestep > len(self.obs_history):
            raise IndexError(
                f"invalid timestep for view: t={timestep} len={len(self.obs_history)}"
            )
        return RBIIStateView(
            _state=self,
            timestep=timestep,
            obs_cutoff=timestep,
            program_cutoff=self._program_cutoff_for_timestep(timestep),
        )
