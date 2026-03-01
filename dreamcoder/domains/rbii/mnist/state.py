from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import torch

from dreamcoder.program import Program

from .types import MNISTEvalState, MNISTPrediction


@dataclass(frozen=True)
class MNISTStateView:
    """
    Immutable causal view for prediction at a specific timestep.

    `obs_cutoff=timestep` means label/x lookups are only valid for indices < timestep.
    The current input x_t is provided separately as `current_x_tensor`.
    """

    _state: "MNISTState"
    timestep: int
    obs_cutoff: int
    program_cutoff: int
    current_x_tensor: torch.Tensor
    current_context_id: str

    def current_x(self) -> torch.Tensor:
        return self.current_x_tensor

    def context_id(self) -> str:
        return self.current_context_id

    def x_at(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.obs_cutoff:
            raise IndexError(
                f"x idx out of range for view: t={self.timestep} idx={idx} cutoff={self.obs_cutoff}"
            )
        return self._state.x_history[idx]

    def label_at(self, idx: int) -> int:
        if idx < 0 or idx >= self.obs_cutoff:
            raise IndexError(
                f"label idx out of range for view: t={self.timestep} idx={idx} cutoff={self.obs_cutoff}"
            )
        return self._state.y_history[idx]

    def program_at(self, k: int) -> Callable[[MNISTEvalState], MNISTPrediction]:
        if k < 0 or k >= self.program_cutoff:
            raise IndexError(
                f"program idx out of range for view: t={self.timestep} k={k} cutoff={self.program_cutoff}"
            )
        return self._state.compiled_programs[k]


@dataclass
class MNISTState:
    x_history: List[torch.Tensor] = field(default_factory=list)
    y_history: List[int] = field(default_factory=list)
    context_history: List[str] = field(default_factory=list)

    best_programs: List[Program] = field(default_factory=list)
    compiled_programs: List[Callable[[MNISTEvalState], MNISTPrediction]] = field(default_factory=list)
    program_birth_timestep: List[int] = field(default_factory=list)

    def observe(self, x: torch.Tensor, y: int, context: str) -> None:
        x_cpu = x.detach().float().view(-1).cpu()
        self.x_history.append(x_cpu)
        self.y_history.append(int(y))
        self.context_history.append(str(context))

    def time(self) -> int:
        return len(self.y_history)

    def add_best_program(self, program: Program, birth_timestep: int) -> int:
        fn = program.evaluate([])
        self.best_programs.append(program)
        self.compiled_programs.append(fn)
        self.program_birth_timestep.append(int(birth_timestep))
        return len(self.best_programs) - 1

    def _program_cutoff_for_timestep(self, timestep: int) -> int:
        return sum(1 for born in self.program_birth_timestep if born < timestep)

    def view_for_prediction(self, current_x: torch.Tensor, current_context: str) -> MNISTStateView:
        t = self.time()
        return MNISTStateView(
            _state=self,
            timestep=t,
            obs_cutoff=t,
            program_cutoff=self._program_cutoff_for_timestep(t),
            current_x_tensor=current_x.detach().float().view(-1).cpu(),
            current_context_id=str(current_context),
        )

    def view_for_history_index(self, idx: int) -> MNISTStateView:
        if idx < 0 or idx >= self.time():
            raise IndexError(f"history index out of range: idx={idx}, time={self.time()}")
        return MNISTStateView(
            _state=self,
            timestep=idx,
            obs_cutoff=idx,
            program_cutoff=self._program_cutoff_for_timestep(idx),
            current_x_tensor=self.x_history[idx],
            current_context_id=self.context_history[idx],
        )
