from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from dreamcoder.type import baseType


# Explicit task input type for RBII evaluation state.
trbii_state = baseType("rbii_state")


@runtime_checkable
class RBIIEvalState(Protocol):
    """
    Interface consumed by RBII primitives.

    This keeps primitive implementations decoupled from concrete state
    representations and ready for future alternatives (e.g., version spaces).
    """

    @property
    def timestep(self) -> int:
        ...

    def obs_at(self, idx: int) -> str:
        ...

    def program_at(self, k: int) -> Callable[["RBIIEvalState"], str]:
        ...
