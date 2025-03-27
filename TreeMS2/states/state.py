from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from TreeMS2.states.state_type import StateType

if TYPE_CHECKING:
    from .context import Context  # Import only for type hints


class State(ABC):
    """
    The base State class declares methods that all Concrete States should
    implement and also provides a backreference to the Context object,
    associated with the State. This backreference can be used by States to
    transition the Context to another State.
    """
    STATE_TYPE = StateType

    def __init__(self, context: 'Context') -> None:
        self._context: 'Context' = context

    @property
    def context(self) -> 'Context':
        return self._context

    @context.setter
    def context(self, context: 'Context') -> None:
        self._context = context

    @abstractmethod
    def run(self) -> None:
        pass
