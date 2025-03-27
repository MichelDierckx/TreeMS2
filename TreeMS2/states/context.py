from typing import Optional, Dict, List

from .state import State
from .state_type import StateType
from ..config.config import Config
from ..groups.groups import Groups
from ..similarity_sets import SimilaritySets
from ..vector_store.vector_store_manager import VectorStoreManager


class Context:
    """
    The Context defines the interface of interest to clients. It also maintains
    a reference to an instance of a State subclass, which represents the current
    state of the Context.
    """

    def __init__(self, config: Config) -> None:
        self.states = []

        # shared data across states
        self.config = config
        self.groups: Optional[Groups] = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.similarity_sets: Dict[str, SimilaritySets] = {}

    def next(self):
        if self.states:
            self.states[-1].run()
        return

    def pop_state(self) -> State:
        if self.states:
            return self.states.pop()

    def push_state(self, state: State):
        self.states.append(state)

    def replace_state(self, state: State):
        if self.states:
            self.states[-1] = state

    def contains_states(self, state_types: List[StateType]) -> bool:
        """Check if any of the specified state types exist in the queue."""
        return any(state.state_type in state_types for state in self.states)
