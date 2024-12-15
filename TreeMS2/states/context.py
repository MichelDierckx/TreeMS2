from .state import State
from ..config.config import Config


class Context:
    """
    The Context defines the interface of interest to clients. It also maintains
    a reference to an instance of a State subclass, which represents the current
    state of the Context.
    """

    def __init__(self, config: Config) -> None:
        self.states = []
        self.config = config

    def get_state(self) -> State:
        assert self.states
        return self.states[-1]

    def pop_state(self) -> State:
        if self.states:
            return self.states.pop()

    def push_state(self, state: State):
        self.states.append(state)

    def replace_state(self, state: State):
        if self.states:
            self.states[-1] = state
