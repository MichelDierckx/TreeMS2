import os
from typing import Optional, Dict, List

from TreeMS2.config.treems2_config import Config
from TreeMS2.ingestion.spectra_sets.spectra_sets import SpectraSets
from TreeMS2.ingestion.storage.vector_stores import VectorStores
from TreeMS2.search.search_stats import SearchStats
from TreeMS2.search.similarity_counts import SimilarityCounts
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

RESULTS_DIR_NAME = "results"
LANCE_DIR_NAME = "lance"
INDEXES_DIR_NAME = "indexes"


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
        self.groups: Optional[SpectraSets] = None
        self.vector_store_manager: Optional[VectorStores] = None

        self.similarity_sets: Dict[str, SimilarityCounts] = {}
        self.search_stats_global: Optional[SearchStats] = None

        self.results_dir = os.path.join(self.config.work_dir, RESULTS_DIR_NAME)
        self.lance_dir = os.path.join(self.config.work_dir, LANCE_DIR_NAME)
        self.indexes_dir = os.path.join(self.config.work_dir, INDEXES_DIR_NAME)

        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.lance_dir, exist_ok=True)
        os.makedirs(self.indexes_dir, exist_ok=True)

    def next(self):
        if self.states:
            self.states[-1].run()
        return

    def pop_state(self) -> Optional[State]:
        if self.states:
            return self.states.pop()
        return None

    def push_state(self, state: State):
        self.states.append(state)

    def replace_state(self, state: State):
        if self.states:
            self.states[-1] = state

    def contains_states(self, state_types: List[StateType]) -> bool:
        """Check if any of the specified state types exist in the queue."""
        return any(state.STATE_TYPE in state_types for state in self.states)
