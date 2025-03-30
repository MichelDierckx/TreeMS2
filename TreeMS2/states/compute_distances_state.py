import os

from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.distance_matrix import DistanceMatrix
from TreeMS2.states.state_type import StateType


class ComputeDistancesState(State):
    STATE_TYPE = StateType.COMPUTE_DISTANCES

    def __init__(self, context: Context):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # search parameters
        self.similarity_threshold: float = context.config.similarity

        # post-filtering
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self):
        if not self.context.config.overwrite:
            if os.path.isfile(os.path.join(self.work_dir, "distance_matrix.meg")):
                return
        self._generate()
        self.context.pop_state()

    def _generate(self):
        similarity_sets = SimilaritySets(groups=self.context.groups, vector_store=None)
        # combine similarity sets across charges
        for s in self.context.similarity_sets.values():
            similarity_sets.similarity_sets += s.similarity_sets

        # TODO: remove lazy fix when diagonal elements of similarity sets contain correct number
        for row_group in self.context.groups.get_groups():
            row_group_id = row_group.get_id()
            for col_group in self.context.groups.get_groups():
                col_group_id = col_group.get_id()
                if row_group_id == col_group_id:
                    similarity_sets.similarity_sets[row_group_id, col_group_id] = row_group.total_spectra

        similarity_sets.write(path=os.path.join(self.work_dir, "similarity_sets.txt"))
        DistanceMatrix.create_mega(path=os.path.join(self.work_dir, "distance_matrix.meg"),
                                   similarity_threshold=self.similarity_threshold,
                                   precursor_mz_window=self.precursor_mz_window, similarity_sets=similarity_sets)
