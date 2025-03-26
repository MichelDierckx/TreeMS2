import os

from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.distance_matrix import DistanceMatrix


class ComputeDistancesState(State):

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

    def _generate(self):
        similarity_sets = SimilaritySets(groups=self.context.groups, vector_store=None)
        # combine similarity sets across charges
        for s in self.context.similarity_sets.values():
            similarity_sets.similarity_sets += s.similarity_sets
        DistanceMatrix.create_mega(path=os.path.join(self.work_dir, "distance_matrix.meg"),
                                   similarity_threshold=self.similarity_threshold,
                                   precursor_mz_window=self.precursor_mz_window, similarity_sets=similarity_sets)
