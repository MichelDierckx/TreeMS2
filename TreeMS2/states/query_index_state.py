import os

from TreeMS2.groups.groups import Groups
from TreeMS2.index.ms2_index import MS2Index
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.states.analysis_state import AnalysisState
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.vector_store.vector_store import VectorStore

SIMILARITY_MATRIX = "similarity_matrix.npz"
SIMILARITY_MATRIX_GLOBAL = "similarity_matrix_global.npz"


class QueryIndexState(State):
    MAX_VECTORS_IN_MEM = 1_000

    def __init__(self, context: Context, groups: Groups, vector_store: VectorStore, index: MS2Index):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # data generated from reading/processing spectra
        self.groups: Groups = groups
        self.vector_store: VectorStore = vector_store

        # the index
        self.index = index

        # search parameters
        self.similarity_threshold: float = context.config.similarity

    def run(self, overwrite: bool):
        if overwrite or not self._is_output_generated():
            # generate the required output
            similarity_matrix = self._generate()
        else:
            # load the required output
            similarity_matrix = self._load()
        # move to the analysis state
        self.context.replace_state(
            state=AnalysisState(context=self.context, groups=self.groups, vector_store=self.vector_store,
                                similarity_matrix=similarity_matrix))

    def _generate(self) -> SimilarityMatrix:
        # create a similarity matrix
        similarity_matrix = self.index.range_search(similarity_threshold=self.similarity_threshold,
                                                    vector_store=self.vector_store,
                                                    batch_size=QueryIndexState.MAX_VECTORS_IN_MEM)
        # save similarity matrix to disk
        similarity_matrix.write(os.path.join(self.work_dir, SIMILARITY_MATRIX))
        similarity_matrix.write(os.path.join(self.work_dir, SIMILARITY_MATRIX_GLOBAL))
        return similarity_matrix

    def _load(self):
        # load the similarity matrix from disk
        similarity_matrix = SimilarityMatrix.load_with_threshold(path=os.path.join(self.work_dir, SIMILARITY_MATRIX),
                                                                 similarity_threshold=self.similarity_threshold)
        return similarity_matrix

    def _is_output_generated(self) -> bool:
        if not os.path.isfile(os.path.join(self.work_dir, SIMILARITY_MATRIX)):
            return False
        if not os.path.isfile(os.path.join(self.work_dir, SIMILARITY_MATRIX_GLOBAL)):
            return False
        return True
