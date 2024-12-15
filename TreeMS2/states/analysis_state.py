import os

from TreeMS2.distances import Distances
from TreeMS2.groups.groups import Groups
from TreeMS2.similarity_matrix.pipeline import SimilarityMatrixPipelineFactory
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.vector_store.vector_store import VectorStore

# before post-processing
ANALYSIS_DIR = "analysis"
SIMILARITY_STATISTICS = "analysis/similarity_statistics.txt"
DISTANCES = "analysis/distances.meg"

# after post-processing
SIMILARITY_MATRIX_POST = "analysis/similarity_matrix_post.npz"
SIMILARITY_MATRIX_POST_GLOBAL = "analysis/similarity_matrix_post_global.npz"
SIMILARITY_STATISTICS_POST = "analysis/similarity_statistics_post.txt"
DISTANCES_POST = "analysis/distances_post.meg"


class AnalysisState(State):
    def __init__(self, context: Context, groups: Groups, vector_store: VectorStore,
                 similarity_matrix: SimilarityMatrix):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # data generated from reading/processing spectra
        self.groups: Groups = groups
        self.vector_store: VectorStore = vector_store

        # similarity matrix
        self.similarity_matrix: SimilarityMatrix = similarity_matrix

        # post-processing
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self, overwrite: bool):
        if overwrite or not self._is_output_generated():
            self._generate()
        # this is the last program state, pop it
        self.context.pop_state()

    def _generate(self):
        # create analysis directory if it does not exist
        os.makedirs(os.path.join(self.work_dir, ANALYSIS_DIR), exist_ok=True)

        # Compute similarity sets
        similarity_sets = SimilaritySets(similarity_matrix=self.similarity_matrix, groups=self.groups,
                                         vector_store=self.vector_store)
        # Write similarity sets to file
        similarity_sets.write(path=os.path.join(self.work_dir, SIMILARITY_STATISTICS))
        # Compute distances
        distances = Distances(similarity_sets=similarity_sets)
        distances.create_mega(path=os.path.join(self.work_dir, DISTANCES))
        # Filter similarity matrix
        pipeline = SimilarityMatrixPipelineFactory.create_pipeline(groups=self.groups, vector_store=self.vector_store,
                                                                   precursor_mz_window=self.precursor_mz_window)
        similarity_matrix = pipeline.process(similarity_matrix=self.similarity_matrix,
                                             total_spectra=self.groups.total_spectra,
                                             target_dir=os.path.join(self.work_dir, ANALYSIS_DIR))

        # Write similarity matrix to file after filtering
        similarity_matrix.write(path=os.path.join(self.work_dir, SIMILARITY_MATRIX_POST))
        similarity_matrix.write_global(path=os.path.join(self.work_dir, SIMILARITY_MATRIX_POST_GLOBAL),
                                       total_spectra=self.groups.total_spectra, vector_store=self.vector_store)
        # Compute similarity sets
        similarity_sets = SimilaritySets(similarity_matrix=similarity_matrix, groups=self.groups,
                                         vector_store=self.vector_store)
        # Write similarity sets to file
        similarity_sets.write(path=os.path.join(self.work_dir, SIMILARITY_STATISTICS_POST))
        # Compute distances
        distances = Distances(similarity_sets=similarity_sets)
        distances.create_mega(path=os.path.join(self.work_dir, DISTANCES_POST))

    def _is_output_generated(self) -> bool:
        # Define the list of required file paths
        required_files = [
            SIMILARITY_STATISTICS,
            DISTANCES,
            SIMILARITY_MATRIX_POST,
            SIMILARITY_MATRIX_POST_GLOBAL,
            DISTANCES_POST,
            SIMILARITY_STATISTICS_POST,
        ]
        # Check if all required files exist
        for file in required_files:
            if not os.path.isfile(os.path.join(self.work_dir, file)):
                return False
        return True
