import os

from TreeMS2.config.logger_config import get_logger, log_section_title
from TreeMS2.distance_matrix_computation.distance_matrix import DistanceMatrix
from TreeMS2.search.similarity_counts import SimilarityCounts
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

logger = get_logger(__name__)


class DistanceMatrixComputationState(State):
    STATE_TYPE = StateType.DISTANCE_MATRIX_COMPUTATION_STATE

    def __init__(self, context: Context, similarity_counts: SimilarityCounts):
        super().__init__(context)

        self.query_results_dir: str = os.path.join(context.results_dir, "global")
        os.makedirs(self.query_results_dir, exist_ok=True)

        self.similarity_counts = similarity_counts

        # search parameters
        self.similarity_threshold: float = context.config.similarity

        # post-filtering
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self):
        log_section_title(logger=logger, title=f"[ Distance Matrix Computation ]")

        results_dir = self.context.results_dir
        meg_path = os.path.join(results_dir, "distance_matrix.meg")
        npy_path = os.path.join(results_dir, "distance_matrix.npy")
        labels_path = os.path.join(results_dir, "labels.txt")

        if not self.context.config.overwrite:
            if all(os.path.isfile(p) for p in [meg_path, npy_path, labels_path]):
                logger.info(
                    f"Found existing results ('{meg_path}', '{npy_path}', '{labels_path}'). Skipping."
                )
                self.context.pop_state()
                return
        self._generate()
        self.context.pop_state()

    def _generate(self):
        DistanceMatrix.export_mega(
            path=os.path.join(self.context.results_dir, "distance_matrix.meg"),
            similarity_threshold=self.similarity_threshold,
            precursor_mz_window=self.precursor_mz_window,
            similarity_sets=self.similarity_counts,
        )
        logger.info(
            f"Exported distance matrix to '{os.path.join(self.context.results_dir, 'distance_matrix.meg')}' as a MEGA file."
        )
        # Export .npy matrix and labels.txt
        DistanceMatrix.export_npy(
            output_npy_path=os.path.join(
                self.context.results_dir, "distance_matrix.npy"
            ),
            output_labels_path=os.path.join(self.context.results_dir, "labels.txt"),
            similarity_sets=self.similarity_counts,
        )
        logger.info(
            f"Exported distance matrix to '{os.path.join(self.context.results_dir, 'distance_matrix.npy')}' and labels to 'labels.txt'."
        )
