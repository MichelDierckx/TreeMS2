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
        if not self.context.config.overwrite:
            if os.path.isfile(
                os.path.join(self.context.results_dir, "distance_matrix.meg")
            ):
                logger.info(
                    f"Found existing results ('{os.path.join(self.context.results_dir, "distance_matrix.meg")}'). Skipping."
                )
                self.context.pop_state()
                return
        self._generate()
        self.context.pop_state()

    def _generate(self):
        DistanceMatrix.create_mega(
            path=os.path.join(self.context.results_dir, "distance_matrix.meg"),
            similarity_threshold=self.similarity_threshold,
            precursor_mz_window=self.precursor_mz_window,
            similarity_sets=self.similarity_counts,
        )
        logger.info(
            f"Saved distance matrix to '{os.path.join(self.context.results_dir, 'distance_matrix.meg')}' as a MEGA file."
        )
