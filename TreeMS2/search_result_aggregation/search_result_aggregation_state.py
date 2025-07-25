import os

from TreeMS2.config.logger_config import get_logger, log_section_title
from TreeMS2.distance_matrix_computation.compute_distance_matrix_state import (
    DistanceMatrixComputationState,
)
from TreeMS2.search.similarity_counts import SimilarityCounts
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

logger = get_logger(__name__)


class SearchResultAggregationState(State):
    STATE_TYPE = StateType.SEARCH_RESULT_AGGREGATION_STATE

    def __init__(self, context: Context):
        super().__init__(context)

        self.query_results_dir: str = os.path.join(context.results_dir, "global")
        os.makedirs(self.query_results_dir, exist_ok=True)

    def run(self):
        log_section_title(logger=logger, title=f"[ Aggregating Search Results ]")
        if not self.context.config.overwrite:
            similarity_counts = SimilarityCounts.load(
                path=os.path.join(self.query_results_dir, "similarity_sets.txt"),
                groups=self.context.groups,
            )
            if similarity_counts:
                logger.info(
                    f"Found existing results ('{os.path.join(self.query_results_dir, "similarity_sets.txt")}'). Skipping."
                )
                self.context.replace_state(
                    state=DistanceMatrixComputationState(
                        context=self.context, similarity_counts=similarity_counts
                    )
                )
                return
        similarity_counts = self._generate()
        self.context.replace_state(
            state=DistanceMatrixComputationState(
                context=self.context, similarity_counts=similarity_counts
            )
        )

    def _generate(self) -> SimilarityCounts:
        if self.context.search_stats_global is not None:
            self.context.search_stats_global.export_hit_counts_to_histogram(
                path=os.path.join(
                    self.query_results_dir, "hit_frequency_distribution.png"
                )
            )
            logger.info(
                f"Saved histogram displaying distribution of spectra based on the number of similar spectra found to '{os.path.join(self.query_results_dir,
                                                                                                                                    "hit_frequency_distribution.png")}'."
            )
            self.context.search_stats_global.export_sim_counts_to_histogram(
                path=os.path.join(self.query_results_dir, "similarity_distribution.png")
            )
            logger.info(
                f"Saved histogram displaying the distribution of similar spectra pairs by similarity score to '{os.path.join(self.query_results_dir,
                                                                                                                             "similarity_distribution.png")}'."
            )

        similarity_sets = SimilarityCounts(groups=self.context.groups)
        # combine similarity sets across charges
        for s in self.context.similarity_sets.values():
            similarity_sets.merge(s)

        similarity_sets.write(
            path=os.path.join(self.query_results_dir, "similarity_sets.txt")
        )
        logger.info(
            f"Saved matrix showing the number of spectra in each group that have at least one similar spectrum in another group. File saved to '{os.path.join(self.query_results_dir, "similarity_sets.txt")}'."
        )
        return similarity_sets
