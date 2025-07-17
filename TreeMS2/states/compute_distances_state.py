import os

from TreeMS2.distance_matrix import DistanceMatrix
from TreeMS2.logger_config import get_logger, log_section_title
from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

logger = get_logger(__name__)


class ComputeDistancesState(State):
    STATE_TYPE = StateType.COMPUTE_DISTANCES

    def __init__(self, context: Context):
        super().__init__(context)

        self.query_results_dir: str = os.path.join(context.results_dir, "global")
        os.makedirs(self.query_results_dir, exist_ok=True)

        # search parameters
        self.similarity_threshold: float = context.config.similarity

        # post-filtering
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self):
        log_section_title(logger=logger, title=f"[ Aggregating Search Results ]")
        if not self.context.config.overwrite:
            if os.path.isfile(os.path.join(self.context.results_dir, "distance_matrix.meg")) and SimilaritySets.load(
                    path=os.path.join(self.query_results_dir, "similarity_sets.txt"),
                    groups=self.context.groups):
                logger.info(
                    f"Found existing results ('{os.path.join(self.context.results_dir, "distance_matrix.meg")}', '{os.path.join(self.query_results_dir, "similarity_sets.txt")}'). Skipping.")
                self.context.pop_state()
                return
        self._generate()
        self.context.pop_state()

    def _generate(self):
        if self.context.hit_histogram_global is not None:
            self.context.hit_histogram_global.plot(
                path=os.path.join(self.query_results_dir,
                                  "hit_frequency_distribution.png"))
            logger.info(
                f"Saved histogram displaying distribution of spectra based on the number of similar spectra found to '{os.path.join(self.query_results_dir,
                                                                                                                                    "hit_frequency_distribution.png")}'.")
        if self.context.similarity_histogram_global is not None:
            self.context.similarity_histogram_global.plot(
                path=os.path.join(self.query_results_dir,
                                  "similarity_distribution.png"))
            logger.info(
                f"Saved histogram displaying the distribution of similar spectra pairs by similarity score to '{os.path.join(self.query_results_dir,
                                                                                                                             "similarity_distribution.png")}'.")

        similarity_sets = SimilaritySets(groups=self.context.groups)
        # combine similarity sets across charges
        for s in self.context.similarity_sets.values():
            similarity_sets.similarity_sets += s.similarity_sets

        similarity_sets.write(path=os.path.join(self.query_results_dir, "similarity_sets.txt"))
        logger.info(
            f"Saved matrix showing the number of spectra in each group that have at least one similar spectrum in another group. File saved to '{os.path.join(self.query_results_dir, "similarity_sets.txt")}'.")

        DistanceMatrix.create_mega(path=os.path.join(self.context.results_dir, "distance_matrix.meg"),
                                   similarity_threshold=self.similarity_threshold,
                                   precursor_mz_window=self.precursor_mz_window, similarity_sets=similarity_sets)
        logger.info(
            f"Saved distance matrix to '{os.path.join(self.context.results_dir, 'distance_matrix.meg')}' as a MEGA file.")
