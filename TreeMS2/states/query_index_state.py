import os

import numpy as np

from TreeMS2.histogram import HitHistogram, SimilarityHistogram
from TreeMS2.index.vector_store_index import VectorStoreIndex
from TreeMS2.logger_config import log_section_title, get_logger
from TreeMS2.similarity_matrix.pipeline import SimilarityMatrixPipelineFactory
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.compute_distances_state import ComputeDistancesState
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

logger = get_logger(__name__)


class QueryIndexState(State):
    STATE_TYPE = StateType.QUERY_INDEX
    BATCH_SIZE = 10_000
    NUMBER_OF_NEIGHBOURS = 1024
    NPROBE = 32  # 128

    def __init__(self, context: Context, index: VectorStoreIndex):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # the index
        self.index = index

        # search parameters
        self.similarity_threshold: float = context.config.similarity

        # post-filtering
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self):
        log_section_title(logger=logger, title=f"[ Searching Similarities ({self.index.vector_store.name}) ]")
        if not self.context.config.overwrite:
            s = SimilaritySets.load(
                path=os.path.join(self.index.vector_store.directory,
                                  f"{self.index.vector_store.name}_similarity_sets.txt"),
                groups=self.context.groups, vector_store=self.index.vector_store)
            if s is not None:
                logger.info(
                    f"Found existing results ('{os.path.join(self.index.vector_store.directory,
                                                             f"{self.index.vector_store.name}_similarity_sets.txt")}'). Skipping processing and loading results from disk.")
                self.context.similarity_sets[self.index.vector_store.name] = s
                self._transition()
                return

        self.context.similarity_sets[self.index.vector_store.name] = self._generate()
        self._transition()

    def _transition(self):
        # if all indexes have been created and queried, transition to computing the distance matrix
        self.context.pop_state()
        if not self.context.contains_states([StateType.CREATE_INDEX, StateType.QUERY_INDEX]):
            self.context.push_state(ComputeDistancesState(self.context))

    def _generate(self) -> SimilaritySets:

        # init similarity sets
        similarity_sets = SimilaritySets(groups=self.context.groups,
                                         vector_store=self.index.vector_store)
        # init filtering pipeline
        pipeline = SimilarityMatrixPipelineFactory.create_pipeline(groups=self.context.groups,
                                                                   vector_store=self.index.vector_store,
                                                                   precursor_mz_window=self.precursor_mz_window)

        def construct_bin_edges(max_hits):
            """Constructs bin edges up to the nearest power of 10 above max_hits, ensuring the first bin is [1, 2)."""
            bins = [1, 2]  # ensure the first bin is [1, 2)
            last_bin = 2
            power = 1
            while last_bin < max_hits:
                next_bin = 10 ** power
                bins.append(next_bin)
                last_bin = next_bin
                power += 1
            return np.array(bins)

        hit_histogram_local = HitHistogram(bin_edges=construct_bin_edges(self.index.vector_store.vector_count))
        similarity_histogram_local = SimilarityHistogram(bin_edges=np.linspace(self.similarity_threshold, 1, 21))
        if self.context.similarity_histogram_global is None:
            self.context.similarity_histogram_global = SimilarityHistogram(
                bin_edges=np.linspace(self.similarity_threshold, 1, 21))
        if self.context.hit_histogram_global is None:
            self.context.hit_histogram_global = HitHistogram(
                bin_edges=construct_bin_edges(self.context.groups.total_valid_spectra()))

        # query index
        batch_nr = 0
        for d, i, query_ids in self.index.knn_search(k=QueryIndexState.NUMBER_OF_NEIGHBOURS,
                                                     nprobe=QueryIndexState.NPROBE,
                                                     batch_size=QueryIndexState.BATCH_SIZE):
            # compute a mask (similarity threshold)
            mask = d >= self.similarity_threshold
            # post filter distances using mask
            filtered_distances = d[mask]

            # update histograms
            hits_per_query = mask.sum(axis=1)  # count retained hits per query
            hit_histogram_local.update(hits_per_query=hits_per_query)
            similarity_histogram_local.update(d=filtered_distances)
            self.context.hit_histogram_global.update(hits_per_query=hits_per_query)
            self.context.similarity_histogram_global.update(d=filtered_distances)

            filtered_indices = i[mask]  # flatten and filter indices
            flat_mask = mask.flatten()  # flatten mask
            row_indices = np.repeat(query_ids, i.shape[1])[flat_mask]  # repeat query id for each neighbour find
            data = np.ones_like(filtered_indices, dtype=bool)  # store 1's for a hit
            similarity_matrix = SimilarityMatrix(self.context.groups.total_spectra,
                                                 similarity_threshold=self.similarity_threshold)
            similarity_matrix.update(data=data, rows=row_indices, cols=filtered_indices)

            # post filter (precursor mz window)
            similarity_matrix = pipeline.process(similarity_matrix=similarity_matrix,
                                                 total_spectra=self.context.groups.total_spectra,
                                                 target_dir=os.path.join(self.index.vector_store.directory,
                                                                         "filters",
                                                                         f"{batch_nr}_filter.txt"))

            # update similarity sets
            similarity_sets.update_similarity_sets(similarity_matrix=similarity_matrix)

            # update batch nr
            batch_nr += 1

        hit_histogram_local.plot(
            path=os.path.join(self.index.vector_store.directory,
                              f"{self.index.vector_store.name}_hit_frequency_distribution.png"))

        logger.info(
            f"Saved histogram displaying distribution of spectra based on the number of similar spectra found to '{os.path.join(self.index.vector_store.directory,
                                                                                                                                f"{self.index.vector_store.name}_hit_frequency_distribution.png")}'.")

        similarity_histogram_local.plot(
            path=os.path.join(self.index.vector_store.directory,
                              f"{self.index.vector_store.name}_similarity_distribution.png"))
        logger.info(
            f"Saved histogram displaying the distribution of similar spectra pairs by similarity score to '{os.path.join(self.index.vector_store.directory,
                                                                                                                         f"{self.index.vector_store.name}_similarity_distribution.png")}'.")

        similarity_sets.write(
            path=os.path.join(self.index.vector_store.directory, f"{self.index.vector_store.name}_similarity_sets.txt"))

        logger.info(
            f"Saved matrix showing the number of spectra in each group that have at least one similar spectrum in another group. File saved to '{os.path.join(self.index.vector_store.directory, f"{self.index.vector_store.name}_similarity_sets.txt")}'.")

        return similarity_sets
