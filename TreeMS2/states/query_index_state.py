import os

import numpy as np

from TreeMS2.histogram import HitHistogram, SimilarityHistogram
from TreeMS2.index.vector_store_index import VectorStoreIndex
from TreeMS2.similarity_matrix.pipeline import SimilarityMatrixPipelineFactory
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.compute_distances_state import ComputeDistancesState
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType


class QueryIndexState(State):
    STATE_TYPE = StateType.QUERY_INDEX
    MAX_VECTORS_IN_MEM = 10_000

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
        if not self.context.config.overwrite:
            s = SimilaritySets.load(
                path=os.path.join(self.index.vector_store.directory,
                                  f"{self.index.vector_store.name}_similarity_sets.txt"),
                groups=self.context.groups, vector_store=self.index.vector_store)
            if s is not None:
                self.context.similarity_sets[self.index.vector_store.name] = s
                self.context.pop_state()
                return

        self.context.similarity_sets[self.index.vector_store.name] = self._generate()
        # if all indexes have been created and queried, transition to computing the distance matrix
        self.context.pop_state()
        if not self.context.contains_states([StateType.CREATE_INDEX, StateType.QUERY_INDEX]):
            self.context.hit_histogram_global.plot(
                path=os.path.join(self.work_dir,
                                  "hit_frequency_distribution.png"))
            self.context.similarity_histogram_global.plot(
                path=os.path.join(self.work_dir,
                                  "similarity_distribution.png"))
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
        for lims, d, i in self.index.range_search(similarity_threshold=self.similarity_threshold,
                                                  batch_size=QueryIndexState.MAX_VECTORS_IN_MEM):
            hit_histogram_local.update(lims=lims)
            similarity_histogram_local.update(d=d)
            self.context.hit_histogram_global.update(lims=lims)
            self.context.similarity_histogram_global.update(d=d)

            num_queries = len(lims) - 1
            row_indices = np.repeat(np.arange(num_queries, dtype=np.int64), np.diff(lims).astype(np.int64))
            col_indices = i.astype(np.int64)
            data = np.ones_like(i, dtype=np.bool_)

            similarity_matrix = SimilarityMatrix(self.context.groups.total_valid_spectra(),
                                                 similarity_threshold=self.similarity_threshold)
            similarity_matrix.update(data=data, rows=row_indices, cols=col_indices)

            # filter similarity matrix
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
        similarity_histogram_local.plot(
            path=os.path.join(self.index.vector_store.directory,
                              f"{self.index.vector_store.name}_similarity_distribution.png"))
        similarity_sets.write(
            path=os.path.join(self.index.vector_store.directory, f"{self.index.vector_store.name}_similarity_sets.txt"))
        return similarity_sets
