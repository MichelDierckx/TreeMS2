import os

import numpy as np

from TreeMS2.histogram import HitHistogram, SimilarityHistogram
from TreeMS2.index.ms2_index import MS2Index
from TreeMS2.similarity_matrix.pipeline import SimilarityMatrixPipelineFactory
from TreeMS2.similarity_sets import SimilaritySets
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.vector_store.vector_store import VectorStore


class QueryIndexState(State):
    MAX_VECTORS_IN_MEM = 10_000

    def __init__(self, context: Context, vector_store: VectorStore, index: MS2Index):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # data generated from reading/processing spectra
        self.vector_store: VectorStore = vector_store

        # the index
        self.index = index

        # search parameters
        self.similarity_threshold: float = context.config.similarity

        # post-filtering
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self):
        if not self.context.config.overwrite:
            s = SimilaritySets.load(
                path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}_similarity_sets.txt"),
                groups=self.context.groups, vector_store=self.vector_store)
            if s is not None:
                self.context.similarity_sets[self.vector_store.name] = s
                self.context.pop_state()
        self.context.similarity_sets[self.vector_store.name] = self._generate()
        self.context.pop_state()

    def _generate(self) -> SimilaritySets:

        # init similarity sets
        similarity_sets = SimilaritySets(groups=self.context.groups,
                                         vector_store=self.vector_store)
        # init filtering pipeline
        pipeline = SimilarityMatrixPipelineFactory.create_pipeline(groups=self.context.groups,
                                                                   vector_store=self.vector_store,
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

        hit_histogram = HitHistogram(bin_edges=construct_bin_edges(self.context.groups.total_valid_spectra()))
        similarity_histogram = SimilarityHistogram(bin_edges=np.linspace(self.similarity_threshold, 1, 21))

        # query index
        batch_nr = 0
        for similarity_matrix in self.index.range_search(similarity_threshold=self.similarity_threshold,
                                                         vector_store=self.vector_store,
                                                         batch_size=QueryIndexState.MAX_VECTORS_IN_MEM,
                                                         hit_histogram=hit_histogram,
                                                         similarity_histogram=similarity_histogram):
            # filter similarity matrix
            similarity_matrix = pipeline.process(similarity_matrix=similarity_matrix,
                                                 total_spectra=self.context.groups.total_spectra,
                                                 target_dir=os.path.join(self.vector_store.directory,
                                                                         "filters",
                                                                         f"{batch_nr}_filter.txt"))

            # update similarity sets
            similarity_sets.update_similarity_sets(similarity_matrix=similarity_matrix)

            # update batch nr
            batch_nr += 1

        hit_histogram.plot(
            path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}_hit_frequency_distribution.png"))
        similarity_histogram.plot(
            path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}_similarity_distribution.png"))
        similarity_sets.write(
            path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}_similarity_sets.txt"))
        return similarity_sets
