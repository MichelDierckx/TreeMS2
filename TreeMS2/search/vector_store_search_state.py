import os
from typing import Tuple

import numpy as np

from TreeMS2.config.logger_config import log_section_title, get_logger
from TreeMS2.indexing.vector_store_index import VectorStoreIndex
from TreeMS2.search.hit_exporter import HitExporter
from TreeMS2.search.post_processing.filters import (
    SimilarityThresholdFilter,
    PrecursorMzFilter,
)
from TreeMS2.search.post_processing.processor import SearchResultProcessor
from TreeMS2.search.search_stats import SearchStats
from TreeMS2.search.similarity_counts import SimilarityCounts, SimilarityCountsUpdater
from TreeMS2.search_result_aggregation.search_result_aggregation_state import (
    SearchResultAggregationState,
)
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

logger = get_logger(__name__)


class QueryIndexState(State):
    STATE_TYPE = StateType.SEARCH_STATE

    def __init__(self, context: Context, index: VectorStoreIndex):
        super().__init__(context)
        # query results dir
        self.query_results_dir: str = os.path.join(
            context.results_dir, "per_charge", index.vector_store.name
        )
        os.makedirs(self.query_results_dir, exist_ok=True)

        # the index
        self.index = index

        # search parameters
        self.batch_size: int = context.config.batch_size

        # Clip num_neighbours if necessary
        total_vectors = self.index.vector_count
        if context.config.num_neighbours > total_vectors:
            logger.debug(
                f"Requested num_neighbours ({context.config.num_neighbours}) exceeds total vectors in index ({total_vectors}). Clipping to {total_vectors}."
            )
            self.num_neighbours = total_vectors
        else:
            self.num_neighbours: int = context.config.num_neighbours

        # Clip num_probe if necessary
        nlist = self.index.nlist
        if nlist == 0:
            logger.debug("Index is brute-force (nlist=0), setting num_probe to 1.")
            self.num_probe = 1
        elif context.config.num_probe > nlist:
            logger.debug(
                f"Requested num_probe ({context.config.num_probe}) exceeds number of clusters (nlist={nlist}). Clipping to {nlist}."
            )
            self.num_probe = nlist
        else:
            self.num_probe = context.config.num_probe

        # post-filtering
        self.similarity_threshold: float = context.config.similarity
        self.precursor_mz_window: float = context.config.precursor_mz_window

    def run(self):
        log_section_title(
            logger=logger,
            title=f"[ Searching Similarities ({self.index.vector_store.name}) ]",
        )
        if not self.context.config.overwrite:
            s = SimilarityCounts.load(
                path=os.path.join(
                    self.query_results_dir,
                    f"{self.index.vector_store.name}_similarity_sets.txt",
                ),
                groups=self.context.groups,
            )
            if s is not None:
                logger.info(
                    f"Found existing results ('{os.path.join(self.query_results_dir,
                                                             f"{self.index.vector_store.name}_similarity_sets.txt")}'). Skipping processing and loading results from disk."
                )
                self.context.similarity_sets[self.index.vector_store.name] = s
                self._transition()
                return

        self.context.similarity_sets[self.index.vector_store.name] = self._generate()
        self._transition()

    def _transition(self):
        # if all indexes have been created and queried, transition to computing the distance matrix
        self.context.pop_state()
        if not self.context.contains_states(
            [StateType.INDEXING_STATE, StateType.SEARCH_STATE]
        ):
            self.context.push_state(SearchResultAggregationState(self.context))

    def _setup_search_result_processor(self) -> SearchResultProcessor:
        similarity_threshold_filter = SimilarityThresholdFilter(
            self.similarity_threshold
        )
        precursor_mz_filter = None
        if self.precursor_mz_window is not None:
            precursor_mzs = (
                self.index.vector_store.get_col("precursor_mz")
                .to_numpy(dtype=np.float32)
                .ravel()
            )
            precursor_mz_filter = PrecursorMzFilter(
                precursor_mzs=precursor_mzs, mz_window=self.precursor_mz_window
            )

        group_ids = (
            self.index.vector_store.get_col("group_id")
            .to_numpy(dtype=np.uint16)
            .ravel()
        )
        vector_store_similarity_counts_updater = SimilarityCountsUpdater(group_ids)

        file_ids = (
            self.index.vector_store.get_col("file_id").to_numpy(dtype=np.uint16).ravel()
        )
        scan_numbers = (
            self.index.vector_store.get_col("scan_number")
            .to_numpy(dtype=np.uint32)
            .ravel()
        )

        hit_exporter = HitExporter(
            output_file_path=os.path.join(self.context.results_dir, "hits.tsv"),
            file_ids=file_ids,
            group_ids=group_ids,
            scan_numbers=scan_numbers,
        )

        return SearchResultProcessor(
            similarity_threshold_filter=similarity_threshold_filter,
            precursor_mz_filter=precursor_mz_filter,
            vector_store_similarity_counts_updater=vector_store_similarity_counts_updater,
            hit_exporter=hit_exporter,
        )

    def _setup_search_stats(self) -> Tuple[SearchStats, SearchStats]:

        def construct_bin_edges(max_hits):
            """Constructs bin edges up to the nearest power of 10 above max_hits, ensuring the first bin is [1, 2)."""
            bins = [1, 2]  # ensure the first bin is [1, 2)
            last_bin = 2
            power = 1
            while last_bin < max_hits:
                next_bin = 10**power
                bins.append(next_bin)
                last_bin = next_bin
                power += 1
            return np.array(bins)

        search_stats_vector_store = SearchStats(
            hit_bin_edges=construct_bin_edges(self.index.vector_count),
            sim_bin_edges=np.linspace(self.similarity_threshold, 1, 21),
        )

        if self.context.search_stats_global is None:
            search_stats_vector_stores = SearchStats(
                hit_bin_edges=construct_bin_edges(
                    self.context.groups.get_stats()[1].high_quality
                ),
                sim_bin_edges=np.linspace(self.similarity_threshold, 1, 21),
            )
            self.context.search_stats_global = search_stats_vector_stores
        return search_stats_vector_store, self.context.search_stats_global

    def _export_search_results(
        self,
        search_stats_vector_store: SearchStats,
        vector_store_similarity_counts: SimilarityCounts,
    ):
        search_stats_vector_store.export_hit_counts_to_histogram(
            path=os.path.join(
                self.query_results_dir,
                f"{self.index.vector_store.name}_hit_frequency_distribution.png",
            )
        )

        logger.info(
            f"Saved histogram displaying distribution of spectra based on the number of similar spectra found to '{os.path.join(self.query_results_dir,
                                                                                                                                f"{self.index.vector_store.name}_hit_frequency_distribution.png")}'."
        )

        search_stats_vector_store.export_sim_counts_to_histogram(
            path=os.path.join(
                self.query_results_dir,
                f"{self.index.vector_store.name}_similarity_distribution.png",
            )
        )
        logger.info(
            f"Saved histogram displaying the distribution of similar spectra pairs by similarity score to '{os.path.join(self.query_results_dir,
                                                                                                                         f"{self.index.vector_store.name}_similarity_distribution.png")}'."
        )

        vector_store_similarity_counts.write(
            path=os.path.join(
                self.query_results_dir,
                f"{self.index.vector_store.name}_similarity_sets.txt",
            )
        )

        logger.info(
            f"Saved matrix showing the number of spectra in each group that have at least one similar spectrum in another group. File saved to '{os.path.join(self.query_results_dir, f"{self.index.vector_store.name}_similarity_sets.txt")}'."
        )

    def _generate(self) -> SimilarityCounts:

        # init similarity sets
        vector_store_similarity_counts = SimilarityCounts(groups=self.context.groups)

        search_result_processor = self._setup_search_result_processor()
        search_stats_list = list(self._setup_search_stats())

        # query index
        batch_nr = 0
        for d, i, query_ids in self.index.knn_search(
            k=self.num_neighbours, nprobe=self.num_probe, batch_size=self.batch_size
        ):
            vector_store_similarity_counts, search_stats_list = (
                search_result_processor.process(
                    d=d,
                    i=i,
                    query_ids=query_ids,
                    vector_store_similarity_counts=vector_store_similarity_counts,
                    search_stats=search_stats_list,
                )
            )

            # update batch nr
            batch_nr += 1

        self._export_search_results(
            search_stats_vector_store=search_stats_list[0],
            vector_store_similarity_counts=vector_store_similarity_counts,
        )
        self.context.search_stats_global = search_stats_list[1]

        return vector_store_similarity_counts
