from typing import Tuple, List, Optional

import numpy as np
import numpy.typing as npt

from TreeMS2.search.post_processing.filters import (
    SimilarityThresholdFilter,
    PrecursorMzFilter,
)
from TreeMS2.search.search_stats import SearchStats
from TreeMS2.search.similarity_counts import SimilarityCountsUpdater, SimilarityCounts


class SearchResultProcessor:
    def __init__(
        self,
        similarity_threshold_filter: SimilarityThresholdFilter,
        precursor_mz_filter: Optional[PrecursorMzFilter],
        vector_store_similarity_counts_updater: SimilarityCountsUpdater,
    ):
        self.similarity_threshold_filter = similarity_threshold_filter
        self.precursor_mz_filter = precursor_mz_filter
        self.vector_store_similarity_counts_updater = (
            vector_store_similarity_counts_updater
        )

    def process(
        self,
        d: npt.NDArray[np.float32],
        i: npt.NDArray[np.int32],
        query_ids: npt.NDArray[np.int32],
        vector_store_similarity_counts: SimilarityCounts,
        search_stats: List[SearchStats],
    ) -> Tuple[SimilarityCounts, List[SearchStats]]:

        candidate_queries = np.repeat(
            query_ids, i.shape[1]
        )  # repeat each query ID k times
        candidate_targets = i.ravel()  # flatten neighbor indices
        candidate_distances = d.ravel()  # flatten distances

        mask = self.similarity_threshold_filter.filter(
            query_ids=candidate_queries,
            target_ids=candidate_targets,
            distances=candidate_distances,
        )
        candidate_queries = candidate_queries[mask]
        candidate_targets = candidate_targets[mask]
        candidate_distances = candidate_distances[mask]

        for search_stat in search_stats:
            search_stat.update_similarity(similarities=candidate_distances)

        mask = self.precursor_mz_filter.filter(
            query_ids=candidate_queries,
            target_ids=candidate_targets,
            distances=candidate_distances,
        )
        candidate_queries = candidate_queries[mask]
        candidate_targets = candidate_targets[mask]

        valid_queries = candidate_queries
        valid_targets = candidate_targets

        _, hit_counts = np.unique(valid_queries, return_counts=True)
        for search_stat in search_stats:
            search_stat.update_hits(hits_per_query=hit_counts)

        # Update similarity sets
        return (
            self.vector_store_similarity_counts_updater.update(
                valid_queries, valid_targets, vector_store_similarity_counts
            ),
            search_stats,
        )
