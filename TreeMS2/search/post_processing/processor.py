from typing import Tuple, List

import numpy as np
import numpy.typing as npt

from TreeMS2.search.post_processing.filters import PairFilter
from TreeMS2.search.search_stats import SearchStats
from TreeMS2.search.similarity_counts import SimilarityCountsUpdater, SimilarityCounts


class SearchResultProcessor:
    def __init__(
        self,
        filters: list[PairFilter],
        vector_store_similarity_counts_updater: SimilarityCountsUpdater,
    ):
        self.filters = filters
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

        for search_stat in search_stats:
            search_stat.update_similarity(similarities=candidate_distances)

        # Apply filters sequentially
        mask = np.ones(len(candidate_queries), dtype=bool)
        for f in self.filters:
            mask &= f.filter(
                query_ids=candidate_queries,
                target_ids=candidate_targets,
                distances=candidate_distances,
            )

        # Keep only valid pairs
        valid_queries = candidate_queries[mask]
        valid_targets = candidate_targets[mask]

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
