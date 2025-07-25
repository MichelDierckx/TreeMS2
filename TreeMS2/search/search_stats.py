from typing import Sequence

import numpy as np

from TreeMS2.search.vector_store_search_plots import HitHistogram, SimilarityHistogram


class SearchStats:
    """Aggregates search result statistics for hits and similarities."""

    def __init__(self, hit_bin_edges: Sequence[int], sim_bin_edges: Sequence[float]):
        self.hit_bin_edges = np.array(hit_bin_edges)
        self.sim_bin_edges = np.array(sim_bin_edges)

        self.hit_counts = np.zeros(len(self.hit_bin_edges) - 1, dtype=int)
        self.similarity_counts = np.zeros(len(self.sim_bin_edges) - 1, dtype=int)

    def update_hits(self, hits_per_query: np.ndarray):
        """Update histogram using precomputed hit counts per query."""
        binned_hits = np.digitize(hits_per_query, self.hit_bin_edges) - 1
        binned_hits = binned_hits[binned_hits <= (len(self.hit_counts) - 1)]
        unique, counts = np.unique(binned_hits, return_counts=True)
        np.add.at(self.hit_counts, unique, counts)

    def update_similarity(self, similarities: np.ndarray):
        """Update histogram based on FAISS similarity scores."""
        binned_similarities = np.digitize(similarities, self.sim_bin_edges) - 1
        binned_similarities = np.clip(
            binned_similarities, 0, len(self.similarity_counts) - 1
        )
        unique, counts = np.unique(binned_similarities, return_counts=True)
        np.add.at(self.similarity_counts, unique, counts)

    def export_hit_counts_to_histogram(self, path: str):
        HitHistogram.plot(
            bin_edges=self.hit_bin_edges, counts=self.hit_counts, path=path
        )

    def export_sim_counts_to_histogram(self, path: str):
        SimilarityHistogram.plot(
            bin_edges=self.sim_bin_edges, counts=self.similarity_counts, path=path
        )
