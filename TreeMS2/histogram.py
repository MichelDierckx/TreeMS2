import matplotlib.pyplot as plt
import numpy as np

from TreeMS2.logger_config import get_logger

logger = get_logger(__name__)


class HitHistogram:
    """Tracks the number of hits per query spectrum."""

    def __init__(self, bin_edges=np.arange(0, 1000, 10)):
        self.bin_edges = bin_edges
        self.counts = np.zeros(len(bin_edges) - 1, dtype=int)

    def update(self, lims):
        """Updates the histogram based on FAISS range search results."""
        hits_per_query = np.diff(lims)  # Compute the number of results per query

        binned_hits = np.digitize(hits_per_query, self.bin_edges) - 1
        binned_hits = np.clip(binned_hits, 0, len(self.counts) - 1)

        # Efficiently accumulate counts
        unique, counts = np.unique(binned_hits, return_counts=True)
        np.add.at(self.counts, unique, counts)

    def plot(self, path: str):
        """Plots the histogram of hits per spectrum."""
        plt.bar(self.bin_edges[:-1], self.counts, width=1.0,
                align="edge", edgecolor="black", alpha=0.7)
        plt.xlabel("Number of Similar Spectra Found (Hits)")
        plt.ylabel("Number of Spectra Queries")
        plt.title("Distribution of Query Hits")
        plt.xscale('log')
        plt.xticks(ticks=self.bin_edges, labels=[str(int(edge)) for edge in self.bin_edges], rotation=45)
        plt.savefig(path, bbox_inches="tight")
        plt.close()


class SimilarityHistogram:
    """Tracks the distribution of similarity scores."""

    def __init__(self, bin_edges=np.linspace(0, 1, 21)):
        self.bin_edges = bin_edges
        print(bin_edges)
        self.counts = np.zeros(len(bin_edges) - 1, dtype=int)

    def update(self, d):
        """Updates the histogram based on FAISS similarity scores."""
        binned_similarities = np.digitize(d, self.bin_edges) - 1
        binned_similarities = np.clip(binned_similarities, 0, len(self.counts) - 1)

        # Efficiently accumulate counts
        unique, counts = np.unique(binned_similarities, return_counts=True)
        np.add.at(self.counts, unique, counts)

    def plot(self, path: str):
        """Plots the histogram of similarity scores."""
        plt.bar(self.bin_edges[:-1], self.counts, width=np.diff(self.bin_edges),
                align="edge", edgecolor="black", alpha=0.7)
        plt.xlabel("Similarity Score")
        plt.ylabel("Number of Spectrum Pairs")
        plt.title("Similarity Score Distribution")
        plt.xticks(rotation=45)
        plt.xlim(self.bin_edges[0], self.bin_edges[-1])
        plt.savefig(path, bbox_inches="tight")
        plt.close()
