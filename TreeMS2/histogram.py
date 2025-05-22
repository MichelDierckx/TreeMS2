from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from TreeMS2.logger_config import get_logger

logger = get_logger(__name__)


class HitHistogram:
    """Tracks the number of hits per query spectrum."""

    def __init__(self, bin_edges):
        self.bin_edges = bin_edges
        self.counts = np.zeros(len(bin_edges) - 1, dtype=int)

    def update(self, hits_per_query: np.ndarray):
        """Updates histogram using precomputed hit counts per query."""
        binned_hits = np.digitize(hits_per_query, self.bin_edges) - 1
        binned_hits = binned_hits[binned_hits <= (len(self.counts) - 1)]
        unique, counts = np.unique(binned_hits, return_counts=True)
        np.add.at(self.counts, unique, counts)

    def plot(self, path: str):
        """Plots the histogram of hits per spectrum with uniform bar widths."""
        # Use range(len(self.counts)) to make all bars the same width
        total_spectra = self.counts.sum()
        relative_frequencies = self.counts / total_spectra
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(self.counts)), relative_frequencies, width=1.0, edgecolor="black", align="edge")

        # Add relative frequency labels on top of bars
        for i, rel_freq in enumerate(relative_frequencies):
            if rel_freq > 0:  # Avoid labeling zero-height bars
                plt.text(i + 0.5, rel_freq, f"{rel_freq:.2%}", ha="center", va="bottom", fontsize=8)

        plt.xlabel("Number of Similar Spectra Found (Hits)")
        plt.ylabel("Proportion of Queried Spectra")
        total_spectra_str = f"{total_spectra:,}".replace(",", " ")
        plt.title(f"Distribution of Query Hits (N = {total_spectra_str})")

        # plt.xscale('log')  # Keep the x-axis logarithmic

        # Create interval labels for bins
        interval_labels = [f"{n:,}".replace(",", " ") for n in self.bin_edges]

        # Set custom x-ticks at uniform positions
        plt.xticks(ticks=range(len(self.counts) + 1), labels=interval_labels, rotation=-90)
        plt.xlim(left=0, right=len(self.counts))

        plt.savefig(path, bbox_inches="tight")
        plt.close()


class SimilarityHistogram:
    """Tracks the distribution of similarity scores."""

    def __init__(self, bin_edges=np.linspace(0, 1, 21)):
        self.bin_edges = bin_edges
        self.counts = np.zeros(len(bin_edges) - 1, dtype=int)

    def update(self, d):
        """Updates the histogram based on FAISS similarity scores."""
        binned_similarities = np.digitize(d, self.bin_edges) - 1
        # clip so that 100% is also included last bin
        binned_similarities = np.clip(binned_similarities, 0, len(self.counts) - 1)

        # Efficiently accumulate counts
        unique, counts = np.unique(binned_similarities, return_counts=True)
        np.add.at(self.counts, unique, counts)

    def plot(self, path: str):
        """Plots the histogram of similarity scores."""
        plt.figure(figsize=(10, 5))
        plt.bar(self.bin_edges[:-1], self.counts, width=np.diff(self.bin_edges),
                align="edge", edgecolor="black")
        plt.xlabel("Similarity Score")
        plt.ylabel("Number of Spectrum Pairs")
        plt.title("Similarity Score Distribution")
        plt.xticks(rotation=45)
        plt.xlim(self.bin_edges[0], self.bin_edges[-1])
        plt.savefig(path, bbox_inches="tight")
        plt.close()


class PrecursorChargeHistogram:
    def __init__(self):
        """
        Initializes the ChargeHistogram.
        """
        self.charge_counts = Counter()

    def update(self, charge_counts: Counter):
        """
        Updates the histogram given charge counts.
        :param charge_counts: A Counter dictionary where keys are charge categories and values are their counts.
        :return:
        """
        self.charge_counts.update(charge_counts)

    def plot(self, path: str):
        """
        Plots and saves the charge histogram.

        :param path: Path where the plot image will be saved.
        """
        if not self.charge_counts:
            return

        # Extract charge categories and counts
        charges, counts = zip(*sorted(self.charge_counts.items()))  # Sort for better visualization
        charges = [str(c) for c in charges]  # convert into categories to prevent x-axis scaling
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.bar(charges, counts, edgecolor="black")

        # Labels & styling
        plt.xlabel("Precursor Charge")
        plt.ylabel("Number of Spectra")
        plt.title("Number of Spectra by Precursor Charge")
        plt.xticks(charges)  # Ensure all charge categories are labeled
        plt.grid(axis="y", linestyle="--")

        # Save plot
        plt.savefig(path, bbox_inches="tight")
        plt.close()
