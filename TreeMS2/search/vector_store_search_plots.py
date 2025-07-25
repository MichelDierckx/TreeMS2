from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


class HitHistogram:
    """Plots the number of hits per query ingestion."""

    @staticmethod
    def plot(bin_edges: Sequence[int], counts: np.ndarray, path: str):
        """Plots the histogram from precomputed counts."""
        total_spectra = counts.sum()
        relative_frequencies = counts / total_spectra

        plt.figure(figsize=(10, 5))
        plt.bar(
            range(len(counts)),
            relative_frequencies,
            width=1.0,
            edgecolor="black",
            align="edge",
        )

        for i, rel_freq in enumerate(relative_frequencies):
            if rel_freq > 0:
                plt.text(
                    i + 0.5,
                    rel_freq,
                    f"{rel_freq:.2%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.xlabel("Number of Similar Spectra Found (Hits)")
        plt.ylabel("Proportion of Queried Spectra")
        total_spectra_str = f"{total_spectra:,}".replace(",", " ")
        plt.title(f"Distribution of Query Hits (N = {total_spectra_str})")

        # Interval labels
        interval_labels = [f"{n:,}".replace(",", " ") for n in bin_edges]
        plt.xticks(ticks=range(len(counts) + 1), labels=interval_labels, rotation=-90)
        plt.xlim(left=0, right=len(counts))

        plt.savefig(path, bbox_inches="tight")
        plt.close()


class SimilarityHistogram:
    """Plots the similarity score distribution."""

    @staticmethod
    def plot(bin_edges: Sequence[float], counts: np.ndarray, path: str):
        plt.figure(figsize=(10, 5))
        plt.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            align="edge",
            edgecolor="black",
        )
        plt.xlabel("Similarity Score")
        plt.ylabel("Number of Spectrum Pairs")
        plt.title("Similarity Score Distribution")
        plt.xticks(rotation=45)
        plt.xlim(bin_edges[0], bin_edges[-1])
        plt.savefig(path, bbox_inches="tight")
        plt.close()
