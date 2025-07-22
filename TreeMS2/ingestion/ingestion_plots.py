from typing import Dict
import matplotlib.pyplot as plt

class PrecursorChargeHistogram:
    @staticmethod
    def plot(charge_counts: Dict[int, int], path: str) -> None:
        if not charge_counts:
            print("No data to plot.")
            return

        charges, counts = zip(*sorted(charge_counts.items()))
        charges_str = [str(c) for c in charges]

        plt.figure(figsize=(10, 5))
        plt.bar(charges_str, counts, edgecolor="black")

        plt.xlabel("Precursor Charge")
        plt.ylabel("Number of Spectra")
        plt.title("Number of Spectra by Precursor Charge")
        plt.xticks(charges_str)
        plt.grid(axis="y", linestyle="--")

        plt.savefig(path, bbox_inches="tight")
        plt.close()
