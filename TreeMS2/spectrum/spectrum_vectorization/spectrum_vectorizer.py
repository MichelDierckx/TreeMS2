from typing import List

import numpy as np
import spectrum_utils.spectrum as sus

from .dimensionality_reducer import DimensionalityReducer
from .spectrum_binner import SpectrumBinner


class SpectrumVectorizer:
    """
    Vectorizes spectra using binning and dimensionality reduction.
    """

    def __init__(self, binner: SpectrumBinner, reducer: DimensionalityReducer, norm: bool = True):
        self.binner = binner
        self.reducer = reducer
        if self.binner.dim != self.reducer.low_dim:
            raise ValueError(
                f"Dimensionality mismatch: Binner dimensionality ({self.binner.dim}) "
                f"does not match reducer input dimensionality ({self.reducer.low_dim})."
            )
        self.norm = norm

    def vectorize(self, spectra: List[sus.MsmsSpectrum]) -> np.ndarray:
        """
        Convert spectra into low-dimensional dense vectors.
        """
        sparse_vectors = self.binner.bin(spectra)
        dense_vectors = self.reducer.reduce(sparse_vectors, normalize=self.norm)
        return dense_vectors
