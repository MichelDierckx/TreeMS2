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
        self.norm = norm

    def vectorize(self, spectra: List[sus.MsmsSpectrum]) -> np.ndarray:
        """
        Convert spectra into low-dimensional dense vectors.
        """
        sparse_vectors = self.binner.bin(spectra)
        dense_vectors = self.reducer.reduce(sparse_vectors, normalize=self.norm)
        return dense_vectors
