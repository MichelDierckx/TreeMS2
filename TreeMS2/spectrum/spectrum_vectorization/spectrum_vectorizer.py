from typing import List

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.logger_config import get_logger
from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import (
    DimensionalityReducer,
)
from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner

logger = get_logger(__name__)


class SpectrumVectorizer:
    """
    Vectorizes spectra using binning and dimensionality reduction.
    """

    def __init__(
        self, binner: SpectrumBinner, reducer: DimensionalityReducer, norm: bool = True
    ):
        self.binner = binner
        self.reducer = reducer
        if self.binner.dim != self.reducer.high_dim:
            raise ValueError(
                f"Dimensionality mismatch: Binner dimensionality ({self.binner.dim}) "
                f"does not match reducer input dimensionality ({self.reducer.high_dim})."
            )
        self.norm = norm
        logger.debug(f"Created {self}")

    def vectorize(self, spectra: List[sus.MsmsSpectrum]) -> np.ndarray:
        """
        Convert spectra into low-dimensional dense vectors.
        """
        sparse_vectors = self.binner.bin(spectra)
        dense_vectors = self.reducer.reduce(sparse_vectors, normalize=self.norm)
        return dense_vectors

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}):\n\t{self.binner}\n\t{self.reducer}"
