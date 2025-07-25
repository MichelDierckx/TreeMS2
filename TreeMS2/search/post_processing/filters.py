from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class PairFilter(ABC):
    @abstractmethod
    def filter(
        self,
        query_ids: npt.NDArray[np.int32],
        target_ids: npt.NDArray[np.int32],
        distances: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        """Return boolean mask for valid pairs."""
        pass


class SimilarityThresholdFilter(PairFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def filter(
        self,
        query_ids: npt.NDArray[np.int32],
        target_ids: npt.NDArray[np.int32],
        distances: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        # Use row and col index from FAISS result to get scores
        return distances >= self.threshold


class PrecursorMzFilter(PairFilter):
    def __init__(self, precursor_mzs: npt.NDArray[np.float32], mz_window: float):
        self.precursor_mzs = precursor_mzs
        self.mz_window = mz_window

    def filter(
        self,
        query_ids: npt.NDArray[np.int32],
        target_ids: npt.NDArray[np.int32],
        distances: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        mz_queries = self.precursor_mzs[query_ids]
        mz_targets = self.precursor_mzs[target_ids]
        return np.abs(mz_queries - mz_targets) <= self.mz_window
