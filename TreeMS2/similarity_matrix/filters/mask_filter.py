from abc import ABC, abstractmethod
from typing import Optional

from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix


class MaskFilter(ABC):
    def __init__(self, mask: Optional[SpectraMatrix]):
        self.mask: Optional[SpectraMatrix] = mask

    def apply(self, similarity_matrix: SimilarityMatrix):
        """
        Apply a mask filter to the similarity matrix.
        :param similarity_matrix: the similarity matrix to which the mask filter is applied
        :return:
        """
        if self.mask is None:
            self.mask = self.construct_mask(similarity_matrix)
        similarity_matrix.subtract(self.mask)

    @abstractmethod
    def construct_mask(self, similarity_matrix: SimilarityMatrix) -> SpectraMatrix:
        pass

    @abstractmethod
    def write_filter_statistics(self, work_dir: str):
        pass

    @abstractmethod
    def save_mask(self, work_dir: str):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
