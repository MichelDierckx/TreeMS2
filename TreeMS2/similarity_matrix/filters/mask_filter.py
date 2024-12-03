from abc import ABC, abstractmethod

from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix


class MaskFilter(ABC):
    def __init__(self, mask: SpectraMatrix):
        self.mask: SpectraMatrix = mask

    def apply(self, similarity_matrix: SimilarityMatrix):
        """
        Apply a mask filter to the similarity matrix.
        :param similarity_matrix: the similarity matrix to which the mask filter is applied
        :return:
        """
        similarity_matrix.subtract(self.mask)
