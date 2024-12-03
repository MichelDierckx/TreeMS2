from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix


class PrecursorMzFilter(MaskFilter):
    def __init__(self, similarity_matrix: SimilarityMatrix):
        mask = self.construct_mask(similarity_matrix)
        super().__init__(mask)

    def construct_mask(self, similarity_matrix: SimilarityMatrix) -> SpectraMatrix:
        pass
