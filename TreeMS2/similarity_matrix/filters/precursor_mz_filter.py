import numpy as np
from scipy.sparse import csr_matrix

from TreeMS2.groups.groups import Groups
from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix
from TreeMS2.vector_store.vector_store import VectorStore


class PrecursorMzFilter(MaskFilter):
    def __init__(self, similarity_matrix: SimilarityMatrix, precursor_mz_window: float, vector_store: VectorStore,
                 groups: Groups):
        self.window = precursor_mz_window
        mask = self.construct_mask(similarity_matrix, vector_store, groups)
        super().__init__(mask)

    def construct_mask(self, similarity_matrix: SimilarityMatrix, vector_store: VectorStore,
                       groups: Groups) -> SpectraMatrix:
        rows, cols = similarity_matrix.matrix.nonzero()

        mask_data = np.zeros(shape=rows.shape, dtype=np.bool_)

        for index, row, col in enumerate(zip(rows, cols)):
            row_group_id = groups.get_group_id_from_global_id(row)
            col_group_id = groups.get_group_id_from_global_id(col)

            row_precursor_mz = vector_store.get_metadata(row_group_id, row, groups, "precursor_mz")
            col_precursor_mz = vector_store.get_metadata(col_group_id, col, groups, "precursor_mz")

            if abs(row_precursor_mz - col_precursor_mz) <= self.window:
                mask_data[index] = False

        m = csr_matrix(mask_data, (rows, cols))
        mask = SpectraMatrix(m)
        return mask
