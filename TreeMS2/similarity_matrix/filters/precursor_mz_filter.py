import numpy as np
from scipy.sparse import csr_matrix

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class PrecursorMzFilter(MaskFilter):
    def __init__(self, vector_store: VectorStore,
                 precursor_mz_window: float):
        self.precursor_mz_window = precursor_mz_window
        self.vector_store = vector_store
        super().__init__(None)

    def construct_mask(self, similarity_matrix: SimilarityMatrix) -> SpectraMatrix:
        rows, cols = similarity_matrix.matrix.nonzero()

        # precursor mz values for all entries
        precursor_mz = self.vector_store.get_col("precursor_mz").to_numpy(dtype=np.float32).ravel()

        # Vectorized lookup using rows and cols
        precursor_mz_rows = precursor_mz[rows]
        precursor_mz_cols = precursor_mz[cols]

        mask_data = np.abs(precursor_mz_rows - precursor_mz_cols) > self.precursor_mz_window
        # Filter the rows and columns based on the mask
        rows = rows[mask_data]
        cols = cols[mask_data]
        mask_data = np.ones(rows.size, dtype=np.bool_)

        m = csr_matrix((mask_data, (rows, cols)), shape=similarity_matrix.matrix.shape,
                       dtype=np.bool_)
        mask = SpectraMatrix(m)
        return mask
