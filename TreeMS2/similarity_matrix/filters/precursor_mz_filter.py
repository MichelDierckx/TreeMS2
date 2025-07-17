import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix

logger = get_logger(__name__)


class PrecursorMzFilter(MaskFilter):
    def __init__(self,
                 precursor_mz_window: float, precursor_mzs: npt.NDArray[np.float32]):
        self.precursor_mz_window = precursor_mz_window
        self.precursor_mzs = precursor_mzs
        super().__init__(None)

    def construct_mask(self, similarity_matrix: SimilarityMatrix) -> SpectraMatrix:
        rows, cols = similarity_matrix.matrix.nonzero()

        # Vectorized lookup using rows and cols
        precursor_mz_rows = self.precursor_mzs[rows]
        precursor_mz_cols = self.precursor_mzs[cols]

        mask_data = np.abs(precursor_mz_rows - precursor_mz_cols) > self.precursor_mz_window
        # Filter the rows and columns based on the mask
        rows = rows[mask_data]
        cols = cols[mask_data]
        mask_data = np.ones(rows.size, dtype=np.bool_)

        m = csr_matrix((mask_data, (rows, cols)), shape=similarity_matrix.matrix.shape,
                       dtype=np.bool_)
        mask = SpectraMatrix(m)
        return mask
