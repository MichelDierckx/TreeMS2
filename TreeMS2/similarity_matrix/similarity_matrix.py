import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix

logger = get_logger(__name__)


class SimilarityMatrix(SpectraMatrix):
    def update(self, data: npt.NDArray[np.bool_], rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]):
        self.matrix += csr_matrix((data, (rows, cols)), self.matrix.shape)

    def write(self, path: str):
        logger.info(f"Writing similarity matrix to '{path}'.")
        super().write(path)

    def __sub__(self, other):
        if isinstance(other, SpectraMatrix):
            # Perform the subtraction and return a SimilarityMatrix
            return SimilarityMatrix(self.matrix - other.matrix)
        else:
            raise ValueError("Subtraction is only supported between SimilarityMatrix and SpectraMatrix instances.")
