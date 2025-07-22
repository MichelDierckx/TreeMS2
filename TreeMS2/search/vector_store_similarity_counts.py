import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from TreeMS2.config.logger_config import get_logger
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix

logger = get_logger(__name__)


class SimilarityMatrix(SpectraMatrix):
    def __init__(self, *args, similarity_threshold: float):
        # Initialize the parent class
        super().__init__(*args)
        self.similarity_threshold = similarity_threshold

    def update(
        self,
        data: npt.NDArray[np.bool_],
        rows: npt.NDArray[np.int64],
        cols: npt.NDArray[np.int64],
    ):
        self.matrix += csr_matrix((data, (rows, cols)), self.matrix.shape)
