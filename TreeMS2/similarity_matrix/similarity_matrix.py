import os

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix

logger = get_logger(__name__)


class SimilarityMatrix(SpectraMatrix):
    def __init__(self, *args, similarity_threshold: float):
        # Initialize the parent class
        super().__init__(*args)
        self.similarity_threshold = similarity_threshold

    def update(self, data: npt.NDArray[np.bool_], rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]):
        self.matrix += csr_matrix((data, (rows, cols)), self.matrix.shape)

    def write(self, work_dir: str, filename: str) -> str:
        # create path
        similarities_dir = os.path.join(work_dir, "similarities")
        os.makedirs(similarities_dir, exist_ok=True)

        path = super().write(similarities_dir, filename)
        logger.info(f"Similarity matrix has been written to '{path}'.")
        return path

    def __sub__(self, other):
        if isinstance(other, SpectraMatrix):
            # Perform the subtraction and return a SimilarityMatrix
            return SimilarityMatrix(self.matrix - other.matrix, similarity_threshold=self.similarity_threshold)
        else:
            raise ValueError("Subtraction is only supported between SimilarityMatrix and SpectraMatrix instances.")
