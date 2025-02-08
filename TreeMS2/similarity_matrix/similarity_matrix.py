import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, load_npz

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class SimilarityMatrix(SpectraMatrix):
    def __init__(self, *args, similarity_threshold: float):
        # Initialize the parent class
        super().__init__(*args)
        self.similarity_threshold = similarity_threshold

    def update(self, data: npt.NDArray[np.bool_], rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]):
        self.matrix += csr_matrix((data, (rows, cols)), self.matrix.shape)

    def write(self, path: str) -> str:
        path = super().write(path)
        return path

    def write_global(self, path: str, total_spectra: int, vector_store: VectorStore):
        path = super().write_global(path, total_spectra, vector_store)
        return path

    @classmethod
    def load_with_threshold(cls, path: str, similarity_threshold: float) -> 'SimilarityMatrix':
        # Load the matrix from file
        matrix = load_npz(path)
        # Create an instance of SimilarityMatrix with the loaded matrix and threshold
        return cls(matrix, similarity_threshold=similarity_threshold)

    def __sub__(self, other):
        if isinstance(other, SpectraMatrix):
            # Perform the subtraction and return a SimilarityMatrix
            return SimilarityMatrix(self.matrix - other.matrix, similarity_threshold=self.similarity_threshold)
        else:
            raise ValueError("Subtraction is only supported between SimilarityMatrix and SpectraMatrix instances.")
