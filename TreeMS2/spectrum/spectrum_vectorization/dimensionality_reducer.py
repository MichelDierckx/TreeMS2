import faiss
import numpy as np
import scipy.sparse as ss

from sklearn.random_projection import SparseRandomProjection


class DimensionalityReducer:
    """
    Reduces high-dimensional sparse vectors to low-dimensional dense vectors.
    """

    def __init__(self, low_dim: int, high_dim: int):
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.transformation = self._create_transformation()

    def _create_transformation(self) -> ss.csr_matrix:
        """
        Generate a sparse random projection transformation matrix.
        """
        srp = SparseRandomProjection(n_components=self.low_dim, random_state=0)
        dummy_vector = np.zeros((1, self.high_dim))
        return srp.fit(dummy_vector).components_.astype(np.float32).T

    def reduce(self, vectors: ss.csr_matrix, normalize: bool = True) -> np.ndarray:
        """
        Apply dimensionality reduction and normalization to the input vectors.
        """
        dense_vectors = (vectors @ self.transformation).toarray()
        if normalize:
            # Normalize for cosine similarity
            faiss.normalize_L2(dense_vectors)
        return dense_vectors
