import faiss
import numpy as np
import scipy.sparse as ss

from sklearn.random_projection import SparseRandomProjection


class DimensionalityReducer:
    """
    Reduces high-dimensional sparse vectors to low-dimensional dense vectors.
    """

    def __init__(self, low_dim: int, high_dim: int):
        if low_dim > high_dim:
            raise ValueError(
                f"low_dim ({low_dim}) cannot be greater than high_dim ({high_dim})."
            )
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
        if vectors.shape[1] != self.high_dim:
            raise ValueError(
                f"Input vectors have {vectors.shape[1]} dimensions, but the reducer expects {self.high_dim} dimensions."
            )

        if vectors.shape[0] == 0:
            raise ValueError("Input vectors must contain at least one row.")

        dense_vectors = (vectors @ self.transformation).toarray()
        if normalize:
            if dense_vectors.shape[0] > 0 and np.allclose(
                np.linalg.norm(dense_vectors, axis=1), 0
            ):
                raise ValueError(
                    "Input vectors must not be zero-vectors for normalization."
                )
            # Normalize for cosine similarity
            faiss.normalize_L2(dense_vectors)
        return dense_vectors

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(low_dim={self.low_dim}, high_dim={self.high_dim})"
