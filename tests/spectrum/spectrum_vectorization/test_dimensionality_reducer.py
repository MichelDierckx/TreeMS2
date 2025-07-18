import unittest

import numpy as np
import scipy.sparse as ss
from sklearn.metrics.pairwise import cosine_similarity

from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import DimensionalityReducer


class TestDimensionalityReducer(unittest.TestCase):
    def setUp(self):
        self.high_dim = 28000
        self.low_dim = 400
        self.num_vectors = 1000
        self.reducer = DimensionalityReducer(low_dim=self.low_dim, high_dim=self.high_dim)

    @staticmethod
    def _generate_sparse_matrix(num_vectors, high_dim, density=0.0005):
        matrix = ss.rand(num_vectors, high_dim, density=density, format='csr', dtype=np.float32)
        return matrix

    def test_dimensionality_reduction(self):
        # Create sparse input vectors
        vectors = self._generate_sparse_matrix(self.num_vectors, self.high_dim)

        # Apply dimensionality reduction
        reduced_vectors = self.reducer.reduce(vectors)

        # Check the shape of the output
        self.assertEqual(
            reduced_vectors.shape,
            (self.num_vectors, self.low_dim),
            "Output shape does not match the expected reduced dimensions."
        )

    def test_invalid_input_dimensions(self):
        # Create vectors with invalid dimensions
        self.reducer = DimensionalityReducer(low_dim=5, high_dim=20)
        invalid_vectors = self._generate_sparse_matrix(5, 20 + 1, density=0.5)

        with self.assertRaises(ValueError):
            self.reducer.reduce(invalid_vectors)

    def test_empty_input(self):
        # Create an empty sparse matrix
        empty_vectors = ss.csr_matrix((0, self.high_dim))

        with self.assertRaises(ValueError):
            self.reducer.reduce(empty_vectors)

    def test_cosine_similarity_preservation(self):
        # Create sparse input vectors
        vectors = self._generate_sparse_matrix(self.num_vectors, self.high_dim, density=0.001)

        # Compute original cosine similarities
        original_cosine_sim = cosine_similarity(vectors)

        # Apply dimensionality reduction
        reduced_vectors = self.reducer.reduce(vectors)

        # Compute reduced cosine similarities
        reduced_cosine_sim = cosine_similarity(reduced_vectors)

        # Check if the cosine similarity is approximately preserved
        diff = np.abs(original_cosine_sim - reduced_cosine_sim)
        mean_diff = np.mean(diff)
        tolerance = 0.05  # Tolerance for cosine similarity preservation

        self.assertLess(
            mean_diff,
            tolerance,
            f"Dimensionality reduction did not approximately preserve cosine similarity. Mean difference in cosine similarity ({mean_diff:.3f}) exceeded test tolerance ({tolerance:.3f})."
        )

    def test_zero_vector_normalization(self):
        # Create a zero-vector input
        zero_vector = ss.csr_matrix((self.low_dim, self.high_dim))

        with self.assertRaises(ValueError):
            self.reducer.reduce(zero_vector)


if __name__ == "__main__":
    unittest.main()
