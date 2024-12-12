from typing import Tuple, Optional

import faiss
import numpy as np
from tqdm import tqdm

from ..logger_config import get_logger
from ..similarity_matrix.similarity_matrix import SimilarityMatrix
from ..vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class IndexingMemoryError(Exception):
    """Custom exception raised when there is not enough memory to index the spectra."""

    def __init__(self, total_valid_spectra, d, memory_budget_gigabytes, memory_required):
        message = (
            f"Not enough memory to index {total_valid_spectra} spectra with dimensions {d}. "
            f"Memory required: {memory_required} bytes, but only {memory_budget_gigabytes} GB available."
        )
        super().__init__(message)
        self.total_valid_spectra = total_valid_spectra
        self.d = d
        self.memory_budget_gigabytes = memory_budget_gigabytes
        self.memory_required = memory_required


class IndexingTooLargeError(Exception):
    """Custom exception raised when the number of spectra is too large for the current indexing method."""

    def __init__(self, total_valid_spectra, max_spectra):
        message = (
            f"The number of spectra ({total_valid_spectra}) exceeds the maximum supported limit "
            f"for the current indexing method. The limit is {max_spectra} spectra."
        )
        super().__init__(message)
        self.total_valid_spectra = total_valid_spectra
        self.max_spectra = max_spectra


class MS2Index:
    def __init__(self, total_valid_spectra: int, d: int, work_dir: str):
        """
        Index for fast ms/ms spectrum similarity search.
        :param total_valid_spectra: the total number of valid spectra
        :param d: the dimension of the spectra to be indexed
        :param work_dir: the working directory
        """

        self.total_valid_spectra = total_valid_spectra
        self.d = d
        self.work_dir = work_dir

        self.index, self.index_type, self.nlist = self._initialize_index(total_valid_spectra, d)

        self.metric = faiss.METRIC_INNER_PRODUCT

        logger.info(f"Created index {self}")

    @classmethod
    def _initialize_index(cls, n_spectra: int, d: int) -> Tuple[faiss.Index, str, Optional[int]]:
        """
        Create and initialize the appropriate index structure.
        :param n_spectra: The number of spectra to be indexed
        :param d: the dimension of the spectra to be indexed
        :return:
        """
        # Memory budget is 4GB by default
        memory_budget_gigabytes = 4
        memory_budget_bytes = memory_budget_gigabytes * 10 ** 9

        # Memory required for the index
        m = (memory_budget_bytes - (16 * n_spectra)) // n_spectra  # Ensure m is memory per vector

        if m <= 4 * d:
            # Memory required exceeds the budget
            memory_required = n_spectra * (m + 16)  # calculate total memory required
            raise IndexingMemoryError(n_spectra, d, memory_budget_gigabytes, memory_required)

        # Proceed with index creation depending on the number of spectra
        if n_spectra <= 10 ** 4:
            nlist = None
            index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
            index_type = "Flat"
        elif n_spectra <= 10 ** 6:
            nlist = 16 * 2 ** 10
            index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index_type = "IVF16K,Flat"
        elif n_spectra <= 10 ** 8:
            nlist = 64 * 2 ** 10
            index = faiss.IndexIVFFlat(
                faiss.IndexHNSWFlat(d), d, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index_type = "IVF64K,HNSW"
        else:
            # the number of spectra exceeds the allowed limit
            max_spectra = 10 ** 8  # The largest number of spectra allowed
            raise IndexingTooLargeError(n_spectra, max_spectra)
        return index, index_type, nlist

    def train(self, vector_store: VectorStore):
        if not self.index.is_trained:
            if not self.nlist is None:
                logger.info("Training index...")
                sample_size = min(50 * self.nlist, self.total_valid_spectra)
                training_data = vector_store.sample(sample_size)
                self.index.train(training_data)
                logger.info(f"Trained index.")

    def add(self, vector_store: VectorStore, batch_size: int):
        """
        Add vectors present in the vector store to the FAISS index in batch.
        :param batch_size: the number of vectors in a batch
        :param vector_store: a VectorStore instance to retrieve the vector data
        :return:
        """
        logger.info("Adding spectra to the index...")
        for vectors, ids, nr_vectors in tqdm(
                vector_store.to_vector_batches(batch_size=batch_size),
                desc="Batches added to index", unit=f"{batch_size} spectra", total=vector_store.count_spectra()):
            self.index.add_with_ids(vectors, ids)
        logger.info("Added all spectra to the index.")

    def save_index(self, filepath):
        """
        Save the FAISS index to a file.
        :param filepath: filepath (str): Path to save the index.
        :return:
        """
        faiss.write_index(self.index, filepath)
        logger.debug(f"Saved index to {filepath}")

    def load_index(self, filepath):
        """
        Load a FAISS index from a file.
        :param filepath: str, path to the saved the index.
        :return:
        """
        self.index = faiss.read_index(filepath)
        logger.debug(f"Loaded index from {filepath}")

    def range_search(self, similarity_threshold: float, vector_store: VectorStore,
                     batch_size: int) -> SimilarityMatrix:
        """
        Perform a range search on the FAISS index for every vector in the vector store in batches. Capture result in Distances object.

        :param similarity_threshold: (float), Search radius.
        :param vector_store: a VectorStore instance to retrieve the vector data
        :param batch_size: the number of vectors in a batch

        :return: Distances
        """
        radius = 1.0 - similarity_threshold
        similarity_matrix = SimilarityMatrix(self.total_valid_spectra, similarity_threshold=similarity_threshold)

        logger.info("Querying the index for similar spectra ...")
        # https://github.com/facebookresearch/faiss/wiki/FAQ#is-it-possible-to-dynamically-exclude-vectors-based-on-some-criterion
        # sel = faiss.IDSelectorNot(faiss.IDSelectorRange(group.begin, group.end + 1))
        # params = faiss.SearchParameters(sel=sel)
        for query_vectors, ids, nr_vectors in tqdm(
                vector_store.to_vector_batches(batch_size=batch_size),
                desc="Batches queried", unit=f"{batch_size} spectra", total=vector_store.count_spectra()):

            # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
            lims, d, i = self.index.range_search(query_vectors, radius)

            # nr of similar vectors found
            total_results = lims[-1]
            if total_results == 0:
                continue  # Skip processing if no results found

            # preallocate arrays
            data = np.full(total_results, True, dtype=bool)
            rows = np.empty(total_results, dtype=np.int64)
            cols = np.empty(total_results, dtype=np.int64)

            # Populate rows and cols
            for query_idx in range(query_vectors.shape[0]):
                start = lims[query_idx]
                end = lims[query_idx + 1]
                # Fill only if there are results
                if start < end:
                    rows[start:end] = ids[query_idx]
                    cols[start:end] = i[start:end]

            similarity_matrix.update(data=data, rows=rows, cols=cols)
        logger.info("Finished querying.")
        return similarity_matrix

    def __repr__(self):
        return (f"MS2Index(d={self.d}, "
                f"index_type={self.index_type}, "
                f"metric={'INNER_PRODUCT' if self.metric == faiss.METRIC_INNER_PRODUCT else 'L2'})")
