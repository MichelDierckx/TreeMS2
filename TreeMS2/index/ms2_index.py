from typing import Tuple, Optional, List

import faiss
from tqdm import tqdm

from ..groups.groups import Groups
from ..lance.lance_dataset_manager import LanceDatasetManager
from ..logger_config import get_logger

logger = get_logger(__name__)


class IndexingMemoryError(Exception):
    """Custom exception raised when there is not enough memory to index the spectra."""

    def __init__(self, n_spectra, d, memory_budget_gigabytes, memory_required):
        message = (
            f"Not enough memory to index {n_spectra} spectra with dimensions {d}. "
            f"Memory required: {memory_required} bytes, but only {memory_budget_gigabytes} GB available."
        )
        super().__init__(message)
        self.n_spectra = n_spectra
        self.d = d
        self.memory_budget_gigabytes = memory_budget_gigabytes
        self.memory_required = memory_required


class IndexingTooLargeError(Exception):
    """Custom exception raised when the number of spectra is too large for the current indexing method."""

    def __init__(self, n_spectra, max_spectra):
        message = (
            f"The number of spectra ({n_spectra}) exceeds the maximum supported limit "
            f"for the current indexing method. The limit is {max_spectra} spectra."
        )
        super().__init__(message)
        self.n_spectra = n_spectra
        self.max_spectra = max_spectra


class MS2Index:
    def __init__(self, n_spectra: int, d: int):
        """
        Index for fast ms/ms spectrum similarity search.
        :param n_spectra: the number of spectra to be indexed
        :param d: the dimension of the spectra to be indexed
        """

        self.d = d
        self.n_spectra = n_spectra

        self.index, self.index_type, self.nlist = self._initialize_index(n_spectra, d)

        self.metric = faiss.METRIC_INNER_PRODUCT

        logger.debug(f"Created index {self}")

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

    def train(self, lance_dataset_manager: LanceDatasetManager, group_ids: List[int]):
        if not self.index.is_trained:
            if not self.nlist is None:
                sample_size = min(50 * self.nlist, self.n_spectra)
                training_data = lance_dataset_manager.sample(self.n_spectra, group_ids)
                self.index.train(sample_size, training_data)
            logger.debug(f"Trained index.")

    def index_groups(self, lance_dataset_manager: LanceDatasetManager, groups: Groups, batch_size: int):
        """
        Add vectors to the FAISS index in batch for the given groups.
        :param batch_size: the number of vectors in a batch
        :param lance_dataset_manager: a LanceDatasetManager instance to retrieve the vector data
        :param groups: the groups to be indexed
        :return:
        """
        for group in tqdm(groups.get_groups(), desc="Groups added to index", unit="group"):
            for vectors, ids, nr_vectors in tqdm(
                    lance_dataset_manager.to_vector_batches(batch_size=batch_size, group=group),
                    desc="Batches added to index", unit="batch"):
                self.index.add_with_ids(vectors, nr_vectors, ids)

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

    def range_search(self, radius, lance_dataset_manager: LanceDatasetManager, groups: Groups, batch_size: int):
        """
        Perform a range search on the FAISS index for every vector for every group in batches.

        :param radius: (float), Search radius.
        :param lance_dataset_manager: a LanceDatasetManager instance to retrieve the vector data
        :param groups: the groups to be indexed
        :param batch_size: the number of vectors in a batch

        :return:
        """
        for group in tqdm(groups.get_groups(), desc="Groups queried", unit="group"):
            for query_vectors, ids, nr_vectors in tqdm(
                    lance_dataset_manager.to_vector_batches(batch_size=batch_size, group=group),
                    desc="Batches queried", unit="batch"):
                # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
                lims, d, i = self.index.range_search(query_vectors, nr_vectors, radius)

        return

    def __repr__(self):
        return (f"MS2Index(d={self.d}, "
                f"index_type={self.index_type}, "
                f"metric={'INNER_PRODUCT' if self.metric == faiss.METRIC_INNER_PRODUCT else 'L2'}")
