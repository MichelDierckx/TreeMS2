import math
import os
import time
from typing import Tuple, Optional, Iterator

import faiss
import numpy as np
import psutil
from tqdm import tqdm

from ..logger_config import get_logger
from ..vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class IndexConstructionError(Exception):
    pass


class VectorStoreIndex:
    def __init__(self, vector_store: VectorStore, use_gpu: bool):
        """
        Index for fast ms/ms spectrum similarity search.
        """
        self.vector_store = vector_store

        if use_gpu:
            if faiss.get_num_gpus():
                self.use_gpu = True
                print(faiss.get_num_gpus())
            else:
                logger.warning("No GPU found. Using CPU for index training.")
                self.use_gpu = False
        self.use_gpu = use_gpu

        self.index, self.index_type, self.nlist = self._initialize_index(self.vector_store.vector_count,
                                                                         self.vector_store.vector_count, use_gpu)

        self.metric = faiss.METRIC_INNER_PRODUCT

        logger.info(f"Created index {self}")

    @staticmethod
    def _create_factory_string(vector_count: int, vector_dim: int) -> str:
        # determine the type of index (and number of clusters if applicable)
        if vector_count < 10_000:
            factory_string = "IDMap,Flat"
        elif vector_count < 10 ** 6:
            nlist = min(math.floor(16 * math.sqrt(vector_count)),
                        math.floor(vector_count / 39))  # need a minimum of 39 training points per cluster
            factory_string = f"IVF{nlist}"
        elif vector_count < 10 ** 7:
            nlist = min(math.floor(20 * math.sqrt(vector_count)), math.floor(vector_count / 39))
            factory_string = f"IVF{nlist}_HNSW32"
        elif vector_count < 10 ** 8:
            nlist = min(math.floor(26 * math.sqrt(vector_count)), math.floor(vector_count / 39))
            factory_string = f"IVF{nlist}_HNSW32"
        elif vector_count <= 10 ** 9:
            nlist = min(math.floor(33 * math.sqrt(vector_count)), math.floor(vector_count / 39))
            factory_string = f"IVF{nlist}_HNSW32"
        else:
            raise IndexConstructionError(
                f"An index can be constructed for a maximum of 1B vectors, but got {vector_count} vectors.")

        # determine compression
        if vector_count >= 10 ** 6:
            system_total_ram = psutil.virtual_memory().total
            current_process_ram_usage = psutil.Process().memory_info().rss

            # calculate the amount of RAM that will be reserved for the OS
            giga_byte = 1024 ** 3
            ram_reserved_for_os = 2 * giga_byte  # initial RAM reserved
            if system_total_ram > 4 * giga_byte:
                ram_reserved_for_os += ((min(system_total_ram, 16) - 4) // 4) * giga_byte  # +1GB per 4GB (up to 16GB)
            if system_total_ram > 16 * giga_byte:
                ram_reserved_for_os += ((system_total_ram - 16) // 8) * giga_byte  # +1GB per 8GB above 16GB

            memory_budget = system_total_ram - current_process_ram_usage - ram_reserved_for_os
            memory_budget_per_vector = math.floor(memory_budget / vector_count)
            m = memory_budget_per_vector - 16
            if m < 64:
                raise IndexConstructionError(
                    f"Not enough memory available for indexing {vector_dim} vectors. Index requires at least {(64 + 16) * vector_count} bytes, but only an estimated {memory_budget} bytes of memory available.")
            if m > vector_dim:
                if m > 4 * vector_dim:
                    factory_string += ",Flat"
                elif m > 2 * vector_dim:
                    factory_string += ",SQfp16"
                elif m >= vector_dim:
                    factory_string += ",SQ8"
            else:
                if 4 * m <= vector_dim:
                    factory_string = f"OPQ{m}_{4 * m}," + factory_string + f",PQ{m}"
                else:
                    factory_string = f"OPQ{m}," + factory_string + f",PQ{m}"
        return factory_string

    @classmethod
    def _initialize_index(cls, n_spectra: int, d: int, use_gpu: bool) -> Tuple[faiss.Index, str, Optional[int]]:
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
            nlist = min(16 * 2 ** 10, n_spectra // 39)
            if use_gpu:
                index_type = f"IVF{nlist},SQ8"
                index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
            else:
                index_type = f"IVF{nlist},Flat"
                index = faiss.IndexIVFFlat(
                    faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT
                )
        elif n_spectra <= 10 ** 7:
            nlist = min(64 * 2 ** 10, n_spectra // 39)
            if use_gpu:
                index_type = f"IVF{nlist}_HNSW32,SQ8"
                index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
            else:
                index_type = f"IVF{nlist},HNSW"
                quantizer = faiss.IndexHNSWFlat(d, 32)
                index = faiss.IndexIVFFlat(
                    quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
                )
        elif n_spectra <= 50 ** 7:
            nlist = min(128 * 2 ** 10, n_spectra // 39)
            if use_gpu:
                index_type = f"IVF{nlist}_HNSW32,SQ8"
                index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)
            else:
                index_type = f"IVF{nlist},SQ8"
                index = faiss.IndexIVFFlat(
                    faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT
                )
        else:
            # the number of spectra exceeds the allowed limit
            max_spectra = 50 ** 7  # The largest number of spectra allowed
            raise IndexingTooLargeError(n_spectra, max_spectra)
        return index, index_type, nlist

    def train(self):
        if not self.index.is_trained:
            if not self.nlist is None:
                if self.use_gpu:
                    # extract the clustering index and move to GPU
                    index_ivf = faiss.extract_index_ivf(self.index)
                    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(index_ivf.d))
                    index_ivf.clustering_index = clustering_index

                sample_size = min(39 * self.nlist, self.vector_store.vector_count)
                training_data = self.vector_store.sample(sample_size)
                logger.info(f"Training index on {sample_size} samples.")
                train_time_start = time.time()
                self.index.train(training_data)
                logger.info(f"Finished training index in {time.time() - train_time_start:.3f} seconds.")

    def add(self, batch_size: int):
        """
        Add vectors present in the vector store to the FAISS index in batch.
        :param batch_size: the number of vectors in a batch
        :return:
        """
        logger.info("Adding spectra to the index...")
        with tqdm(desc="Spectra added to index", unit=f" spectrum", total=self.vector_store.count_vectors()) as pbar:
            for vectors, ids, nr_vectors in self.vector_store.to_vector_batches(batch_size=batch_size):
                self.index.add_with_ids(vectors, ids)
                pbar.update(nr_vectors)
        logger.info("Added all spectra to the index.")

    def save_index(self, path):
        """
        Save the FAISS index to a file.
        :param path: filepath (str): Path to save the index.
        :return:
        """
        faiss.write_index(self.index, path)
        logger.debug(f"Saved index to {path}")

    @staticmethod
    def load(path: str, vector_store: VectorStore, use_gpu: bool) -> Optional["VectorStoreIndex"]:
        """
        Load a FAISS index from a specified file.

        Returns an MS2Index instance if successful, otherwise None.
        """
        if not os.path.exists(path):
            return None
        try:
            vector_store_index = VectorStoreIndex(vector_store, use_gpu=use_gpu)
            index = faiss.read_index(path)  # Attempt to load the index
            vector_store_index.index = index
            return vector_store_index
        except Exception:  # Catch all exceptions silently
            return None  # Return None if loading fails

    def range_search(self, similarity_threshold: float,
                     batch_size: int) -> \
            Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform a range search on the FAISS index for every vector in the vector store in batches.

        :param similarity_threshold: (float), Search radius.
        :param batch_size: the number of vectors in a batch

        :yield:

        A tuple containing:
            - lims (np.ndarray): The boundaries of search results for each query.
            - d (np.ndarray): The similarity distances for each result.
            - i (np.ndarray): The indices of the nearest neighbors.
        """

        logger.info("Querying the index for similar spectra ...")
        with tqdm(desc="Spectra queried", unit=f" spectrum", total=self.vector_store.vector_count) as pbar:
            for query_vectors, ids, nr_vectors in self.vector_store.to_vector_batches(batch_size=batch_size):
                # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
                lims, d, i = self.index.range_search(query_vectors, similarity_threshold)
                yield lims, d, i
                pbar.update(nr_vectors)
        logger.info("Finished querying.")
