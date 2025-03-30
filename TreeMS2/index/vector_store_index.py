import math
import os
import time
from typing import Tuple, Optional, Iterator

import faiss
from faiss.contrib import clustering
import numpy as np
import psutil
from tqdm import tqdm

from ..logger_config import get_logger
from ..vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class IndexConstructionError(Exception):
    pass


class VectorStoreIndex:
    def __init__(self, vector_store: VectorStore):
        """
        Index for fast ms/ms spectrum similarity search.
        """
        self.vector_store = vector_store

        factory_string, nlist = self._create_factory_string(vector_count=vector_store.vector_count,
                                                            vector_dim=vector_store.vector_dim)
        self.index = faiss.index_factory(vector_store.vector_dim, factory_string, faiss.METRIC_INNER_PRODUCT)
        self.factory_string = factory_string
        self.nlist = nlist
        logger.info(
            f"Creating FAISS index with factory string '{factory_string}' for vector store '{self.vector_store.name}'.")

    @staticmethod
    def _create_factory_string(vector_count: int, vector_dim: int) -> Tuple[str, int]:
        # determine the type of index (and number of clusters if applicable)
        if vector_count < 10_000:  # N < 10k
            nlist = 0
            factory_string = "IDMap,Flat"
        elif vector_count < 10 ** 6:  # N < 1M
            nlist = min(math.floor(16 * math.sqrt(vector_count)),
                        math.floor(vector_count / 39))  # need a minimum of 39 training points per cluster
            factory_string = f"IVF{nlist}"
        elif vector_count < 10 ** 7:  # N < 10M
            nlist = min(math.floor(20 * math.sqrt(vector_count)), math.floor(vector_count / 39))
            factory_string = f"IVF{nlist}_HNSW32"
        elif vector_count < 10 ** 8:  # N < 100M
            nlist = min(math.floor(26 * math.sqrt(vector_count)), math.floor(vector_count / 39))
            factory_string = f"IVF{nlist}_HNSW32"
        elif vector_count <= 10 ** 9:  # N <= 1B
            nlist = min(math.floor(33 * math.sqrt(vector_count)), math.floor(vector_count / 39))
            factory_string = f"IVF{nlist}_HNSW32"
        else:  # N > 1B
            raise IndexConstructionError(
                f"An index can be constructed for a maximum of 1B vectors, but got {vector_count} vectors.")

        # determine compression
        if vector_count >= 10 ** 6:
            system_total_ram = psutil.virtual_memory().total
            current_process_ram_usage = psutil.Process().memory_info().rss

            # calculate the amount of RAM that will be reserved for the OS
            giga_byte = 1024 ** 3
            system_total_ram_gb = system_total_ram // giga_byte
            ram_reserved_for_os = 2 * giga_byte  # initial RAM reserved
            if system_total_ram > 4 * giga_byte:
                ram_reserved_for_os += ((min(system_total_ram_gb,
                                             16) - 4) // 4) * giga_byte  # +1GB per 4GB (up to 16GB)
            if system_total_ram > 16 * giga_byte:
                ram_reserved_for_os += ((system_total_ram_gb - 16) // 8) * giga_byte  # +1GB per 8GB above 16GB

            memory_budget = max(system_total_ram - current_process_ram_usage - ram_reserved_for_os, 0)
            memory_budget_per_vector = math.floor(memory_budget / vector_count)
            m = max(memory_budget_per_vector - 16, 0)
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
        return factory_string, nlist

    def _load_training_data(self) -> Tuple[int, np.ndarray]:
        nr_of_training_points = min(39 * self.nlist, self.vector_store.vector_count)
        training_data = self.vector_store.sample(nr_of_training_points)
        return nr_of_training_points, training_data

    def _train_cpu(self):
        nr_of_training_points, training_data = self._load_training_data()
        logger.info(f"Training index on CPU using {nr_of_training_points} training points.")
        train_time_start = time.time()
        self.index.train(training_data)
        logger.info(f"Finished training index in {time.time() - train_time_start:.3f} seconds.")

    def _train_gpu(self):
        index_ivf = faiss.extract_index_ivf(self.index)
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(index_ivf.d))
        index_ivf.clustering_index = clustering_index

        nr_of_training_points, training_data = self._load_training_data()
        logger.info(f"Training index on GPU using {nr_of_training_points} training points.")
        train_time_start = time.time()
        self.index.train(training_data)
        logger.info(f"Finished training index in {time.time() - train_time_start:.3f} seconds.")

    def _train_cpu_2level(self):
        nr_of_training_points, training_data = self._load_training_data()
        logger.info(f"Training index on CPU (two-level clustering) using {nr_of_training_points} training points.")
        train_time_start = time.time()
        clustering.train_ivf_index_with_2level(self.index, training_data)
        logger.info(f"Finished training index in {time.time() - train_time_start:.3f} seconds.")

    def train(self, use_gpu: bool = False):
        if self.index.is_trained:
            return
        if self.nlist <= 0:
            return
        if self.vector_store.vector_count < 10 ** 7:
            self._train_cpu()
        else:
            if use_gpu:
                if faiss.get_num_gpus():
                    self._train_gpu()
                else:
                    logger.warning("No GPU found, using CPU (two-level clustering) instead for training.")
                    self._train_cpu_2level()
            else:
                self._train_cpu_2level()

    def add(self, batch_size: int):
        """
        Add vectors present in the vector store to the FAISS index in batch.
        :param batch_size: the number of vectors in a batch
        :return:
        """
        with tqdm(desc="Vectors added to index", unit=f" vector", total=self.vector_store.count_vectors()) as pbar:
            for vectors, ids, nr_vectors in self.vector_store.to_vector_batches(batch_size=batch_size):
                self.index.add_with_ids(vectors, ids)
                pbar.update(nr_vectors)

    def save_index(self, path):
        """
        Save the FAISS index to a file.
        :param path: filepath (str): Path to save the index.
        :return:
        """
        faiss.write_index(self.index, path)
        logger.debug(f"Saved index to {path}")

    @staticmethod
    def load(path: str, vector_store: VectorStore) -> Optional["VectorStoreIndex"]:
        """
        Load a FAISS index from a specified file.
        """
        if not os.path.exists(path):
            return None
        try:
            vector_store_index = VectorStoreIndex(vector_store)
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
        with tqdm(desc="Vectors queried", unit=f" vector", total=self.vector_store.vector_count) as pbar:
            for query_vectors, ids, nr_vectors in self.vector_store.to_vector_batches(batch_size=batch_size):
                # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
                lims, d, i = self.index.range_search(query_vectors, similarity_threshold)
                yield lims, d, i
                pbar.update(nr_vectors)
