import multiprocessing
from typing import Optional, Set, Dict, List

import pandas as pd

from TreeMS2.groups.groups import Groups
from TreeMS2.vector_store.vector_store import VectorStore
from TreeMS2.logger_config import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    def __init__(self, base_path: str, vector_store_names: Set[str]):
        self.base_path = base_path
        self.vector_stores = {
            vector_store_name: VectorStore(base_path=base_path, name=vector_store_name)
            for vector_store_name in vector_store_names
        }

    def create_locks_and_flags(self, manager: multiprocessing.Manager) -> Dict[
        str, Dict[str, multiprocessing.Lock | multiprocessing.Value]]:
        """Returns a dictionary containing locks and overwrite flags for each vector store."""
        return {
            name: {
                "lock": manager.Lock(),
                "overwrite": manager.Value("b", True)
            }
            for name in self.vector_stores
        }

    def write(self, vector_store_name: str, entries_to_write: List[Dict],
              multiprocessing_lock: Optional[multiprocessing.Lock],
              overwrite: multiprocessing.Value):
        self.vector_stores[vector_store_name].write(entries_to_write, multiprocessing_lock, overwrite)

    def cleanup(self):
        for vector_store in self.vector_stores.values():
            vector_store.cleanup()

    def sample(self, vector_store_name: str, n: int):
        return self.vector_stores[vector_store_name].sample(n=n)

    def get_data(self, vector_store_name: str, rows: List[int], columns: List[str]) -> pd.DataFrame:
        return self.vector_stores[vector_store_name].get_data(rows=rows, columns=columns)

    def add_global_ids(self, groups: Groups):
        for vector_store_name in self.vector_stores:
            self.vector_stores[vector_store_name].add_global_ids(groups)

    def count_spectra_vector_store(self, vector_store_name: str):
        return self.vector_stores[vector_store_name].count_spectra()

    def count_spectra(self):
        count = 0
        for vector_store in self.vector_stores.values():
            count += vector_store.count_spectra()
        return count
