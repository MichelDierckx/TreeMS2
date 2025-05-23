import json
import multiprocessing
import os
from typing import Optional, Dict, List, Union, Any

import pandas as pd

from TreeMS2.groups.groups import Groups
from TreeMS2.logger_config import get_logger
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class VectorStoreManager:
    def __init__(self, vector_stores: Dict[str, VectorStore]):

        # self.vector_stores = {
        #     name: VectorStore(name=name, directory=os.path.join(path, name), vector_dim=vector_dim)
        #     for name in vector_store_names
        # }
        self.vector_stores = vector_stores
        self.vector_count = 0

    def create_locks_and_flags(self, manager: multiprocessing.Manager) -> Dict[
        str, Dict[str, Union[multiprocessing.Lock, multiprocessing.Value]]]:
        """Returns a dictionary containing locks and overwrite flags for each vector store."""
        return {
            name: {
                "lock": manager.Lock(),
                "overwrite": manager.Value("b", True)
            }
            for name in self.vector_stores
        }

    def write(self, vector_store_name: str, entries_to_write: List[Dict], use_incremental_compaction: bool,
              multiprocessing_lock: Optional[multiprocessing.Lock],
              overwrite: multiprocessing.Value):
        self.vector_stores[vector_store_name].write(entries_to_write, use_incremental_compaction, multiprocessing_lock,
                                                    overwrite)

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

    def update_vector_count(self):
        for vector_store in self.vector_stores.values():
            self.vector_count += vector_store.update_vector_count()
        return self.vector_count

    def update_group_count(self, vector_store_name: str, group_id: int, nr_vectors: int):
        self.vector_stores[vector_store_name].update_group_count(group_id, nr_vectors)

    def clear(self):
        for vector_store in self.vector_stores.values():
            vector_store.clear()

    def save(self, parent_dir: str, filename: str):
        """
        Write the metadata for the vector store to a JSON-file.
        :param filename: The name of the file to which the metadata is written.
        :param parent_dir: The directory in which the JSON file will be written.
        :return:
        """
        os.makedirs(parent_dir, exist_ok=True)
        d = {
            "vector_count": self.vector_count,
            "vector_stores": {
                name: store.save(os.path.join(parent_dir, f"vector_store_{name}.json"))
                for name, store in self.vector_stores.items()
            },
        }
        with open(os.path.join(parent_dir, filename), "w") as json_file:
            json.dump(d, json_file, indent=4)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Optional["VectorStoreManager"]:
        try:
            vector_stores = {}
            for name, uri in data["vector_stores"].items():
                vector_store = VectorStore.load(path=uri)
                if vector_store is not None:
                    vector_stores[name] = vector_store
                else:
                    return None

            vector_store_manager = cls(vector_stores=vector_stores)
            vector_store_manager.vector_count = data["vector_count"]
            return vector_store_manager
        except (KeyError, TypeError, AttributeError):
            return None  # Return None if the data structure is incorrect

    @classmethod
    def load(cls, path: str) -> Optional["VectorStoreManager"]:
        """
        Loads a vector store manager from a JSON-file.
        :param path: The path to the JSON-file.
        :return: a vector store manager if it can be loaded correctly from the JSON-file, None otherwise.
        """
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
            return cls._from_dict(data)
        except (json.JSONDecodeError, OSError, PermissionError):
            return None  # Return None if the file is unreadable or corrupted
