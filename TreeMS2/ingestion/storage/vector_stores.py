import json
import os
from typing import Optional, Dict, List, Any

from TreeMS2.config.logger_config import get_logger
from TreeMS2.ingestion.storage.vector_store import VectorStore

logger = get_logger(__name__)


class VectorStores:
    def __init__(self, vector_stores: Dict[str, VectorStore]):
        self.vector_stores = vector_stores

    def write(
        self,
        vector_store_name: str,
        entries_to_write: List[Dict],
        use_incremental_compaction: bool,
    ):
        self.vector_stores[vector_store_name].write(
            entries_to_write,
            use_incremental_compaction,
        )

    def cleanup(self):
        for vector_store in self.vector_stores.values():
            vector_store.cleanup()

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
            "vector_stores": {
                name: store.save(os.path.join(parent_dir, f"vector_store_{name}.json"))
                for name, store in self.vector_stores.items()
            },
        }
        with open(os.path.join(parent_dir, filename), "w") as json_file:
            json.dump(d, json_file, indent=4)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Optional["VectorStores"]:
        try:
            vector_stores = {}
            for name, uri in data["vector_stores"].items():
                vector_store = VectorStore.load(path=uri)
                if vector_store is not None:
                    vector_stores[name] = vector_store
                else:
                    return None

            vector_store_manager = cls(vector_stores=vector_stores)
            return vector_store_manager
        except (KeyError, TypeError, AttributeError):
            return None  # Return None if the data structure is incorrect

    @classmethod
    def load(cls, path: str) -> Optional["VectorStores"]:
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
