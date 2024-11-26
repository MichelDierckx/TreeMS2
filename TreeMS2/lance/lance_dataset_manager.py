from threading import Lock
from typing import List, Optional, Dict

import lance
import pyarrow as pa

from ..logger_config import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class LanceDatasetManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.schema = pa.schema(
            [
                pa.field("id", pa.uint16()),
                pa.field("file_id", pa.uint16()),
                pa.field("group_id", pa.uint16()),
                pa.field("identifier", pa.string()),
                pa.field("precursor_mz", pa.float32()),
                pa.field("precursor_charge", pa.int8()),
                pa.field("mz", pa.list_(pa.float32())),
                pa.field("intensity", pa.list_(pa.float32())),
                pa.field("retention_time", pa.float32()),
                pa.field("vector", pa.list_(pa.float32())),
            ]
        )
        self.dataset: Optional[lance.LanceDataset] = None
        self.lock = Lock()  # Lock to ensure thread-safe access to `self.dataset`

    def write_to_dataset(self, entries_to_write: List[Dict]):
        new_rows = pa.Table.from_pylist(entries_to_write, self.schema)
        with self.lock:
            if self.dataset is None:
                self._create_lance_dataset()
            lance.write_dataset(new_rows, self.dataset.uri, mode="append")
            return len(new_rows)

    def _create_lance_dataset(self):
        lance_path = self.dataset_path
        self.dataset = lance.write_dataset(
            pa.Table.from_pylist([], self.schema),
            lance_path,
            mode="overwrite",
            data_storage_version="stable",
        )
        logger.debug(f"Creating lance dataset at '{lance_path}'.")
