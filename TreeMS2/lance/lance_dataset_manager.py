import os
from collections import defaultdict
from threading import Lock
from typing import List, Dict

import lance
import pyarrow as pa

from ..logger_config import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class LanceDatasetManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
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
        self.locks = defaultdict(Lock)  # locking mechanism per dataset

    def create_dataset(self, group_id: int):
        dataset_path = self.get_group_path(group_id)
        lance.write_dataset(
            pa.Table.from_pylist([], self.schema),
            dataset_path,
            mode="overwrite",
            data_storage_version="stable",
        )
        logger.debug(f"Creating lance dataset at '{dataset_path}'.")

    def create_datasets(self, group_ids: List[int]):
        for group_id in group_ids:
            self.create_dataset(group_id)

    def get_group_path(self, group_id: int) -> str:
        """Return the path for the group's dataset."""
        return os.path.join(self.base_path, f"group{group_id}")

    def write_to_group(self, group_id: int, entries_to_write: List[Dict]):
        """Write data to a specific group's dataset."""
        group_path = self.get_group_path(group_id)
        os.makedirs(group_path, exist_ok=True)
        with self.locks[group_id]:
            new_rows = pa.Table.from_pylist(entries_to_write, self.schema)
            lance.write_dataset(new_rows, group_path, mode="append")
