import glob
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
    def __init__(self, work_dir: str):
        self.base_path = os.path.join(work_dir, "spectra.lance")
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
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

    def delete_old_datasets(self):
        logger.warning(f"Deleting existing datasets in '{self.base_path}'")
        # Get all subdirectories starting with 'group'
        group_dirs = [d for d in os.listdir(self.base_path) if
                      os.path.isdir(os.path.join(self.base_path, d)) and d.startswith("group")]

        for group_dir in group_dirs:
            group_path = os.path.join(self.base_path, group_dir)

            # Find all the relevant files (.lance, .txn, .manifest) within the "group" directory
            files_to_delete = glob.glob(os.path.join(group_path, 'data', '*.lance')) + \
                              glob.glob(os.path.join(group_path, '_transactions', '*.txn')) + \
                              glob.glob(os.path.join(group_path, '_versions', '*.manifest'))

            for file_path in files_to_delete:
                try:
                    os.remove(file_path)  # Safely remove the file
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")

            # Check if the group directory is now empty after file deletion
            # Ensure that we check if the data, _transactions, and _versions directories are empty
            subdirs = ['data', '_transactions', '_versions']
            is_empty = True

            for subdir in subdirs:
                subdir_path = os.path.join(group_path, subdir)
                # Check if the subdirectory is empty (ignore it if it's not actually a directory)
                if os.path.isdir(subdir_path) and os.listdir(subdir_path):  # Directory is not empty
                    is_empty = False
                    break

            # If the group directory is empty and only contains the empty subdirectories, delete the group directory
            if is_empty and not os.listdir(group_path):  # Group directory is empty
                try:
                    os.rmdir(group_path)  # Remove the group directory
                except Exception as e:
                    logger.error(f"Failed to delete dataset for group {group_path}: {e}")

    def create_dataset(self, group_id: int):
        dataset_path = self.get_group_path(group_id)
        dataset = lance.write_dataset(
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
