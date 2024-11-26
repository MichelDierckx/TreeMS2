import glob
import os
import random
import threading
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from numpy import ndarray

from ..groups.group import Group
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
        self.locks: DefaultDict[int, threading.Lock] = defaultdict(threading.Lock)  # locking mechanism per dataset
        self.datasets: DefaultDict[int, int] = defaultdict(int)  # (group id, nr_rows)

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
            self.datasets[group_id] += len(entries_to_write)
            new_rows = pa.Table.from_pylist(entries_to_write, self.schema)
            lance.write_dataset(new_rows, group_path, mode="append")

    def sample(self, n: int, group_ids: List[int]):
        sorted_group_ids = sorted(group_ids)
        partition_limits = []

        total_rows = 0
        for group_id in sorted_group_ids:
            if group_id not in self.datasets:
                raise ValueError(f"Dataset for group {group_id} does not exist.")

            total_rows += self.datasets[group_id]
            last_index = total_rows - 1
            partition_limits.append(last_index)

        indices = random.sample(range(total_rows), n)
        indices = sorted(indices)

        partitions = _partition_integers(indices, partition_limits)

        data_frames = []
        for group_index, partition in enumerate(partitions):
            dataset = lance.dataset(self.get_group_path(sorted_group_ids[group_index]))
            partition_samples = dataset.take(indices=partition, columns=["vector"]).to_pandas()
            data_frames.append(partition_samples)

        df = pd.concat(data_frames)
        samples = np.stack(df["vector"].to_numpy())
        return samples

    def to_vector_batches(self, batch_size: int, group: Group) -> Tuple[ndarray, ndarray, int]:
        """
        Returns vectors in batch for the given group, along with their global id and the number of vectors in the batch.
        :param batch_size: the maximum number of vectors in the batch
        :param group: the group for which the vectors are retrieved
        :return: A triplet: (vectors, vector_ids, number_of_vectors)
        """
        dataset = lance.dataset(self.get_group_path(group.get_id()))
        for batch in dataset.to_batches(columns=["file_id", "id", "vector"], batch_size=batch_size):
            df = batch.to_pandas()
            df['global_id'] = df.apply(lambda x: group.get_global_id(x.file, x.id), axis=1)
            vectors = np.stack(df["vector"].to_numpy())
            ids = np.stack(df["global_id"].to_numpy())
            yield vectors, ids, batch.num_rows


def _partition_integers(sorted_list: List[int], partition_limits: List[int]) -> List[List[int]]:
    """
    Partitions a sorted list of integers into intervals based on specified partition limits and normalizes
    the values in each partition by subtracting the minimum value of the respective partition.

    The function divides the sorted input list into partitions where each partition is defined by a maximum
    value from `partition_limits`. Each partition contains integers that are within the range
    `[partition_limits[i-1] + 1, partition_limits[i]]` (for partition i, with the first partition starting at 0).
    The minimum value for each partition is subtracted from all integers in that partition.

    Args:
        sorted_list (List[int]): A sorted list of positive integers to be partitioned.
        partition_limits (List[int]): A sorted list of maximum values that define the upper bounds of each partition.
                                      The minimum value for partition 0 is assumed to be 0.
                                      Each partition `i` includes values between `partition_limits[i-1] + 1`
                                      and `partition_limits[i]`.

    Returns:
        List[List[int]]: A list of partitions, where each partition is represented by a list of integers.
                         Each partition contains the normalized values, which are the original values minus the
                         minimum value of the respective partition. The partitions are ordered according to
                         the partition limits.

    Example:
        sorted_list = [3, 8, 15, 20, 25, 30, 40]
        partition_limits = [10, 20, 30]

        result = partition_and_normalize(sorted_list, partition_limits)

        # Output:
        [
            [3, 8],         # Partition for [0-10]
            [4, 9],         # Partition for [11-20]
            [4, 9],         # Partition for [21-30]
            [9]             # Partition for [31+]
        ]
    """
    partitions = []
    partition_index = 0
    partition_min = partition_limits[partition_index - 1] + 1 if partition_index > 0 else 0
    partition_max = partition_limits[partition_index]

    number_of_partitions = len(partition_limits)
    for i in range(number_of_partitions):
        partitions.append([])

    for value in sorted_list:
        while value > partition_max:
            partition_index += 1
            if partition_index > len(partition_limits) - 1:
                return partitions

            partition_min = partition_limits[partition_index - 1] + 1 if partition_index > 0 else 0
            partition_max = partition_limits[partition_index]

        partitions[partition_index].append(value - partition_min)

    return partitions
