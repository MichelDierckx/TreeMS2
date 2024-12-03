import os
import random
import threading
from collections import defaultdict
from datetime import timedelta
from typing import List, Dict, DefaultDict, Tuple

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from numpy import ndarray

from ..groups.group import Group
from ..groups.groups import Groups
from ..logger_config import get_logger
from ..utils.utils import partition_pylist

# Create a logger for this module
logger = get_logger(__name__)


class VectorStore:
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

    def cleanup(self):
        for group_id in self.datasets.keys():
            path = self.get_group_path(group_id)
            try:
                ds = lance.dataset(path)
                time_delta = timedelta(microseconds=1)
                ds.cleanup_old_versions(older_than=time_delta)
            except ValueError:
                return

    def get_group_path(self, group_id: int) -> str:
        """Return the path for the group's dataset."""
        return os.path.join(self.base_path, f"group{group_id}")

    def write_to_group(self, group_id: int, entries_to_write: List[Dict]):
        """Write data to a specific group's dataset."""
        group_path = self.get_group_path(group_id)
        os.makedirs(group_path, exist_ok=True)
        new_rows = pa.Table.from_pylist(entries_to_write, self.schema)
        with self.locks[group_id]:
            if group_id in self.datasets:
                lance.write_dataset(new_rows, group_path, mode="append")
                self.datasets[group_id] += len(entries_to_write)
            else:
                lance.write_dataset(
                    new_rows,
                    group_path,
                    mode="overwrite",
                    data_storage_version="stable",
                )
                self.datasets[group_id] += len(entries_to_write)
                logger.debug(f"Creating lance dataset at '{group_path}'.")

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

        partitions = partition_pylist(indices, partition_limits)

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
            df['global_id'] = df.apply(lambda x: group.get_global_id(x.file_id, x.id), axis=1)
            vectors = np.stack(df["vector"].to_numpy())
            ids = np.stack(df["global_id"].to_numpy())
            yield vectors, ids, batch.num_rows

    def get_metadata(self, group_id: int, global_spectrum_id: int, groups: Groups, column: str):
        columns = [column]
        row = _get_row(group_id=group_id, global_spectrum_id=global_spectrum_id, groups=groups)
        ds = lance.dataset(self.get_group_path(group_id=group_id))
        ta = ds.take([row], columns=columns)
        return ta[0][0]


def _get_row(group_id: int, global_spectrum_id: int, groups: Groups) -> int:
    """
    Retrieve the corresponding group and row index in the vector store given a global spectrum id and group information.
    :param group_id: the id of the group for which the row index should be computed
    :param global_spectrum_id: the global spectrum id
    :param groups: a groups instance, used to retrieve parsing information
    :return: the row index in the vector store corresponding to the global spectrum id
    """
    group = groups.get_group(group_id)

    invalid_group_spectra = 0
    for file in group.get_peak_files():
        if file.begin() <= global_spectrum_id <= file.end():
            for invalid_spectrum in file.filtered:
                if invalid_spectrum < global_spectrum_id:
                    invalid_group_spectra += 1
            row_nr = global_spectrum_id - invalid_group_spectra
            return row_nr
        else:
            invalid_group_spectra += file.failed_parsed + file.failed_read
    raise ValueError(f"Global id {global_spectrum_id} does not belong to any file.")
