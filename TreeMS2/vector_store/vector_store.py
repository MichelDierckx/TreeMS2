import os
import threading
from datetime import timedelta
from typing import List, Dict, Tuple

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from numpy import ndarray

from ..groups.groups import Groups
from ..logger_config import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class VectorStore:
    def __init__(self, path: str):
        self.base_path = path
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.schema = pa.schema(
            [
                pa.field("spectrum_id", pa.uint16()),
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
        self.lock = threading.Lock()  # locking mechanism per dataset
        self.has_data: bool = False

    def cleanup(self):
        try:
            ds = lance.dataset(self.base_path)
            time_delta = timedelta(microseconds=1)
            ds.cleanup_old_versions(older_than=time_delta)
        except ValueError:
            return

    def write(self, entries_to_write: List[Dict]):
        """Write data to a specific group's dataset."""
        os.makedirs(self.base_path, exist_ok=True)
        new_rows = pa.Table.from_pylist(entries_to_write, self.schema)
        with self.lock:
            if self.has_data:
                lance.write_dataset(new_rows, self.base_path, mode="append")
            else:
                lance.write_dataset(
                    new_rows,
                    self.base_path,
                    mode="overwrite",
                    data_storage_version="stable",
                )
                self.has_data = True

    def sample(self, n: int):
        ds = lance.dataset(self.base_path)
        df = ds.sample(num_rows=n, columns=["vector"]).to_pandas()
        samples = np.stack(df["vector"].to_numpy())
        return samples

    def to_vector_batches(self, batch_size: int) -> Tuple[ndarray, ndarray, int]:
        """
        Returns vectors in batch, along with their dataset row ids and the number of vectors in the batch.
        :param batch_size: the maximum number of vectors in the batch
        :return: A triplet: (vectors, vector_ids, number_of_vectors)
        """
        dataset = lance.dataset(self.base_path)
        first = 0
        for batch in dataset.to_batches(columns=["vector"], batch_size=batch_size):
            df = batch.to_pandas()
            vectors = np.stack(df["vector"].to_numpy())
            ids = np.arange(start=first, stop=first + batch.num_rows, dtype=np.uint64)
            first += batch.num_rows
            yield vectors, ids, batch.num_rows

    def get_data(self, rows: List[int], columns: List[str]) -> pd.DataFrame:
        ds = lance.dataset(self.base_path)
        df = ds.take(indices=rows, columns=columns).to_pandas()
        return df

    def add_global_ids(self, groups: Groups):

        def compute_global_id(row):
            offset = groups.get_group(row['group_id']).get_peak_file(row['file_id']).begin
            return offset + row["spectrum_id"]

        @lance.batch_udf()
        def add_global_ids_batch(batch):
            global_ids = batch.to_pandas().apply(compute_global_id, axis=1).astype(np.int32)
            return pd.DataFrame({"global_id": global_ids}, dtype=np.int32)

        ds = lance.dataset(self.base_path)
        ds.add_columns(add_global_ids_batch)

    def count_spectra(self):
        ds = lance.dataset(self.base_path)
        return ds.count_rows()
