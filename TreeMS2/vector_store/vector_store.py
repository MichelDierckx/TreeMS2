import multiprocessing
import os
import time
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

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
    def __init__(self, base_path: str, name: str):
        self.dataset_path = os.path.join(base_path, name, "spectra.lance")
        self.directory = os.path.join(base_path, name)
        self.name = name
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
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

    @staticmethod
    def load(base_path: str, name: str) -> Optional["VectorStore"]:
        """Loads a VectorStore if possible, otherwise returns None."""
        dataset_path = os.path.join(base_path, name, "spectra.lance")
        if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
            return None
        return VectorStore(base_path, name)

    def cleanup(self):
        try:
            ds = lance.dataset(self.dataset_path)
        except ValueError:
            return
        try:
            ds.optimize.compact_files(target_rows_per_fragment=1024 * 1024)
            time.sleep(0.1)
            time_delta = timedelta(microseconds=1)
            ds.cleanup_old_versions(older_than=time_delta)
        except ValueError as e:
            logger.warning(f"Could not cleanup lance dataset at '{self.dataset_path}': error: {e}")
            return

    def write(self, entries_to_write: List[Dict], multiprocessing_lock: Optional[multiprocessing.Lock],
              overwrite: multiprocessing.Value):
        """Writes entries to the vector store."""
        new_rows = pa.Table.from_pylist(entries_to_write, self.schema)

        with multiprocessing_lock:
            if overwrite.value:
                lance.write_dataset(
                    new_rows,
                    self.dataset_path,
                    mode="overwrite",
                    data_storage_version="stable",
                )
                overwrite.value = False
            else:
                lance.write_dataset(new_rows, self.dataset_path, mode="append")

    def sample(self, n: int):
        ds = lance.dataset(self.dataset_path)
        # df = ds.sample(n, columns=["vector"])["vector"].combine_chunks().flatten().to_numpy().reshape(n, 400)
        return np.vstack(ds.sample(n, columns=["vector"])["vector"].to_numpy())

    def to_vector_batches(self, batch_size: int) -> Tuple[ndarray, ndarray, int]:
        """
        Returns vectors in batch, along with their dataset row ids and the number of vectors in the batch.
        :param batch_size: the maximum number of vectors in the batch
        :return: A triplet: (vectors, vector_ids, number_of_vectors)
        """
        dataset = lance.dataset(self.dataset_path)
        first = 0
        for batch in dataset.to_batches(columns=["vector"], batch_size=batch_size):
            df = batch.to_pandas()
            vectors = np.stack(df["vector"].to_numpy())
            ids = np.arange(start=first, stop=first + batch.num_rows, dtype=np.int64)
            first += batch.num_rows
            yield vectors, ids, batch.num_rows

    def get_data(self, rows: List[int], columns: List[str]) -> pd.DataFrame:
        ds = lance.dataset(self.dataset_path)
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

        ds = lance.dataset(self.dataset_path)
        ds.add_columns(add_global_ids_batch)

    def count_spectra(self):
        ds = lance.dataset(self.dataset_path)
        return ds.count_rows()

    def is_empty(self):
        ds = lance.dataset(self.dataset_path)
        return ds.count_rows() == 0

    def clear(self):
        ds = lance.dataset(self.dataset_path)
        ds.delete("TRUE")
