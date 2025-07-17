import json
import multiprocessing
import os
import time
from collections import defaultdict
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Generator, Any

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from joblib import Parallel, delayed
from lance import LanceDataset
from numpy import ndarray

from ..environment_variables import TREEMS2_NUM_CPUS
from ..groups.groups import Groups
from ..logger_config import get_logger
from ..utils.utils import format_execution_time

# Create a logger for this module
logger = get_logger(__name__)


class VectorStore:
    def __init__(self, name: str, directory: str, vector_dim: int):
        self.dataset_path: str = os.path.join(directory, "spectra.lance")
        self.directory: str = directory
        self.name: str = name
        self.vector_dim: int = vector_dim

        self.vector_count: int = 0
        self.group_counts = defaultdict(int)

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        self.schema: pa.Schema = pa.schema([
            pa.field("spectrum_id", pa.uint16()),
            pa.field("file_id", pa.uint16()),
            pa.field("group_id", pa.uint16()),
            # pa.field("identifier", pa.string()),
            pa.field("precursor_mz", pa.float32()),
            pa.field("precursor_charge", pa.int8()),
            # pa.field("mz", pa.list_(pa.float32())),
            # pa.field("intensity", pa.list_(pa.float32())),
            pa.field("retention_time", pa.float32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim))            ,
        ])

    def _get_dataset(self) -> Optional[LanceDataset]:
        try:
            return lance.dataset(self.dataset_path)
        except (ValueError, FileNotFoundError):
            return None

    def cleanup(self) -> None:
        ds = self._get_dataset()
        if ds is None:
            return
        try:
            ds.optimize.compact_files(target_rows_per_fragment=1024 * 1024)
            time.sleep(0.1)
            ds.cleanup_old_versions(older_than=timedelta(microseconds=1))
        except ValueError as e:
            logger.warning(f"Could not cleanup Lance dataset at '{self.dataset_path}': {e}")

    def write(self, entries_to_write: List[Dict], use_incremental_compaction: bool,
              multiprocessing_lock: Optional[multiprocessing.Lock],
              overwrite: multiprocessing.Value) -> None:
        new_rows = pa.Table.from_pylist(entries_to_write, self.schema)
        with multiprocessing_lock:
            if overwrite.value:
                ds = lance.write_dataset(new_rows, self.dataset_path, mode="overwrite", data_storage_version="stable")
                overwrite.value = False
            else:
                ds = lance.write_dataset(new_rows, self.dataset_path, mode="append")

            if use_incremental_compaction:
                if VectorStore.should_compact(ds):
                    VectorStore.compact_and_remove_prev_versions(ds)

    @staticmethod
    def should_compact(lance_ds, fragment_threshold: int = 512, small_file_threshold: int = 512) -> bool:
        dataset_stats = lance_ds.stats.dataset_stats()
        return (
                dataset_stats["num_fragments"] > fragment_threshold or
                dataset_stats["num_small_files"] > small_file_threshold
        )

    @staticmethod
    def compact_and_remove_prev_versions(lance_ds):
        lance_ds.optimize.compact_files(target_rows_per_fragment=1024 * 1024)
        lance_ds.cleanup_old_versions(older_than=timedelta(microseconds=1))

    @staticmethod
    def sample_chunk(ds, indices):
        table = ds.take(indices, columns=["vector"])
        vectors = table["vector"].to_numpy()
        return np.vstack(vectors)

    def parallel_sample(self, n_samples):
        ds = self._get_dataset()
        if ds is None:
            return np.empty((0, self.vector_dim), dtype=np.float32)
        total_rows = ds.count_rows()

        max_workers = int(os.environ.get(TREEMS2_NUM_CPUS, multiprocessing.cpu_count()))

        # Generate all random indices
        indices = np.random.choice(total_rows, size=n_samples, replace=False)
        indices.sort()

        # Split indices into n_jobs chunks as evenly as possible
        chunks = np.array_split(indices, max_workers)

        # Fetch in parallel
        results = Parallel(n_jobs=max_workers, backend="loky")(
            delayed(VectorStore.sample_chunk)(ds, chunk) for chunk in chunks
        )

        return np.vstack(results)

    def sample(self, n: int) -> ndarray:
        ds = self._get_dataset()
        if ds is None:
            return np.empty((0, self.vector_dim), dtype=np.float32)

        max_workers = int(os.environ.get(TREEMS2_NUM_CPUS, multiprocessing.cpu_count()))


        # Step 1: Sample from Lance dataset
        t = time.time()
        #r1 r1 = ds.sample(n, columns=["vector"])

        indices = np.random.randint(0, ds.count_rows(), size=n)
        r1 = ds.take(indices, columns=["vector"])

        logger.info(f"Sampled from lance in {format_execution_time(time.time() - t)}")
        logger.info(f"Shape after sampling: {len(r1)} rows")

        # Step 2: Convert to NumPy
        t = time.time()
        r1 = r1["vector"].to_numpy()
        logger.info(f"Converted pyarrow to numpy in {format_execution_time(time.time() - t)}")
        logger.info(f"Shape after conversion: {r1.shape}")  # Expect a 1D array of objects

        # Step 3: Stack arrays into single 2D NumPy array
        t = time.time()
        r1 = np.vstack(r1)
        logger.info(f"Stacked in {format_execution_time(time.time() - t)}")
        logger.info(f"Final stacked shape: {r1.shape}")  # Should be (n, vector_dim)

        return r1

        # return np.vstack(ds.sample(n, columns=["vector"])["vector"].to_numpy())

    def to_vector_batches(self, batch_size: int) -> Generator[Tuple[ndarray, ndarray, int], None, None]:
        ds = self._get_dataset()
        if ds is None:
            return
        first = 0
        for batch in ds.to_batches(columns=["vector"], batch_size=batch_size):
            df = batch.to_pandas()
            vectors = np.stack(df["vector"].to_numpy())
            ids = np.arange(start=first, stop=first + batch.num_rows, dtype=np.int64)
            first += batch.num_rows
            yield vectors, ids, batch.num_rows

    def get_data(self, rows: List[int], columns: List[str]) -> pd.DataFrame:
        ds = self._get_dataset()
        if ds is None:
            return pd.DataFrame(columns=columns)
        return ds.take(indices=rows, columns=columns).to_pandas()

    def get_col(self, column) -> pd.DataFrame:
        ds = self._get_dataset()
        if ds is None:
            return pd.DataFrame(columns=column)
        return ds.to_table(columns=[column]).to_pandas()

    def add_global_ids(self, groups: Groups) -> None:
        def compute_global_id(row: pd.Series) -> int:
            offset = groups.get_group(row['group_id']).get_peak_file(row['file_id']).begin
            return offset + row["spectrum_id"]

        @lance.batch_udf()
        def add_global_ids_batch(batch: pa.RecordBatch) -> pd.DataFrame:
            global_ids = batch.to_pandas().apply(compute_global_id, axis=1).astype(np.int32)
            return pd.DataFrame({"global_id": global_ids}, dtype=np.int32)

        ds = self._get_dataset()
        if ds is None:
            return
        ds.add_columns(add_global_ids_batch)

    def _count_vectors(self) -> int:
        ds = self._get_dataset()
        return ds.count_rows() if ds else 0

    def update_vector_count(self) -> int:
        self.vector_count = self._count_vectors()
        return self.vector_count

    def update_group_count(self, group_id: int, nr_vectors: int):
        self.group_counts[group_id] += nr_vectors

    def is_empty(self) -> bool:
        return self._count_vectors() == 0

    def clear(self) -> None:
        ds = self._get_dataset()
        if ds:
            ds.delete("TRUE")

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "directory": self.directory,
            "vector_dim": self.vector_dim,
            "vector_count": self.vector_count,
            "group_counts": dict(self.group_counts),
        }

    def save(self, path: str) -> str:
        """
        Write the metadata for the vector store to a JSON-file.
        :param path: The path to which the JSON file will be written.
        :return:
        """
        with open(path, "w") as json_file:
            json.dump(self._to_dict(), json_file, indent=4)
        return path

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Optional["VectorStore"]:
        try:
            vector_store = cls(name=data["name"], directory=data["directory"], vector_dim=data["vector_dim"])
            vector_store.vector_count = data["vector_count"]
            group_counts = {int(k): v for k, v in data["group_counts"].items()}  # convert str keys back to integers
            vector_store.group_counts = defaultdict(int, group_counts)
            return vector_store
        except (KeyError, TypeError, AttributeError):
            return None  # Return None if the data structure is incorrect

    @classmethod
    def load(cls, path: str) -> Optional["VectorStore"]:
        """
        Loads a vector store from a JSON-file.
        :param path: The path to the JSON-file.
        :return: a vector store if it can be loaded correctly from the JSON-file, None otherwise.
        """
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
            return cls._from_dict(data)
        except (json.JSONDecodeError, OSError, PermissionError):
            return None  # Return None if the file is unreadable or corrupted
