import multiprocessing
import queue
from typing import List, Optional

import lance
import pyarrow as pa

from TreeMS2.spectrum.group_spectrum import GroupSpectrum
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
            ]
        )
        self.dataset: Optional[lance.LanceDataset] = None
        self.lock = multiprocessing.Lock()

    def write_spectra(self, spectra_queue: queue.Queue[Optional[GroupSpectrum]]):
        spec_to_write = []

        while True:
            spec = spectra_queue.get()
            if spec is None:
                if len(spec_to_write) == 0:
                    return
                self._write_to_dataset(
                    spec_to_write,
                )
                spec_to_write.clear()
                return
            spec_to_write.append(spec)
            if len(spec_to_write) >= 10_000:
                self._write_to_dataset(
                    spec_to_write,
                )
                spec_to_write.clear()

    def _write_to_dataset(self, spectra_to_write: List[GroupSpectrum]):
        # Write the spectra to the dataset.
        dict_list = [gs.to_dict() for gs in spectra_to_write]
        new_rows = pa.Table.from_pylist(dict_list, self.schema)
        with self.lock:
            if self.dataset is None:
                self._create_lance_dataset()
            lance.write_dataset(new_rows, self.dataset, mode="append")
        return len(new_rows)

    def _create_lance_dataset(self) -> lance.LanceDataset:
        lance_path = self.dataset_path
        dataset = lance.write_dataset(
            pa.Table.from_pylist([], self.schema),
            lance_path,
            mode="overwrite",
            data_storage_version="stable",
        )
        logger.info(f"Creating dataset at '{lance_path}'.")
        return dataset
