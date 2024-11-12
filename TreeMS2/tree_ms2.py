import functools
import logging
import multiprocessing
import os
import queue
import sys
from typing import Union, List, Callable, Dict, Tuple

import joblib
import lance
import numpy as np
import pyarrow as pa
from sklearn.random_projection import SparseRandomProjection

from .cluster import spectrum
from .config import config, Config
from .metadata_readers.metadata_reader_manager import MetadataReaderManager
from .peak_file_readers.peak_file_reader_manager import PeakFileReaderManager

logger = logging.getLogger("TreeMS2")


def main(args: Union[str, List[str]] = None) -> int:
    config.parse(args)  # Parse arguments from config file or command-line

    # compute the dimensionality to which the spectra get reduced
    vec_len, min_mz, max_mz = spectrum.get_dim(
        config.min_mz, config.max_mz, config.fragment_tol
    )

    process_spectrum = functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=min_mz,
        mz_max=max_mz,
        remove_precursor_tolerance=config.remove_precursor_tol,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=None if config.scaling == "off" else config.scaling,
    )

    transformation = (
        SparseRandomProjection(config.low_dim, random_state=0)
        .fit(np.zeros((1, vec_len)))
        .components_.astype(np.float32)
        .T
    )
    vectorize = functools.partial(
        spectrum.to_vector,
        transformation=transformation,
        min_mz=min_mz,
        bin_size=config.fragment_tol,
        dim=vec_len,
        norm=True,
    )

    dataset_path = _prepare_spectra(process_spectrum, vectorize)

    return 0


def configure_logging():
    """
    Configures logging to capture warnings, set log levels, and format output.
    """
    # Capture warnings and display them as log messages
    logging.captureWarnings(True)

    # Set up the root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Create a stream handler to output logs to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)

    # Set a formatter for the log messages
    handler.setFormatter(
        logging.Formatter(
            "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
            "{message}",
            style="{",
        )
    )

    # Add the handler to the root logger
    root.addHandler(handler)

    # Disable non-critical log messages from certain dependencies
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)


def _prepare_spectra(process_spectrum: Callable, vectorize: Callable) -> str:
    """
    Read the spectra from the input peak files and partition to intermediate
    files split and sorted by precursor m/z.

    Parameters
    ----------
    process_spectrum : Callable
        The function to process the spectra.

    Returns
    -------
    str
        The path to the lance dataset
    """
    # read metadata from file, must include data that associates peak files with groups
    metadata_file_extension = os.path.splitext(config[Config.SAMPLE_TO_GROUP_FILE])[1]
    metadata_reader_manager = MetadataReaderManager()
    metadata_reader = metadata_reader_manager.get_reader(metadata_file_extension)
    peak_file_group_mapping = metadata_reader.get_metadata(config[Config.SAMPLE_TO_GROUP_FILE])
    logger.info(
        f"Found {peak_file_group_mapping.nr_groups} groups and {peak_file_group_mapping.nr_files} peak files in '{config[Config.SAMPLE_TO_GROUP_FILE]}'.")

    # Use multiple worker processes to read the peak files.
    max_file_workers = min(peak_file_group_mapping.nr_files, multiprocessing.cpu_count())

    max_spectra_in_memory = 1_000_000
    spectra_queue = queue.Queue(maxsize=max_spectra_in_memory)
    # Start the lance writers.
    lance_lock = multiprocessing.Lock()
    schema = pa.schema(
        [
            pa.field("identifier", pa.string()),
            pa.field("precursor_mz", pa.float32()),
            pa.field("precursor_charge", pa.int8()),
            pa.field("mz", pa.list_(pa.float32())),
            pa.field("intensity", pa.list_(pa.float32())),
            pa.field("retention_time", pa.float32()),
            pa.field("filename", pa.string()),
            pa.field("group_id", pa.uint16()),  # group id should be in range [0, 65535]
            pa.field("vector", pa.list_(pa.float32())),
        ]
    )
    lance_writers = multiprocessing.pool.ThreadPool(
        max_file_workers,
        _write_spectra_lance,
        (spectra_queue, lance_lock, schema, vectorize),
    )
    # Read the peak files and put their spectra in the queue for consumption
    # by the lance writers.
    low_quality_counter = 0
    for file_spectra, lqc in joblib.Parallel(n_jobs=max_file_workers)(
            joblib.delayed(_read_spectra)(config.ms2_dir, file, group_id, process_spectrum)
            for file, group_id in peak_file_group_mapping.sample_to_group_id.items()
    ):
        low_quality_counter += lqc
        for spec in file_spectra:
            spectra_queue.put(spec)
    # Add sentinels to indicate stopping.
    for _ in range(max_file_workers):
        spectra_queue.put(None)
    lance_writers.close()
    lance_writers.join()

    # Count the total number of spectra in the datasets.
    n_spectra = 0
    dataset_path = os.path.join(config.work_dir, "spectra", f"spectra.lance")
    try:
        dataset = lance.dataset(dataset_path)
        n_spectra = dataset.count_rows()
    except ValueError:
        logger.error(f"Failed to create lance dataset %d")
    logger.info(
        "Read %d spectra from %d peak files", n_spectra, peak_file_group_mapping.nr_files
    )
    logger.info("Skipped %d low-quality spectra", low_quality_counter)

    return dataset_path


def _read_spectra(
        ms2_dir: str,
        filename: str,
        group_id: int,
        process_spectrum: Callable,
) -> Tuple[List[Dict[str, Union[str, float, int, np.ndarray]]], int]:
    """
    Get the spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.
    process_spectrum : Callable
        The function to process the spectra.

    Returns
    -------
    Tuple[List[Dict[str, Union[str, float, int, np.ndarray]]], int]
        The spectra read from the given file as a list of dictionaries and
        the number of low-quality spectra.
    """
    low_quality_counter = 0
    spectra = []
    file_extension = os.path.splitext(filename)[1]
    file_path = os.path.join(ms2_dir, filename)

    peak_file_reader_manager = PeakFileReaderManager()
    peak_file_reader = peak_file_reader_manager.get_reader(file_extension)

    for spec in peak_file_reader.get_spectra(file_path):
        spec.filename = filename
        spec.group_id = group_id
        spec = process_spectrum(spec)
        if spec is None:
            low_quality_counter += 1
        else:
            spectra.append(spec)
    return spectra, low_quality_counter


def _write_spectra_lance(
        spectra_queue: queue.Queue,
        lance_lock: multiprocessing.synchronize.Lock,
        schema: pa.Schema,
        vectorize: Callable
) -> None:
    """
    Read spectra from a queue and write to a lance dataset.

    Parameters
    ----------
    spectra_queue : queue.Queue
        Queue from which to read spectra for writing to pickle files.
    lance_lock : multiprocessing.synchronize.Lock
        Lock to synchronize writing to the dataset.
    schema : pa.Schema
        The schema of the dataset.
    """
    spec_to_write = []
    while True:
        spec = spectra_queue.get()
        if spec is None:
            if len(spec_to_write) == 0:
                return
            _write_to_dataset(
                spec_to_write,
                lance_lock,
                schema,
                config.work_dir,
                vectorize
            )
            spec_to_write.clear()
            return
        spec_to_write.append(spec)
        if len(spec_to_write) >= 10_000:
            _write_to_dataset(
                spec_to_write,
                lance_lock,
                schema,
                config.work_dir,
                vectorize
            )
            spec_to_write.clear()


def _write_to_dataset(
        spec_to_write: List,
        lock: multiprocessing.synchronize.Lock,
        schema: pa.Schema,
        work_dir: str,
        vectorize: Callable
) -> int:
    """
    Write a list of spectra to a lance dataset.

    Parameters
    ----------
    spec_to_write : List[Dict]
        The spectra to write.
    lock : multiprocessing.Lock
        Lock to synchronize writing to the dataset.
    schema : pa.Schema
        The schema of the dataset.
    work_dir : str
        The directory in which the dataset is stored.
    Returns
    -------
    int
        The number of spectra written to the dataset.
    """
    # Vectorize the spectra and add them to the dictionary.
    vectors = vectorize(spec_to_write)
    for i, vector in enumerate(vectors):
        spec_to_write[i]["vector"] = vector

    # Write the spectra to the dataset.
    new_rows = pa.Table.from_pylist(spec_to_write, schema)
    path = os.path.join(work_dir, "spectra", f"spectra.lance")
    with lock:
        if not os.path.exists(path):
            _create_lance_dataset(schema)
        lance.write_dataset(new_rows, path, mode="append")
    return len(new_rows)


def _create_lance_dataset(
        schema: pa.Schema
) -> lance.LanceDataset:
    """
    Create a lance dataset.

    Parameters
    ----------
    schema : pa.Schema
        The schema of the dataset.

    Returns
    -------
    lance.LanceDataset
        The lance dataset.
    """
    lance_path = os.path.join(
        config.work_dir, "spectra", f"spectra.lance"
    )
    dataset = lance.write_dataset(
        pa.Table.from_pylist([], schema),
        lance_path,
        mode="overwrite",
        data_storage_version="stable",
    )
    logger.debug("Creating lance dataset at %s", lance_path)
    return dataset
