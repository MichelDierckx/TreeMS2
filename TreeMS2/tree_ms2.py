import multiprocessing
import os
import queue
from typing import Optional

import joblib

from TreeMS2.config.config import Config
from TreeMS2.groups.groups import Groups
from TreeMS2.lance.lance_dataset_manager import LanceDatasetManager
from TreeMS2.spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline, ProcessingPipelineFactory
from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import DimensionalityReducer
from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner
from TreeMS2.spectrum.spectrum_vectorization.spectrum_vectorizer import SpectrumVectorizer
from .logger_config import get_logger
from .peak_file.peak_file import PeakFile
from .spectrum.group_spectrum import GroupSpectrum

logger = get_logger(__name__)


class TreeMS2:
    def __init__(self, config_factory: Config):
        # Initialize configuration, set memory limits, and set up required components.
        self.config_factory = config_factory
        self.max_spectra_in_memory = 1_000_000
        logger.debug(f"max_spectra_in_memory = {self.max_spectra_in_memory}")

        # Set up dataset manager, spectrum vectorizer, and processing pipeline
        self.lance_dataset_manager = self._setup_lance_dataset_manager()
        self.vectorizer = self._setup_vectorizer()
        self.processing_pipeline = self._setup_spectrum_processing_pipeline(min_mz=self.vectorizer.binner.min_mz,
                                                                            max_mz=self.vectorizer.binner.max_mz)

    def run(self):
        # Read group information: which groups exist and which peak files belong to the group
        groups = self._read_groups()
        # Reads spectra from the peak files and stores them a lance dataset
        self._read_and_process_spectra(groups)

    def _setup_lance_dataset_manager(self) -> LanceDatasetManager:
        # Create a LanceDatasetManager instance for managing output storage.
        config = self.config_factory.create_output_config()
        lance_dataset_path = os.path.join(config.work_dir, "spectra", f"spectra.lance")
        return LanceDatasetManager(lance_dataset_path)

    def _setup_spectrum_processing_pipeline(self, min_mz: float, max_mz: float) -> SpectrumProcessingPipeline:
        # Create a spectrum processing pipeline using configuration and mass-to-charge range.
        config = self.config_factory.create_spectrum_processing_config()
        processing_pipeline = ProcessingPipelineFactory.create_pipeline(config=config, min_mz=min_mz, max_mz=max_mz)
        return processing_pipeline

    def _setup_vectorizer(self):
        # Initialize the spectrum vectorizer, which combines binning and dimensionality reduction.
        config = self.config_factory.create_vectorization_config()
        binner = SpectrumBinner(config.min_mz, config.max_mz, config.fragment_tol)
        reducer = DimensionalityReducer(low_dim=config.low_dim, high_dim=binner.dim)
        spectrum_vectorizer = SpectrumVectorizer(binner=binner, reducer=reducer)
        return spectrum_vectorizer

    def _read_groups(self) -> Groups:
        # Read group and file information
        groups_config = self.config_factory.create_groups_config()
        groups = Groups.from_file(groups_config.sample_to_group_file)
        return groups

    def _read_and_process_spectra(self, groups: Groups):
        """
        Main function to orchestrate producers and consumers using queues.
        """
        # Use multiple worker processes to read the peak files.
        max_file_workers = min(groups.get_nr_files(), multiprocessing.cpu_count())
        spectra_queue = queue.Queue(maxsize=self.max_spectra_in_memory)

        lance_writers = multiprocessing.pool.ThreadPool(
            max_file_workers,
            self._vectorize_and_write_spectra,
            (spectra_queue,)
        )

        low_quality_counter = 0
        failed_to_parse_counter = 0
        total_spectra_counter = 0

        # Flatten the list of all peak files across groups for processing.
        all_files = [
            file
            for group in groups.get_groups()
            for file in group.get_peak_files()
        ]

        logger.info(f"Processing spectra from {len(all_files)} files ...")

        # Process spectra in parallel for each file using joblib.
        for result in joblib.Parallel(n_jobs=max_file_workers)(
                joblib.delayed(TreeMS2._read_spectra)(file, self.processing_pipeline)
                for file in all_files):
            file_spectra, file_failed_parsed, file_failed_processed, file_total_spectra, file_id, file_group_id = result
            groups.get_group(file_group_id).get_peak_file(file_id).failed_parsed = file_failed_parsed
            groups.get_group(file_group_id).failed_processed = file_failed_processed
            groups.get_group(file_group_id).total_spectra = file_total_spectra

            # Update counters for logging.
            low_quality_counter += file_failed_processed
            failed_to_parse_counter += file_failed_parsed
            total_spectra_counter += file_total_spectra

            # Queue spectra for writing.
            for spec in file_spectra:
                spectra_queue.put(spec)

        # Signal writer threads to stop by adding sentinel values to the queue.
        for _ in range(max_file_workers):
            spectra_queue.put(None)

        lance_writers.close()
        lance_writers.join()

        # Log final processing statistics.
        logger.info(
            f"Processed {total_spectra_counter} spectra from {len(all_files)} files:\n"
            f"\t- {low_quality_counter} spectra were filtered out as low quality.\n"
            f"\t- {failed_to_parse_counter} spectra could not be parsed.\n"
            f"\t- {total_spectra_counter - low_quality_counter - failed_to_parse_counter} spectra were successfully written to the dataset."
        )

    @staticmethod
    def _read_spectra(file: PeakFile, processing_pipeline: SpectrumProcessingPipeline):
        # Extract spectra from the file using the provided processing pipeline.
        spectra = list(file.get_spectra(processing_pipeline))
        return spectra, file.failed_parsed, file.failed_processed, file.total_spectra, file.get_id(), file.get_group_id()

    def _vectorize_and_write_spectra(self, spectra_queue: queue.Queue[Optional[GroupSpectrum]]):
        """
        Consumes spectra, vectorizes, and writes them to the dataset.
        """
        spec_to_write = {}

        def _process_batch():
            """Process and write the collected spectra."""
            for group_id, specs in spec_to_write.items():
                # Extract and vectorize spectra
                spectra = [spec.spectrum for spec in specs]
                vectors = self.vectorizer.vectorize(spectra)
                dict_list = [
                    {**spec.to_dict(), "vector": vector}
                    for spec, vector in zip(specs, vectors)
                ]
                # Write to the Lance dataset for the group
                self.lance_dataset_manager.write_to_group(group_id, dict_list)
            spec_to_write.clear()

        while True:
            spectrum = spectra_queue.get()
            if spectrum is None:
                # Signal to finish processing
                _process_batch()
                break

            # Group spectra by group_id
            group_id = spectrum.get_group_id()
            if group_id not in spec_to_write:
                spec_to_write[group_id] = []
            spec_to_write[group_id].append(spectrum)

            # Process batch if size limit is reached
            if sum(len(specs) for specs in spec_to_write.values()) >= 10_000:
                _process_batch()
