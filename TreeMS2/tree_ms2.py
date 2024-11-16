import os
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import joblib

from TreeMS2.config.config_factory import ConfigFactory
from TreeMS2.groups.groups import Groups
from TreeMS2.lance.lance_dataset_manager import LanceDatasetManager
from TreeMS2.peak_file.peak_file import PeakFile
from TreeMS2.spectrum.group_spectrum import GroupSpectrum
from TreeMS2.spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline, ProcessingPipelineFactory
from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import DimensionalityReducer
from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner
from TreeMS2.spectrum.spectrum_vectorization.spectrum_vectorizer import SpectrumVectorizer
from logger_config import get_logger

logger = get_logger(__name__)


class TreeMS2:
    def __init__(self, config_factory: ConfigFactory):
        self.config_factory = config_factory
        self.max_spectra_in_memory = 1_000_000
        logger.debug(f"max_spectra_in_memory = {self.max_spectra_in_memory}")

        self.lance_dataset_manager = self._setup_lance_dataset_manager()
        self.vectorizer = self._setup_vectorizer()
        self.processing_pipeline = self._setup_spectrum_processing_pipeline(min_mz=self.vectorizer.binner.min_mz,
                                                                            max_mz=self.vectorizer.binner.max_mz)

    def run(self):
        groups = self._read_group_data()
        self._read_and_process_spectra(groups)

    def _setup_lance_dataset_manager(self) -> LanceDatasetManager:
        config = self.config_factory.create_output_config()
        lance_dataset_path = os.path.join(config.work_dir, "spectra", f"spectra.lance")
        return LanceDatasetManager(lance_dataset_path)

    def _setup_spectrum_processing_pipeline(self, min_mz: float, max_mz: float) -> SpectrumProcessingPipeline:
        config = self.config_factory.create_spectrum_processing_config()
        processing_pipeline = ProcessingPipelineFactory.create_pipeline(config=config, min_mz=min_mz, max_mz=max_mz)
        return processing_pipeline

    def _setup_vectorizer(self):
        config = self.config_factory.create_vectorization_config()
        binner = SpectrumBinner(config.min_mz, config.max_mz, config.fragment_tol)
        reducer = DimensionalityReducer(low_dim=config.low_dim, high_dim=binner.dim)
        spectrum_vectorizer = SpectrumVectorizer(binner=binner, reducer=reducer)
        return spectrum_vectorizer

    def _read_group_data(self) -> Groups:
        # Produce config related to groups configuration
        groups_config = self.config_factory.create_groups_config()
        # A groups object containing information about groups and associated peak files
        groups = Groups.from_file(groups_config.sample_to_group_file)  # read information from file
        return groups

    def _read_and_process_spectra(self, groups: Groups):
        """
        Read and process spectra using multi-threading.
        """
        max_file_workers = min(groups.get_nr_files(), os.cpu_count())
        spectra_queue: queue.Queue[Optional[GroupSpectrum]] = queue.Queue(maxsize=self.max_spectra_in_memory)

        with ThreadPoolExecutor(max_workers=max_file_workers) as executor:
            # Launch writer threads
            for _ in range(max_file_workers):
                executor.submit(self._vectorize_and_write_spectra, spectra_queue)

            low_quality_counter = 0
            failed_to_parse_counter = 0
            total_spectra_counter = 0

            all_files = [
                file
                for group in groups.get_groups()
                for file in group.get_peak_files()
            ]

            logger.info(f"Processing spectra from {len(all_files)} files...")

            for result in joblib.Parallel(n_jobs=max_file_workers)(
                    joblib.delayed(TreeMS2._read_spectra)(file, self.processing_pipeline)
                    for file in all_files):
                file_spectra, file_failed_parsed, file_failed_processed, file_total_spectra, file_id, file_group_id = result
                groups.get_group(file_group_id).get_peak_file(file_id).failed_parsed = file_failed_parsed
                groups.get_group(file_group_id).failed_processed = file_failed_processed
                groups.get_group(file_group_id).total_spectra = file_total_spectra

                low_quality_counter += file_failed_processed
                failed_to_parse_counter += file_failed_parsed
                total_spectra_counter += file_total_spectra

                for spec in file_spectra:
                    spectra_queue.put(spec)

            # Add sentinels to signal end of work
            for _ in range(max_file_workers):
                spectra_queue.put(None)

        logger.info(
            f"Processed {total_spectra_counter} spectra from {len(all_files)} files:\n"
            f"  - {low_quality_counter} spectra were filtered out as low quality.\n"
            f"  - {failed_to_parse_counter} spectra could not be parsed.\n"
            f"  - {total_spectra_counter - low_quality_counter - failed_to_parse_counter} spectra were successfully written to the dataset."
        )

    @staticmethod
    def _read_spectra(file: PeakFile, processing_pipeline: SpectrumProcessingPipeline):
        spectra = list(file.get_spectra(processing_pipeline))
        # Return spectra, counters, group_id and file_id
        return spectra, file.failed_parsed, file.failed_processed, file.total_spectra, file.get_id(), file.get_group_id()

    def _vectorize_and_write_spectra(self, spectra_queue: queue.Queue[Optional[GroupSpectrum]]):
        spec_to_write = []

        def _process_batch():
            if not spec_to_write:
                return
            specs = [spec.spectrum for spec in spec_to_write]
            vectors = self.vectorizer.vectorize(specs)
            dict_list = [
                {**spec.to_dict(), "vector": vector}
                for spec, vector in zip(spec_to_write, vectors)
            ]
            self.lance_dataset_manager.write_to_dataset(dict_list)
            spec_to_write.clear()

        while True:
            spectrum = spectra_queue.get()
            if spectrum is None:
                _process_batch()
                return

            spec_to_write.append(spectrum)
            if len(spec_to_write) >= 10_000:
                _process_batch()
