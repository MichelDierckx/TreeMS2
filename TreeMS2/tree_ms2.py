import multiprocessing
import os
import queue
from typing import Optional

import joblib

from TreeMS2.config.config_factory import ConfigFactory
from TreeMS2.groups.groups import Groups
from TreeMS2.lance.lance_dataset_manager import LanceDatasetManager
from TreeMS2.peak_file.peak_file import PeakFile
from TreeMS2.spectrum.group_spectrum import GroupSpectrum
from TreeMS2.spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline, ProcessingPipelineFactory
from TreeMS2.spectrum.spectrum_vectorization.spectrum_vectorizer import SpectrumVectorizer
from logger_config import get_logger

logger = get_logger(__name__)


class TreeMS2:
    def __init__(self, config_factory: ConfigFactory):
        self.config_factory = config_factory
        self.max_spectra_in_memory = 1_000_000
        logger.debug(f"max_spectra_in_memory = {self.max_spectra_in_memory}")
        self.lance_dataset_manager = self._create_lance_dataset_manager()

    def run(self):
        groups = self._read_group_data()
        self._read_and_process_spectra(groups)

    def _create_lance_dataset_manager(self) -> LanceDatasetManager:
        output_config = self.config_factory.create_output_config()
        lance_dataset_path = os.path.join(output_config.work_dir, "spectra", f"spectra.lance")
        return LanceDatasetManager(lance_dataset_path)

    def _read_group_data(self) -> Groups:
        # Produce config related to groups configuration
        groups_config = self.config_factory.create_groups_config()
        # A groups object containing information about groups and associated peak files
        groups = Groups.from_file(groups_config.sample_to_group_file)  # read information from file
        return groups

    def _read_and_process_spectra(self, groups: Groups):
        # Use multiple worker processes to read the peak files.
        max_file_workers = min(groups.get_nr_files(), multiprocessing.cpu_count())
        spectra_queue: queue.Queue[Optional[GroupSpectrum]] = queue.Queue(maxsize=self.max_spectra_in_memory)
        config = self.config_factory.create_spectrum_processing_config()
        processing_pipeline = ProcessingPipelineFactory.create_pipeline(config=config)

        # note, add vectorization in a function here that first vectorizes and then calls self.lance_data_set_manager.write_spectra
        # while true loop must be moved to this function
        lance_writers = multiprocessing.pool.ThreadPool(
            max_file_workers,
            self._vectorize_and_write_spectra,
            spectra_queue,
        )

        low_quality_counter = 0
        failed_to_parse_counter = 0
        total_spectra_counter = 0

        # Flatten all files across all groups
        all_files = [
            file
            for group in groups.get_groups()
            for file in group.get_peak_files()
        ]

        logger.info(
            f"Processing spectra from {len(all_files)} files..."
        )

        for file_spectra, file_failed_parsed, file_failed_processed, file_total_spectra, file_id, file_group_id in joblib.Parallel(
                n_jobs=max_file_workers)(
            joblib.delayed(TreeMS2._read_spectra)(file, processing_pipeline)
            for file in all_files
        ):
            groups.get_group(file_group_id).get_peak_file(file_id).failed_parsed = file_failed_parsed
            groups.get_group(file_group_id).get_peak_file(file_id).failed_processed = file_failed_processed
            groups.get_group(file_group_id).get_peak_file(file_id).total_spectra = file_total_spectra

            low_quality_counter += file_failed_processed
            failed_to_parse_counter += file_failed_parsed
            total_spectra_counter += file_total_spectra

            for spec in file_spectra:
                spectra_queue.put(spec)
        # Add sentinels to indicate stopping.
        for _ in range(max_file_workers):
            spectra_queue.put(None)

        lance_writers.close()
        lance_writers.join()

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

    def _vectorize_and_write_spectra(self, spectra_queue: queue.Queue[Optional[GroupSpectrum]],
                                     vectorizer: SpectrumVectorizer):
        spec_to_write = []

        while True:
            spec = spectra_queue.get()
            if spec is None:
                if len(spec_to_write) == 0:
                    return
                dict_list = []
                specs = [spec.spectrum for spec in spec_to_write]
                vectors = vectorizer.vectorize(specs)
                for i, vec in enumerate(vectors):
                    spec_to_write[i].vector = vec
                    dict_list.append(spec_to_write[i].to_dict())

                self.lance_dataset_manager.write_to_dataset(
                    dict_list,
                )
                spec_to_write.clear()
                return
            spec_to_write.append(spec)
            if len(spec_to_write) >= 10_000:
                dict_list = []
                specs = [spec.spectrum for spec in spec_to_write]
                vectors = vectorizer.vectorize(specs)
                for i, vec in enumerate(vectors):
                    spec_to_write[i].vector = vec
                    dict_list.append(spec_to_write[i].to_dict())

                self.lance_dataset_manager.write_to_dataset(
                    dict_list,
                )
                spec_to_write.clear()
