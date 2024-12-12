import multiprocessing
import queue
from typing import Optional

import joblib

from TreeMS2.config.config import Config
from TreeMS2.groups.groups import Groups
from TreeMS2.spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline, ProcessingPipelineFactory
from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import DimensionalityReducer
from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner
from TreeMS2.spectrum.spectrum_vectorization.spectrum_vectorizer import SpectrumVectorizer
from TreeMS2.vector_store.vector_store import VectorStore
from .distances import Distances
from .groups.peak_file.peak_file import PeakFile
from .index.ms2_index import MS2Index
from .logger_config import get_logger
from .similarity_matrix.pipeline import SimilarityMatrixPipelineFactory
from .similarity_matrix.similarity_matrix import SimilarityMatrix
from .similarity_sets import SimilaritySets
from .spectrum.group_spectrum import GroupSpectrum

logger = get_logger(__name__)


class TreeMS2:
    def __init__(self, config_factory: Config):
        # Initialize configuration, set memory limits, and set up required components.
        self.config_factory = config_factory
        self.max_spectra_in_memory = 1_000_000
        logger.debug(f"max_spectra_in_memory = {self.max_spectra_in_memory}")

        # Set up dataset manager, spectrum vectorizer, and processing pipeline
        self.vector_store = self._setup_vector_store()
        self.vectorizer = self._setup_vectorizer()
        self.processing_pipeline = self._setup_spectrum_processing_pipeline(min_mz=self.vectorizer.binner.min_mz,
                                                                            max_mz=self.vectorizer.binner.max_mz)

    def run(self):
        # retrieve output configuration
        output_config = self.config_factory.create_output_config()
        # Read group information: which groups exist and which peak files belong to the group
        groups = self._read_groups()
        # Reads spectra from the peak files, convert them to low dimension vectors and store to disk
        self._read_and_process_spectra(groups=groups)
        # Update groups information after all spectra have been processed
        groups.update()
        logger.debug(f"{groups}")
        # Write groups information to file
        groups.write_to_file(output_config.work_dir)
        # Add global ids to vector store
        self.vector_store.add_global_ids(groups)
        # Create an index
        index = self._index(groups=groups)
        # Query the index
        similarity_matrix = self._query(index=index)
        # Write similarity matrix to file before filtering
        similarity_matrix.write(work_dir=output_config.work_dir, filename="similarity_matrix_before_filtering")
        similarity_matrix.write_global(work_dir=output_config.work_dir, filename="similarity_matrix_before_filtering",
                                       total_spectra=groups.total_spectra, vector_store=self.vector_store)
        # Compute similarity sets
        similarity_sets = SimilaritySets(similarity_matrix=similarity_matrix, groups=groups)
        # Write similarity sets to file
        similarity_sets.write(work_dir=output_config.work_dir, filename="similarity_statistics_before_filtering")
        # Compute distances
        distances = Distances(similarity_sets=similarity_sets)
        distances.create_mega(work_dir=output_config.work_dir, filename="distances_before_filtering")
        # Filter similarity matrix
        similarity_matrix = self._filter_similarity_matrix(similarity_matrix=similarity_matrix, groups=groups,
                                                           work_dir=output_config.work_dir)
        # Write similarity matrix to file after filtering
        similarity_matrix.write(work_dir=output_config.work_dir, filename="similarity_matrix_after_filtering")
        similarity_matrix.write_global(work_dir=output_config.work_dir, filename="similarity_matrix_after_filtering",
                                       total_spectra=groups.total_spectra, vector_store=self.vector_store)
        # Compute similarity sets
        similarity_sets = SimilaritySets(similarity_matrix=similarity_matrix, groups=groups)
        # Write similarity sets to file
        similarity_sets.write(work_dir=output_config.work_dir, filename="similarity_statistics_after_filtering")
        # Compute distances
        distances = Distances(similarity_sets=similarity_sets)
        distances.create_mega(work_dir=output_config.work_dir, filename="distances_after_filtering")

    def _setup_vector_store(self) -> VectorStore:
        # Create a VectorStore instance for storing spectra and their vector representations
        config = self.config_factory.create_output_config()
        vector_store_path = config.work_dir
        return VectorStore(vector_store_path)

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

        vector_store_writer = multiprocessing.pool.ThreadPool(
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
            file_spectra, file_filtered, file_failed_parsed, file_failed_processed, file_total_spectra, file_id, file_group_id = result
            group = groups.get_group(file_group_id)
            group.failed_parsed += file_failed_parsed
            group.failed_processed += file_failed_processed
            group.total_spectra += file_total_spectra
            peak_file = group.get_peak_file(file_id)
            peak_file.filtered = file_filtered
            peak_file.failed_parsed = file_failed_parsed
            peak_file.failed_processed = file_failed_processed
            peak_file.total_spectra = file_total_spectra
            groups.failed_parsed += file_failed_parsed
            groups.failed_processed += file_failed_processed
            groups.total_spectra += file_total_spectra

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

        vector_store_writer.close()
        vector_store_writer.join()

        self.vector_store.cleanup()

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
        return spectra, file.filtered, file.failed_parsed, file.failed_processed, file.total_spectra, file.get_id(), file.get_group_id()

    def _vectorize_and_write_spectra(self, spectra_queue: queue.Queue[Optional[GroupSpectrum]]):
        """
        Consumes spectra, vectorizes, and writes them to the dataset.
        """
        spec_to_write = []

        def _process_batch():
            """Process and write the collected spectra."""

            spectra = [spec.spectrum for spec in spec_to_write]
            vectors = self.vectorizer.vectorize(spectra)
            dict_list = [
                {**spec.to_dict(), "vector": vector}
                for spec, vector in zip(spec_to_write, vectors)
            ]
            self.vector_store.write(dict_list)
            spec_to_write.clear()

        while True:
            spectrum = spectra_queue.get()
            if spectrum is None:
                # Signal to finish processing
                if spec_to_write:
                    _process_batch()
                break

            # Group spectra by group_id
            spec_to_write.append(spectrum)

            # Process batch if size limit is reached
            if len(spec_to_write) >= 10_000:
                _process_batch()

    def _index(self, groups: Groups) -> MS2Index:
        output_config = self.config_factory.create_output_config()
        d = self.vectorizer.reducer.low_dim
        total_valid_spectra = groups.total_valid_spectra()
        # create index instance
        index = MS2Index(total_valid_spectra=total_valid_spectra, d=d,
                         work_dir=output_config.work_dir)
        # train the index
        index.train(vector_store=self.vector_store)
        # index the spectra for the groups
        index.add(vector_store=self.vector_store, batch_size=1000)
        return index

    def _query(self, index: MS2Index) -> SimilarityMatrix:
        index_config = self.config_factory.create_index_config()
        similarity_threshold = index_config.similarity
        # query each spectrum against the index
        similarity_matrix = index.range_search(similarity_threshold=similarity_threshold,
                                               vector_store=self.vector_store, batch_size=1000)
        return similarity_matrix

    def _filter_similarity_matrix(self, similarity_matrix: SimilarityMatrix, groups: Groups,
                                  work_dir: str) -> SimilarityMatrix:
        sim_matrix_processing_config = self.config_factory.create_sim_matrix_processing_config()
        pipeline = SimilarityMatrixPipelineFactory.create_pipeline(config=sim_matrix_processing_config,
                                                                   vector_store=self.vector_store, groups=groups)
        similarity_matrix = pipeline.process(similarity_matrix=similarity_matrix, work_dir=work_dir,
                                             total_spectra=groups.total_spectra)
        return similarity_matrix
