import contextlib
import multiprocessing
import os
import time
from collections import defaultdict, Counter
from typing import Tuple, List, Optional, Dict, Union

import joblib
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

from TreeMS2.config.env_variables import TREEMS2_NUM_CPUS
from TreeMS2.ingestion.batch_writer import BatchWriter
from TreeMS2.ingestion.file_processor import FileProcessor
from TreeMS2.ingestion.preprocessing.pipeline import Pipeline
from TreeMS2.ingestion.spectra_dataset.peak_file.parsing_stats import ParsingStats
from TreeMS2.ingestion.spectra_dataset.peak_file.peak_file import PeakFile
from TreeMS2.ingestion.spectra_dataset.peak_file.quality_stats import QualityStats
from TreeMS2.ingestion.spectra_dataset.peak_file.readers.reader_factory import ReaderFactory
from TreeMS2.ingestion.spectra_dataset.peak_file.spectrum_parser import SpectrumParser
from TreeMS2.ingestion.spectra_dataset.spectra_dataset import SpectraDataset
from TreeMS2.ingestion.ingestion_plots import PrecursorChargeHistogram
from TreeMS2.config.logger_config import get_logger, log_section_title
from TreeMS2.ingestion.spectra_dataset.treems2_spectrum import TreeMS2Spectrum
from TreeMS2.spectrum.spectrum_processing.pipeline import (
    ProcessingPipelineFactory,
    SpectrumProcessingPipeline,
)
from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import (
    ScalingMethod,
)
from TreeMS2.ingestion.vectorization.dimensionality_reducer import (
    DimensionalityReducer,
)
from TreeMS2.ingestion.vectorization.spectra_binner import SpectraBinner
from TreeMS2.ingestion.vectorization.spectra_vector_transformer import (
    SpectraVectorTransformer,
)
from TreeMS2.states.context import Context
from TreeMS2.indexing.vector_store_indexing_state import VectorStoreIndexingState
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType
from TreeMS2.config.logger_config import format_execution_time
from TreeMS2.ingestion.storage.vector_store import VectorStore
from TreeMS2.ingestion.storage.vector_stores import VectorStores

logger = get_logger(__name__)

GROUPS_SUMMARY_FILE = "groups.json"
VECTOR_STORES = {"charge_1", "charge_2", "charge_3", "charge_4plus", "charge_unknown"}
VECTOR_STORE_MANAGER_SAVE_FILE = "vector_stores.json"
VECTOR_STORE_METADATA_DIR_NAME = "vector_stores_metadata"


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class IngestionState(State):
    STATE_TYPE = StateType.INGESTION_STATE

    def __init__(self, context: Context):
        super().__init__(context)

        # group file
        self.sample_to_group_file: str = context.config.sample_to_group_file

        self.buffer_size: int = context.config.batch_size
        self.incremental_compaction: bool = context.config.incremental_compaction

        # spectrum preprocessing
        self.min_peaks: int = context.config.min_peaks
        self.min_mz_range: float = context.config.min_mz_range
        self.remove_precursor_tol: float = context.config.remove_precursor_tol
        self.min_intensity: float = context.config.min_intensity
        self.max_peaks_used: int = context.config.max_peaks_used
        self.scaling: ScalingMethod = ScalingMethod(context.config.scaling)

        # vectorization
        self.min_mz: float = context.config.min_mz
        self.max_mz: float = context.config.max_mz
        self.fragment_tol: float = context.config.fragment_tol
        self.low_dim: int = context.config.low_dim

    def run(self):
        log_section_title(logger=logger, title="[ Processing Peak Files ]")
        # Try loading existing data if overwrite is not enabled
        if not self.context.config.overwrite:
            self.context.groups = SpectraDataset.load(
                path=os.path.join(self.context.results_dir, GROUPS_SUMMARY_FILE)
            )
            self.context.vector_store_manager = VectorStores.load(
                path=os.path.join(
                    os.path.join(
                        self.context.results_dir, VECTOR_STORE_METADATA_DIR_NAME
                    ),
                    VECTOR_STORE_MANAGER_SAVE_FILE,
                )
            )

            if self.context.groups and self.context.vector_store_manager:
                logger.info(
                    f"Found existing results ('{os.path.join(self.context.results_dir, GROUPS_SUMMARY_FILE)}', '{os.path.join(self.context.results_dir, VECTOR_STORE_MANAGER_SAVE_FILE)}'). Skipping processing and loading results from disk."
                )
                self._transition()
                return  # Exit early if loading was successful

        # If loading failed or overwrite is enabled, generate fresh data
        self.context.groups, self.context.vector_store_manager = self._generate()

        # Proceed to indexing
        self._transition()

    def _transition(self):
        """
        Adds create index states for every non-empty vector store and a compute_distances_state
        :return:
        """
        self.context.pop_state()
        for vector_store in self.context.vector_store_manager.vector_stores.values():
            if not vector_store.is_empty():
                self.context.push_state(VectorStoreIndexingState(self.context, vector_store))

    def _generate(self) -> Tuple[SpectraDataset, VectorStores]:
        # parse sample to group file
        groups = SpectraDataset.read(self.sample_to_group_file)
        logger.info(
            f"Loaded group mapping from '{self.sample_to_group_file}': {groups.count_spectra_sets()} groups across {groups.count_peak_files()} peak files."
        )
        # create vector store manager instance
        vector_store_manager = VectorStores(
            vector_stores={
                name: VectorStore(
                    name=name,
                    directory=os.path.join(self.context.lance_dir, name),
                    vector_dim=self.low_dim,
                )
                for name in VECTOR_STORES
            }
        )

        vector_store_manager.clear()
        # create vectorizer instance (converts high dimensional spectra to low dimensional spectra)
        vectorizer = self._setup_vectorizer()
        # create spectrum preprocessor instance (preprocesses the spectra)
        spectrum_processor = ProcessingPipelineFactory.create_pipeline(
            min_peaks=self.min_peaks,
            min_mz_range=self.min_mz_range,
            remove_precursor_tol=self.remove_precursor_tol,
            min_intensity=self.min_intensity,
            max_peaks_used=self.max_peaks_used,
            scaling=self.scaling,
            min_mz=vectorizer.binner.min_mz,
            max_mz=vectorizer.binner.max_mz,
        )
        # read, preprocess, vectorize and store spectra
        self._read_and_process_spectra(
            groups=groups,
            processing_pipeline=spectrum_processor,
            vectorizer=vectorizer,
            vector_store_manager=vector_store_manager,
        )
        # cleanup old versions to compact dataset
        logger.info("Compacting dataset(s)...")
        compacting_time_start = time.time()
        vector_store_manager.cleanup()
        logger.info(
            f"Finished compaction in {format_execution_time(time.time() - compacting_time_start)}"
        )

        vector_store_manager.cleanup()

        # write groups summary and reading/processing statistics to file
        groups.write_to_file(
            path=os.path.join(self.context.results_dir, GROUPS_SUMMARY_FILE)
        )
        logger.info(
            f"Saved detailed processing summary to '{os.path.join(self.context.results_dir, GROUPS_SUMMARY_FILE)}'."
        )
        vector_store_manager.save(
            parent_dir=os.path.join(
                self.context.results_dir, VECTOR_STORE_METADATA_DIR_NAME
            ),
            filename=VECTOR_STORE_MANAGER_SAVE_FILE,
        )
        logger.info(
            f"For additional information per lance dataset, refer to '{os.path.join(self.context.results_dir, VECTOR_STORE_MANAGER_SAVE_FILE)}'"
        )
        return groups, vector_store_manager

    def _setup_vectorizer(self) -> SpectraVectorTransformer:
        binner = SpectraBinner(
            min_mz=self.min_mz, max_mz=self.max_mz, bin_size=self.fragment_tol
        )
        reducer = DimensionalityReducer(low_dim=self.low_dim, high_dim=binner.dim)
        spectrum_vectorizer = SpectraVectorTransformer(binner=binner, reducer=reducer)
        return spectrum_vectorizer

    def _read_and_process_spectra(
        self,
        groups: SpectraDataset,
        processing_pipeline: Pipeline,
        vectorizer: SpectraVectorTransformer,
        vector_store_manager: VectorStores,
    ):
        """
        Main function to orchestrate producers and consumers using queues.
        """

        all_files = [f for g in groups.get_spectra_sets() for f in g.get_peak_files()]
        max_workers = min(len(all_files), int(os.getenv(TREEMS2_NUM_CPUS, multiprocessing.cpu_count())))

        logger.info(f"Processing spectra from {len(all_files)} peak files...")


        def process_file(file):
            reader = ReaderFactory().get_reader(file.file_path)
            batch_writer = BatchWriter(self.buffer_size, vectorizer, vector_store_manager)
            processor = FileProcessor(reader=reader, pipeline=processing_pipeline, batch_writer=batch_writer)
            return processor.process(file)

        with tqdm_joblib(
            tqdm(
                desc="Processing Files",
                total=len(all_files),
                unit="file",
                position=0,
                leave=True,
            )
        ):
            results = joblib.Parallel(n_jobs=max_workers, backend="loky")(
                joblib.delayed(process_file)(file) for file in all_files
            )

        # Explicitly shut down the loky reusable executor to avoid idle lingering processes
        get_reusable_executor().shutdown(wait=True)

        for result in results:
            group = groups.get_spectra_set(result.file_id)
            peak_file = group.get_peak_file(result.file_id)

            peak_file.parsing_stats = result.parsing_stats
            peak_file.quality_stats = result.quality_stats

        # Log final processing statistics.
        parsing_stat_counts, quality_stat_counts = groups.get_stats()
        logger.info(
            f"Processed {parsing_stat_counts.valid + parsing_stat_counts.invalid} spectra from {len(all_files)} peak files:\n"
            f"\t- {parsing_stat_counts.invalid} spectra could not be parsed.\n"
            f"\t- {quality_stat_counts.low_quality} spectra were filtered out as low quality.\n"
            f"\t- {quality_stat_counts.high_quality} spectra were successfully processed, vectorized and written to the lance dataset(s)."
        )

        PrecursorChargeHistogram.plot(charge_counts=parsing_stat_counts.precursor_charge_counts,
            path=os.path.join(
                self.context.results_dir, "precursor_charge_distribution.png"
            )
        )
        logger.info(
            f"Saved histogram displaying distribution of spectra by precursor charge to '{os.path.join(self.context.results_dir, "precursor_charge_distribution.png")}'."
        )

        return
