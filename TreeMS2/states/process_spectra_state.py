import multiprocessing
import os
import time
from collections import defaultdict, Counter
from typing import Tuple, List, Optional, Dict, Union

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

from TreeMS2.environment_variables import TREEMS2_NUM_CPUS
from TreeMS2.groups.groups import Groups
from TreeMS2.groups.peak_file.peak_file import PeakFile
from TreeMS2.histogram import PrecursorChargeHistogram
from TreeMS2.logger_config import get_logger, log_section_title
from TreeMS2.spectrum.group_spectrum import GroupSpectrum
from TreeMS2.spectrum.spectrum_processing.pipeline import ProcessingPipelineFactory, SpectrumProcessingPipeline
from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import ScalingMethod
from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import DimensionalityReducer
from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner
from TreeMS2.spectrum.spectrum_vectorization.spectrum_vectorizer import SpectrumVectorizer
from TreeMS2.states.context import Context
from TreeMS2.states.create_index_state import CreateIndexState
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType
from TreeMS2.utils.utils import format_execution_time
from TreeMS2.vector_store.vector_store import VectorStore
from TreeMS2.vector_store.vector_store_manager import VectorStoreManager

logger = get_logger(__name__)

GROUPS_SUMMARY_FILE = "groups.json"
VECTOR_STORES = {"charge_1", "charge_2", "charge_3", "charge_4plus", "charge_unknown"}
VECTOR_STORE_MANAGER_SAVE_FILE = "vector_stores.json"


def map_charge_to_vector_store(charge: Optional[int]) -> str:
    """Maps precursor charge to the name of the corresponding vector store."""
    if charge == 1:
        return "charge_1"
    elif charge == 2:
        return "charge_2"
    elif charge == 3:
        return "charge_3"
    elif charge is not None and charge >= 4:
        return "charge_4plus"
    return "charge_unknown"  # Covers cases where charge is None or not recognized


class ProcessSpectraState(State):
    STATE_TYPE = StateType.PROCESS_SPECTRA
    MAX_SPECTRA_IN_MEM = 1_000_00

    def __init__(self, context: Context):
        super().__init__(context)

        # work directory
        self.work_dir: str = context.config.work_dir

        # group file
        self.sample_to_group_file: str = context.config.sample_to_group_file

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
            self.context.groups = Groups.load(path=os.path.join(self.work_dir, GROUPS_SUMMARY_FILE))
            self.context.vector_store_manager = VectorStoreManager.load(
                path=os.path.join(self.work_dir, VECTOR_STORE_MANAGER_SAVE_FILE))

            if self.context.groups and self.context.vector_store_manager:
                logger.info(
                    f"Found existing results ('{os.path.join(self.work_dir, GROUPS_SUMMARY_FILE)}', '{os.path.join(self.work_dir, VECTOR_STORE_MANAGER_SAVE_FILE)}'). Skipping processing and loading results from disk.")
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
                self.context.push_state(CreateIndexState(self.context, vector_store))

    def _generate(self) -> Tuple[Groups, VectorStoreManager]:
        # parse sample to group file
        groups = Groups.read(self.sample_to_group_file)
        logger.info(
            f"Loaded group mapping from '{self.sample_to_group_file}': {groups.get_size()} groups across {groups.get_nr_files()} peak files.")
        # create vector store manager instance
        vector_store_manager = VectorStoreManager(vector_stores={
            name: VectorStore(name=name, directory=os.path.join(self.work_dir, name), vector_dim=self.low_dim)
            for name in VECTOR_STORES
        })

        vector_store_manager.clear()
        # create vectorizer instance (converts high dimensional spectra to low dimensional spectra)
        vectorizer = self._setup_vectorizer()
        # create spectrum preprocessor instance (preprocesses the spectra)
        spectrum_processor = ProcessingPipelineFactory.create_pipeline(min_peaks=self.min_peaks,
                                                                       min_mz_range=self.min_mz_range,
                                                                       remove_precursor_tol=self.remove_precursor_tol,
                                                                       min_intensity=self.min_intensity,
                                                                       max_peaks_used=self.max_peaks_used,
                                                                       scaling=self.scaling,
                                                                       min_mz=vectorizer.binner.min_mz,
                                                                       max_mz=vectorizer.binner.max_mz)
        # read, preprocess, vectorize and store spectra
        precursor_charge_histogram = self._read_and_process_spectra(groups=groups,
                                                                    processing_pipeline=spectrum_processor,
                                                                    vectorizer=vectorizer,
                                                                    vector_store_manager=vector_store_manager)
        precursor_charge_histogram.plot(path=os.path.join(self.work_dir, "precursor_charge_distribution.png"))
        logger.info(
            f"Saved histogram displaying distribution of spectra by precursor charge to '{os.path.join(self.work_dir, "precursor_charge_distribution.png")}'.")
        # cleanup old versions to compact dataset
        logger.info("Compacting dataset(s)...")
        compacting_time_start = time.time()
        vector_store_manager.cleanup()
        vector_store_manager.update_vector_count()
        logger.info(f"Finished compaction in {format_execution_time(time.time() - compacting_time_start)}")

        logger.info(
            "Creating total ordering for the vectors based on the corresponding group identifier, file identifier and spectrum position within file...")
        total_ordering_time_start = time.time()
        # create global ordering of spectra based on (group, file and spectrum position in file)
        groups.update()
        # add global identifier (based on total ordering) to each spectrum in the vector store
        vector_store_manager.add_global_ids(groups=groups)
        vector_store_manager.cleanup()
        logger.info(f"Added total ordering in {format_execution_time(time.time() - total_ordering_time_start)}")
        # write groups summary and reading/processing statistics to file
        groups.write_to_file(path=os.path.join(self.work_dir, GROUPS_SUMMARY_FILE))
        logger.info(f"Saved detailed processing summary to '{os.path.join(self.work_dir, GROUPS_SUMMARY_FILE)}'.")
        vector_store_manager.save(path=os.path.join(self.work_dir, VECTOR_STORE_MANAGER_SAVE_FILE))
        logger.info(
            f"For additional information per lance dataset, refer to '{os.path.join(self.work_dir, VECTOR_STORE_MANAGER_SAVE_FILE)}'")
        return groups, vector_store_manager

    def _setup_vectorizer(self) -> SpectrumVectorizer:
        binner = SpectrumBinner(min_mz=self.min_mz, max_mz=self.max_mz, bin_size=self.fragment_tol)
        reducer = DimensionalityReducer(low_dim=self.low_dim, high_dim=binner.dim)
        spectrum_vectorizer = SpectrumVectorizer(binner=binner, reducer=reducer)
        return spectrum_vectorizer

    @staticmethod
    def _process_file(file: PeakFile, processing_pipeline: SpectrumProcessingPipeline, vectorizer: SpectrumVectorizer,
                      vector_store_manager: VectorStoreManager,
                      locks_and_flags: Dict[str, Union[multiprocessing.Lock, multiprocessing.Value]]) -> Tuple[
        PeakFile, Counter, Counter]:
        buffers: defaultdict[str, List[GroupSpectrum]] = defaultdict(list)
        nr_spectra_per_precursor_charge = Counter()  # Store counts per charge
        nr_vectors_per_store = Counter()

        def vectorize_batch(buffer: List[GroupSpectrum]):
            spectra = [spec.spectrum for spec in buffer]
            return vectorizer.vectorize(spectra)

        def write_batch(vector_store_name: str, buffer: List[GroupSpectrum]):
            """Writes a batch of spectra to the correct vector store."""
            vectors = vectorize_batch(buffer)
            dict_list = [{**spec.to_dict(), "vector": vector} for spec, vector in zip(buffer, vectors)]
            vector_store_manager.write(vector_store_name=vector_store_name, entries_to_write=dict_list,
                                       multiprocessing_lock=locks_and_flags[vector_store_name]["lock"],
                                       overwrite=locks_and_flags[vector_store_name]["overwrite"])
            nr_vectors_per_store[vector_store_name] += len(buffer)

        for processed_spectrum in file.get_spectra(processing_pipeline=processing_pipeline):
            charge_category = map_charge_to_vector_store(processed_spectrum.spectrum.precursor_charge)
            buffers[charge_category].append(processed_spectrum)
            nr_spectra_per_precursor_charge[processed_spectrum.spectrum.precursor_charge] += 1
            if len(buffers[charge_category]) >= 10_000:
                write_batch(charge_category, buffers[charge_category])
                buffers[charge_category].clear()

        # Write remaining spectra in each buffer
        for store_name, buffer in buffers.items():
            if buffer:
                write_batch(store_name, buffer)
                buffer.clear()
        return file, nr_spectra_per_precursor_charge, nr_vectors_per_store

    def _read_and_process_spectra(self, groups: Groups, processing_pipeline: SpectrumProcessingPipeline,
                                  vectorizer: SpectrumVectorizer,
                                  vector_store_manager: VectorStoreManager) -> PrecursorChargeHistogram:
        """
        Main function to orchestrate producers and consumers using queues.
        """
        multiprocessing.set_start_method('spawn')  # lance does not work with FORK method

        precursor_charge_histogram = PrecursorChargeHistogram()

        # Flatten the list of all peak files across groups for processing.
        all_files = [
            file
            for group in groups.get_groups()
            for file in group.get_peak_files()
        ]

        # Use multiple worker processes to read the peak files.
        max_file_workers = int(os.environ.get(TREEMS2_NUM_CPUS, multiprocessing.cpu_count()))

        max_file_workers = min(groups.get_nr_files(), max_file_workers)

        logger.info(f"Processing spectra from {len(all_files)} peak files...")

        with multiprocessing.Manager() as m:
            locks_and_flags = vector_store_manager.create_locks_and_flags(manager=m)

            results = Parallel(n_jobs=max_file_workers, backend="loky")(
                delayed(self._process_file)(
                    file, processing_pipeline, vectorizer, vector_store_manager, locks_and_flags
                )
                for file in tqdm(all_files, desc="Processing Files", unit="file", position=0, leave=True)
            )

        # Explicitly shut down the loky reusable executor to avoid idle lingering processes
        get_reusable_executor().shutdown(wait=True)
        logger.debug("Loky worker pool explicitly shut down.")

        for file, nr_spectra_per_precursor_charge, nr_vectors_per_store in results:
            groups.failed_parsed += file.failed_parsed
            groups.failed_processed += file.failed_processed
            groups.total_spectra += file.total_spectra

            group = groups.get_group(file.get_group_id())
            group.failed_parsed += file.failed_parsed
            group.failed_processed += file.failed_processed
            group.total_spectra += file.total_spectra

            peak_file = group.get_peak_file(file.get_id())
            peak_file.filtered = file.filtered
            peak_file.failed_parsed = file.failed_parsed
            peak_file.failed_processed = file.failed_processed
            peak_file.total_spectra = file.total_spectra

            # update histogram to display charge distribution
            precursor_charge_histogram.update(nr_spectra_per_precursor_charge)

            # update the count of the number of vectors for a group in each vector store
            for vector_store_name, nr_vectors in nr_vectors_per_store.items():
                vector_store_manager.update_group_count(vector_store_name=vector_store_name,
                                                        group_id=file.get_group_id(),
                                                        nr_vectors=nr_vectors)

        # Log final processing statistics.
        logger.info(
            f"Processed {groups.total_spectra} spectra from {len(all_files)} peak files:\n"
            f"\t- {groups.failed_parsed} spectra could not be parsed.\n"
            f"\t- {groups.failed_processed} spectra were filtered out as low quality.\n"
            f"\t- {groups.total_spectra - groups.failed_processed - groups.failed_parsed} spectra were successfully processed, vectorized and written to the lance dataset(s)."
        )
        return precursor_charge_histogram
