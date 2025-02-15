import multiprocessing
import os
from functools import partial
from typing import Tuple, List

from TreeMS2.groups.groups import Groups
from TreeMS2.groups.peak_file.peak_file import PeakFile
from TreeMS2.logger_config import get_logger
from TreeMS2.spectrum.group_spectrum import GroupSpectrum
from TreeMS2.spectrum.spectrum_processing.pipeline import ProcessingPipelineFactory, SpectrumProcessingPipeline
from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import ScalingMethod
from TreeMS2.spectrum.spectrum_vectorization.dimensionality_reducer import DimensionalityReducer
from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner
from TreeMS2.spectrum.spectrum_vectorization.spectrum_vectorizer import SpectrumVectorizer
from TreeMS2.states.context import Context
from TreeMS2.states.create_index_state import CreateIndexState
from TreeMS2.states.state import State
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)

GROUPS_SUMMARY_FILE = "groups.json"
VECTOR_STORE_DIR = "spectra.lance"


class ProcessSpectraState(State):
    MAX_SPECTRA_IN_MEM = 1_000_000

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

    def run(self, overwrite: bool):
        if overwrite or not self._is_output_generated():
            # generate the required output
            groups, vector_store = self._generate()
        else:
            # load the required output
            groups, vector_store = self._load()
        # move to the create index state
        self.context.replace_state(
            state=CreateIndexState(context=self.context, groups=groups, vector_store=vector_store))

    def _generate(self) -> Tuple[Groups, VectorStore]:
        # parse sample to group file
        groups = Groups.read(self.sample_to_group_file)
        # create vector store instance
        vector_store = VectorStore(path=os.path.join(self.work_dir, VECTOR_STORE_DIR))
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
        self._read_and_process_spectra(groups=groups, processing_pipeline=spectrum_processor, vectorizer=vectorizer,
                                       vector_store=vector_store)
        # write groups summary and reading/processing statistics to file
        groups.write_to_file(path=os.path.join(self.work_dir, GROUPS_SUMMARY_FILE))
        return groups, vector_store

    def _load(self) -> Tuple[Groups, VectorStore]:
        # load groups from file
        groups = Groups.from_file(path=os.path.join(self.work_dir, GROUPS_SUMMARY_FILE))
        # init vector store
        vector_store = VectorStore(path=os.path.join(self.work_dir, VECTOR_STORE_DIR))
        vector_store.has_data = True
        return groups, vector_store

    def _is_output_generated(self) -> bool:
        if not os.path.isdir(os.path.join(self.work_dir, VECTOR_STORE_DIR)):
            return False
        if not os.path.isfile(os.path.join(self.work_dir, GROUPS_SUMMARY_FILE)):
            return False
        return True

    def _setup_vectorizer(self) -> SpectrumVectorizer:
        binner = SpectrumBinner(min_mz=self.min_mz, max_mz=self.max_mz, bin_size=self.fragment_tol)
        reducer = DimensionalityReducer(low_dim=self.low_dim, high_dim=binner.dim)
        spectrum_vectorizer = SpectrumVectorizer(binner=binner, reducer=reducer)
        return spectrum_vectorizer

    @staticmethod
    def _process_file(file: PeakFile, processing_pipeline: SpectrumProcessingPipeline, vectorizer: SpectrumVectorizer,
                      vector_store: VectorStore) -> PeakFile:
        buffer: List[GroupSpectrum] = []

        def vectorize_batch():
            spectra = [spec.spectrum for spec in buffer]
            return vectorizer.vectorize(spectra)

        def write_batch(vectors):
            dict_list = [
                {**spec.to_dict(), "vector": vector}
                for spec, vector in zip(buffer, vectors)
            ]
            vector_store.write(dict_list)

        for processed_spectrum in file.get_spectra(processing_pipeline=processing_pipeline):
            buffer.append(processed_spectrum)
            if len(buffer) >= 1_000:
                write_batch(vectorize_batch())
                buffer.clear()
        if buffer:
            write_batch(vectorize_batch())
            buffer.clear()
        return file

    def _read_and_process_spectra(self, groups: Groups, processing_pipeline: SpectrumProcessingPipeline,
                                  vectorizer: SpectrumVectorizer, vector_store: VectorStore):
        """
        Main function to orchestrate producers and consumers using queues.
        """
        # Use multiple worker processes to read the peak files.
        max_file_workers = min(groups.get_nr_files(), multiprocessing.cpu_count())

        # Flatten the list of all peak files across groups for processing.
        all_files = [
            file
            for group in groups.get_groups()
            for file in group.get_peak_files()
        ]

        process_file_partial = partial(self._process_file, processing_pipeline=processing_pipeline,
                                       vectorizer=vectorizer, vector_store=vector_store)

        logger.info(f"Processing spectra from {len(all_files)} files ...")

        with multiprocessing.Pool(processes=max_file_workers) as pool:
            results = pool.map(process_file_partial, all_files)  # Process files in parallel

        for file in results:
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

        # Log final processing statistics.
        logger.info(
            f"Processed {groups.total_spectra} spectra from {len(all_files)} files:\n"
            f"\t- {groups.failed_processed} spectra were filtered out as low quality.\n"
            f"\t- {groups.failed_parsed} spectra could not be parsed.\n"
            f"\t- {groups.total_spectra - groups.failed_processed - groups.failed_parsed} spectra were successfully written to the dataset."
        )

        vector_store.cleanup()
        # create global ordering of spectra based on (group, file and spectrum position in file)
        groups.update()
        # add global identifier (based on total ordering) to each spectrum in the vector store
        vector_store.add_global_ids(groups=groups)
