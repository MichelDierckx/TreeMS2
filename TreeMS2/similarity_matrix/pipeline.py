from typing import List

from TreeMS2.logger_config import get_logger
from .filters.mask_filter import MaskFilter
from .filters.precursor_mz_filter import PrecursorMzFilter
from .similarity_matrix import SimilarityMatrix
from ..config.sim_matrix_processing_config import SimMatrixProcessingConfig
from ..groups.groups import Groups
from ..vector_store.vector_store import VectorStore

logger = get_logger(__name__)


# Processing pipeline that applies each filter in sequence
class SimilarityMatrixPipeline:
    def __init__(self, mask_filters: List[MaskFilter]):
        self.mask_filters = mask_filters
        logger.debug(f"Created {self}")

    def process(self, similarity_matrix: SimilarityMatrix, work_dir: str) -> SimilarityMatrix:
        for mask_filter in self.mask_filters:  # Iterate over the list of mask filters
            mask_filter.apply(similarity_matrix)  # Apply each mask filter
            mask_filter.save_mask(work_dir=work_dir)
            mask_filter.write_filter_statistics(work_dir=work_dir)
        return similarity_matrix

    def __repr__(self) -> str:
        """Provide a textual representation of the pipeline and its processors."""
        mask_filters_repr = "\n\t".join([repr(mask_filter) for mask_filter in self.mask_filters])
        return f"{self.__class__.__name__}:\n\t{mask_filters_repr}"


# Pipeline Factory that creates the processing pipeline based on configuration
class SimilarityMatrixPipelineFactory:
    @staticmethod
    def create_pipeline(config: SimMatrixProcessingConfig,
                        vector_store: VectorStore, groups: Groups) -> SimilarityMatrixPipeline:
        mask_filters = []
        if config.precursor_mz_window is not None:
            mask_filters.append(
                PrecursorMzFilter(precursor_mz_window=config.precursor_mz_window,
                                  vector_store=vector_store, groups=groups))
        pipeline = SimilarityMatrixPipeline(mask_filters=mask_filters)
        return pipeline
