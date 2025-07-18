from typing import List

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix

logger = get_logger(__name__)


# Processing pipeline that applies each filter in sequence
class SimilarityMatrixPipeline:
    def __init__(self, mask_filters: List[MaskFilter]):
        self.mask_filters = mask_filters

    def process(
        self, similarity_matrix: SimilarityMatrix, total_spectra: int, target_dir: str
    ) -> SimilarityMatrix:
        for mask_filter in self.mask_filters:  # Iterate over the list of mask filters
            mask_filter.apply(similarity_matrix)  # Apply each mask filter
        return similarity_matrix

    def add_filter(self, mask_filter: MaskFilter) -> MaskFilter:
        self.mask_filters.append(mask_filter)
        return mask_filter
