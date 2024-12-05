import numpy as np
from scipy.sparse import csr_matrix

from TreeMS2.groups.groups import Groups
from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class PrecursorMzFilter(MaskFilter):
    def __init__(self, similarity_matrix: SimilarityMatrix, precursor_mz_window: float, vector_store: VectorStore,
                 groups: Groups):
        self.precursor_mz_window = precursor_mz_window
        mask = self.construct_mask(similarity_matrix, vector_store, groups)
        super().__init__(mask)

    def construct_mask(self, similarity_matrix: SimilarityMatrix, vector_store: VectorStore,
                       groups: Groups) -> SpectraMatrix:
        rows, cols = similarity_matrix.matrix.nonzero()

        mask_data = np.zeros(shape=rows.shape, dtype=np.bool_)

        for index, row, col in enumerate(zip(rows, cols)):
            row_group_id = groups.get_group_id_from_global_id(row)
            col_group_id = groups.get_group_id_from_global_id(col)

            row_precursor_mz = vector_store.get_metadata(row_group_id, row, groups, "precursor_mz")
            col_precursor_mz = vector_store.get_metadata(col_group_id, col, groups, "precursor_mz")

            if abs(row_precursor_mz - col_precursor_mz) <= self.precursor_mz_window:
                mask_data[index] = False

        m = csr_matrix(mask_data, (rows, cols))
        mask = SpectraMatrix(m)
        return mask

    def write_filter_statistics(self, groups: Groups):
        nr_groups = groups.get_size()
        s = np.zeros((nr_groups, nr_groups), dtype=np.uint64)
        # loop over groups representing the rows in s
        for row_group in groups.get_groups():
            row_group_id = row_group.get_id()
            # retrieve id of first and last spectrum in group
            row_begin = row_group.begin
            row_end = row_group.end
            # loop over groups representing the columns in s
            for col_group in groups.get_groups():
                col_group_id = col_group.get_id()
                # retrieve id of first and last spectrum in group
                col_begin = col_group.begin
                col_end = col_group.end
                # count the number of spectra that have been filtered between group A and group B
                filtered = self.mask.matrix[row_begin:row_end + 1, col_begin:col_end + 1]
                s[row_group_id, col_group_id] = filtered

        # write S to a file in human-readable format
        # group_names = [group.get_group_name() for group in groups.get_groups()]
        # df = pd.DataFrame(s, index=group_names, columns=group_names)
        # path = os.path.join(self.base_path, "s.txt")
        # logger.info(f"Writing similarity statistics to '{path}'.")
        # with open(path, 'w') as f:
        #     df_string = df.to_string()
        #     f.write(df_string)

        # return s to calculate global distance matrix
        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(precursor_mz_window={self.precursor_mz_window:.3f})"
