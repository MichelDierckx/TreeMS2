import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from TreeMS2.groups.groups import Groups
from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.filters.mask_filter import MaskFilter
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.similarity_matrix.spectra_matrix import SpectraMatrix
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class PrecursorMzFilter(MaskFilter):
    def __init__(self, groups: Groups, vector_store: VectorStore,
                 precursor_mz_window: float):
        self.precursor_mz_window = precursor_mz_window
        self.vector_store = vector_store
        self.groups = groups
        super().__init__(None)

    def construct_mask(self, similarity_matrix: SimilarityMatrix) -> SpectraMatrix:
        rows, cols = similarity_matrix.matrix.nonzero()

        # precursor mz values for all entries
        precursor_mz = self.vector_store.get_col("precursor_mz").to_numpy(dtype=np.float32).ravel()

        # Vectorized lookup using rows and cols
        precursor_mz_rows = precursor_mz[rows]
        precursor_mz_cols = precursor_mz[cols]

        mask_data = np.abs(precursor_mz_rows - precursor_mz_cols) > self.precursor_mz_window
        # Filter the rows and columns based on the mask
        rows = rows[mask_data]
        cols = cols[mask_data]
        mask_data = np.ones(rows.size, dtype=np.bool_)

        m = csr_matrix((mask_data, (rows, cols)), shape=similarity_matrix.matrix.shape,
                       dtype=np.bool_)
        mask = SpectraMatrix(m)
        return mask

    def write_filter_statistics(self, target_dir: str, total_spectra):
        if self.mask is None:
            raise ValueError("No mask has been constructed")

        rows, cols = self.mask.matrix.nonzero()

        row_ids = self.vector_store.get_data(rows, ["global_id"])["global_id"].to_numpy(dtype=np.int32)
        col_ids = self.vector_store.get_data(cols, ["global_id"])["global_id"].to_numpy(dtype=np.int32)
        m = csr_matrix((self.mask.matrix.data, (row_ids, col_ids)), shape=(total_spectra, total_spectra),
                       dtype=np.bool_)

        nr_groups = self.groups.get_size()
        s = np.zeros((nr_groups, nr_groups), dtype=np.uint64)
        # loop over groups representing the rows in s
        for row_group in self.groups.get_groups():
            row_group_id = row_group.get_id()
            # retrieve id of first and last spectrum in group
            row_begin = row_group.begin
            row_end = row_group.end
            # loop over groups representing the columns in s
            for col_group in self.groups.get_groups():
                col_group_id = col_group.get_id()
                # retrieve id of first and last spectrum in group
                col_begin = col_group.begin
                col_end = col_group.end
                # count the number of spectra that have been filtered between group A and group B
                filtered = m[row_begin:row_end + 1, col_begin:col_end + 1].nnz
                s[row_group_id, col_group_id] = filtered

        # create a dataframe
        group_names = [group.get_group_name() for group in self.groups.get_groups()]
        df = pd.DataFrame(s, index=group_names, columns=group_names)

        # create path
        filters_dir = os.path.join(target_dir, "filters")
        os.makedirs(filters_dir, exist_ok=True)
        path = os.path.join(filters_dir, "precursor_mz.txt")

        # write statistics to disk
        with open(path, 'w') as f:
            # Write a header explanation
            f.write(
                f"Number of spectra considered similar between each pair of groups with a precursor m/z difference larger than {self.precursor_mz_window}:\n\n")
            # Write the matrix
            f.write(df.to_string())
        logger.info(
            f"Overview of the number of similarities filtered due to precursor m/z difference written to '{path}'")
        return s

    def save_mask(self, target_dir: str):
        # save mask
        path = self.mask.write(os.path.join(target_dir, "precursor_mz"))
        logger.info(f"Precursor mz mask has been written to '{path}'.")
        return path

    def save_mask_global(self, target_dir: str, total_spectra: int):
        # save mask
        path = self.mask.write_global(os.path.join(target_dir, "precursor_mz_global"), total_spectra, self.vector_store)
        logger.info(f"Precursor mz mask has been written to '{path}'.")
        return path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(precursor_mz_window={self.precursor_mz_window:.3f})"
