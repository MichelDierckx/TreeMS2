import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from TreeMS2.groups.groups import Groups
from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix
from TreeMS2.vector_store.vector_store import VectorStore

logger = get_logger(__name__)


class SimilaritySets:
    def __init__(self, groups: Groups, vector_store: VectorStore):
        self.groups = groups
        self.vector_store = vector_store
        self.similarity_sets: npt.NDArray[np.uint64] = np.zeros((self.groups.get_size(), self.groups.get_size()),
                                                                dtype=np.uint64)

    def update_similarity_sets(self, similarity_matrix: SimilarityMatrix):
        rows, cols = similarity_matrix.matrix.nonzero()
        total_spectra = self.groups.total_spectra

        row_ids = self.vector_store.get_data(rows, ["global_id"])["global_id"].to_numpy(dtype=np.int32)
        col_ids = self.vector_store.get_data(cols, ["global_id"])["global_id"].to_numpy(dtype=np.int32)
        m = csr_matrix((similarity_matrix.matrix.data, (row_ids, col_ids)), shape=(total_spectra, total_spectra),
                       dtype=np.bool_)

        # loop over groups representing the rows in s
        for row_group in self.groups.get_groups():
            row_group_id = row_group.get_id()
            # retrieve id of first and last spectrum in group
            row_begin = row_group.begin
            row_end = row_group.end
            # loop over groups representing the columns in s
            for col_group in self.groups.get_groups():
                col_group_id = col_group.get_id()
                # nr of similar spectra group A has to A equals the number of spectra in A
                if row_group_id == col_group_id:
                    self.similarity_sets[row_group_id, col_group_id] = row_group.total_spectra
                    continue
                # retrieve id of first and last spectrum in group
                col_begin = col_group.begin
                col_end = col_group.end
                # count the number of spectra in group A that have at least one similar in group B
                # m[row_begin:row_end + 1, col_begin:col_end] extracts the relevant submatrix
                # .getnnz(axis=1) counts the number of nonzero entries per row
                # > 0 converts it into a boolean array where True means the row has at least one nonzero value
                self.similarity_sets[row_group_id, col_group_id] += (
                        m[row_begin:row_end + 1, col_begin:col_end].getnnz(axis=1) > 0).sum()

    def write(self, path: str):
        s = self.similarity_sets
        # create a dataframe
        group_names = [group.get_group_name() for group in self.groups.get_groups()]
        df = pd.DataFrame(s, index=group_names, columns=group_names)

        # write S to a file in human-readable format
        logger.info(f"Writing similarity statistics to '{path}'.")
        with open(path, 'w') as f:
            df_string = df.to_string()
            f.write(df_string)
        # return s to calculate global distance matrix
        return s

    @staticmethod
    def load(path: str, groups: Groups, vector_store: VectorStore) -> Optional["SimilaritySets"]:
        """
        Load a SimilaritySets object from a file. If loading fails, return None.
        """
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, sep=r"\s+")
            expected_groups = [group.get_group_name() for group in groups.get_groups()]
            if list(df.index) != expected_groups or list(df.columns) != expected_groups:
                return None
            similarity_sets = df.to_numpy(dtype=np.uint64)
            similarity_set_obj = SimilaritySets(groups=groups, vector_store=vector_store)
            similarity_set_obj.similarity_sets = similarity_sets
            return similarity_set_obj
        except Exception:
            return None
