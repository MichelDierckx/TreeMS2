import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from TreeMS2.groups.groups import Groups
from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix

logger = get_logger(__name__)


class SimilaritySets:
    def __init__(self, groups: Groups):
        self.groups = groups
        self.similarity_sets: npt.NDArray[np.uint64] = np.zeros((self.groups.get_size(), self.groups.get_size()),
                                                                dtype=np.uint64)

    def update_similarity_sets(self, similarity_matrix: SimilarityMatrix, group_ids=npt.NDArray[np.uint16]):
        rows, cols = similarity_matrix.matrix.nonzero()

        row_group_ids = group_ids[rows]
        col_group_ids = group_ids[cols]

        pairs = np.vstack((row_group_ids, col_group_ids)).T
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

        self.similarity_sets[unique_pairs[:, 0], unique_pairs[:, 1]] += counts.astype(np.uint64)

    def write(self, path: str):
        s = self.similarity_sets
        # create a dataframe
        group_names = [group.get_group_name() for group in self.groups.get_groups()]
        df = pd.DataFrame(s, index=group_names, columns=group_names)

        # write S to a file in human-readable format
        with open(path, 'w') as f:
            df_string = df.to_string()
            f.write(df_string)
        # return s to calculate global distance matrix
        return s

    @staticmethod
    def load(path: str, groups: Groups) -> Optional["SimilaritySets"]:
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
            similarity_set_obj = SimilaritySets(groups=groups)
            similarity_set_obj.similarity_sets = similarity_sets
            return similarity_set_obj
        except Exception:
            return None
