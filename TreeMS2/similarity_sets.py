import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from TreeMS2.groups.groups import Groups
from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_matrix.similarity_matrix import SimilarityMatrix

logger = get_logger(__name__)


class SimilaritySets:
    def __init__(self, similarity_matrix: SimilarityMatrix, groups: Groups):
        self.similarity_matrix = similarity_matrix
        self.groups = groups
        self.similarity_sets: npt.NDArray[np.uint64] = self._compute_similarity_sets()

    def _compute_similarity_sets(self) -> npt.NDArray[np.uint64]:
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
                # nr of similar spectra group A has to A equals the number of spectra in A
                if row_group_id == col_group_id:
                    s[row_group_id, col_group_id] = row_group.total_spectra
                    continue
                # retrieve id of first and last spectrum in group
                col_begin = col_group.begin
                col_end = col_group.end
                # count the number of spectra in group A that have at least one similar in group B
                counter = 0
                for spec_id in range(row_begin, row_end + 1):
                    # check if there is at least one spectrum in the other group marked as similar
                    nr_similarities = self.similarity_matrix.matrix[spec_id, col_begin:col_end].count_nonzero()
                    if nr_similarities > 0:
                        counter += 1
                # entry S(A,B) = number of spectra A has that have at least one similar spectrum in B
                s[row_group_id, col_group_id] = counter
        # return s to calculate global distance matrix
        return s

    def write(self, work_dir: str, filename: str):
        s = self.similarity_sets
        # create a dataframe
        group_names = [group.get_group_name() for group in self.groups.get_groups()]
        df = pd.DataFrame(s, index=group_names, columns=group_names)

        # create path
        similarities = os.path.join(work_dir, "similarities")
        os.makedirs(similarities, exist_ok=True)
        path = os.path.join(similarities, f"{filename}.txt")

        # write S to a file in human-readable format
        logger.info(f"Writing similarity statistics to '{path}'.")
        with open(path, 'w') as f:
            df_string = df.to_string()
            f.write(df_string)
        # return s to calculate global distance matrix
        return s
