import os
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz

from TreeMS2.groups.groups import Groups


class Distances:
    def __init__(self, nr_spectra: int, work_dir: str):
        self.base_path = os.path.join(work_dir, "distances")
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.dim = nr_spectra
        self.similarity_matrix: csr_matrix = csr_matrix((nr_spectra, nr_spectra), dtype=np.bool_)

    def update(self, data: npt.NDArray[np.bool_], rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]):
        self.similarity_matrix += csr_matrix((data, (rows, cols)), (self.dim, self.dim))

    def nr_bytes(self):
        return self.similarity_matrix.data.nbytes + self.similarity_matrix.indptr.nbytes + self.similarity_matrix.indices.nbytes

    def write_similarity_matrix(self):
        os.path.join(self.base_path, "similarity_matrix.npz")
        save_npz(self.base_path, self.similarity_matrix)

    @classmethod
    def load(cls, work_dir: str) -> 'Distances':
        base_path = os.path.join(work_dir, "distances")
        file = os.path.join(base_path, "similarity_matrix.npz")

        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")

        # Load the sparse matrix
        matrix = load_npz(file)

        # Create a new instance of DistanceMatrix
        instance = cls(nr_spectra=matrix.shape[0], work_dir=work_dir)

        # Set the loaded matrix and update the state
        instance.similarity_matrix = matrix

        return instance

    def write_similarity_sets_counts(self, groups: Groups):
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
                    nr_similarities = self.similarity_matrix[spec_id, col_begin:col_end].count_nonzero()
                    if nr_similarities > 0:
                        counter += 1
                # entry S(A,B) = number of spectra A has that have at least one similar spectrum in B
                s[row_group_id, col_group_id] = counter

        # write S to a file in human-readable format
        group_names = [group.get_group_name() for group in groups.get_groups()]
        df = pd.DataFrame(s, index=group_names, columns=group_names)
        path = os.path.join(self.base_path, "s.txt")
        with open(path, 'w') as f:
            df_string = df.to_string()
            f.write(df_string)

        # return s to calculate global distance matrix
        return s

    @classmethod
    def create_mega(cls, s: npt.NDArray[np.uint64], groups: Groups, similarity_threshold: float):
        lines: List[str] = ["#mega", f"TITLE: {groups.filename} (similarity_threshold={similarity_threshold}))", "\n"]
        for group in groups.get_groups():
            lines.append(f"#{group.get_group_name()}")
        lines.append("\n")

        # TODO: implement mega file writing

        return
