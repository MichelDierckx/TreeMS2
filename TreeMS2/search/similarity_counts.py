import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from TreeMS2.config.logger_config import get_logger
from TreeMS2.ingestion.spectra_sets.spectra_sets import SpectraSets

logger = get_logger(__name__)


class SimilarityCounts:
    def __init__(self, groups: SpectraSets):
        self.groups = groups
        self.similarity_sets: npt.NDArray[np.uint64] = np.zeros(
            (self.groups.count_spectra_sets(), self.groups.count_spectra_sets()),
            dtype=np.uint64,
        )

    def write(self, path: str):
        s = self.similarity_sets
        # create a dataframe
        group_names = [group.get_label() for group in self.groups.get_spectra_sets()]
        df = pd.DataFrame(s, index=group_names, columns=group_names)

        # write S to a file in human-readable format
        with open(path, "w") as f:
            df_string = df.to_string()
            f.write(df_string)
        # return s to calculate global distance matrix
        return s

    @staticmethod
    def load(path: str, groups: SpectraSets) -> Optional["SimilarityCounts"]:
        """
        Load a SimilaritySets object from a file. If loading fails, return None.
        """
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, sep=r"\s+")
            expected_groups = [group.get_label() for group in groups.get_spectra_sets()]
            if list(df.index) != expected_groups or list(df.columns) != expected_groups:
                return None
            similarity_sets = df.to_numpy(dtype=np.uint64)
            similarity_set_obj = SimilarityCounts(groups=groups)
            similarity_set_obj.similarity_sets = similarity_sets
            return similarity_set_obj
        except Exception:
            return None

    def merge(self, other: "SimilarityCounts"):
        self.similarity_sets += other.similarity_sets


class SimilarityCountsUpdater:
    def __init__(self, group_ids: npt.NDArray[np.uint16]):
        self.group_ids = group_ids

    def update(
        self, query_ids, target_ids, similarity_counts: SimilarityCounts
    ) -> SimilarityCounts:
        row_group_ids = self.group_ids[query_ids]
        col_group_ids = self.group_ids[target_ids]

        pairs = np.vstack((row_group_ids, col_group_ids)).T
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

        similarity_counts.similarity_sets[
            unique_pairs[:, 0], unique_pairs[:, 1]
        ] += counts.astype(np.uint64)
        return similarity_counts
