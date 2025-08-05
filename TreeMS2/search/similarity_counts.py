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
        """
        Parameters
        ----------
        group_ids : np.ndarray
            An array mapping each spectrum/vector ID to a group ID.
        """
        self.group_ids = group_ids

    def update(
        self,
        query_ids: npt.NDArray[np.int32],
        target_ids: npt.NDArray[np.int32],
        similarity_counts: SimilarityCounts,
    ) -> SimilarityCounts:
        """
        Update similarity_counts based on how many unique query group IDs
        have at least one hit in a target group ID.

        Each query group is only counted once per target group, even if
        multiple query vectors in the same group hit the same target group.

        Parameters
        ----------
        query_ids : np.ndarray
            Indices of the query vectors.
        target_ids : np.ndarray
            Indices of the matching target vectors.
        similarity_counts : SimilarityCounts
            Object containing the similarity_sets 2D matrix to update.

        Returns
        -------
        similarity_counts : SimilarityCounts
            Updated similarity counts.
        """

        # Map target vector IDs to their group IDs
        target_group_ids = self.group_ids[target_ids]

        # Form array of (query_vector_id, target_group_id) pairs
        pairs = np.stack((query_ids, target_group_ids), axis=1)

        # Keep only unique (query_id, target_group_id) pairs to prevent overcounting
        pairs = np.unique(pairs, axis=0)

        # Replace query vector IDs with their group IDs
        pairs[:, 0] = self.group_ids[pairs[:, 0]]

        # Collapse down to unique (query_group_id, target_group_id) pairs and count occurrences
        unique_group_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

        # Update the similarity matrix with counts
        similarity_counts.similarity_sets[
            unique_group_pairs[:, 0], unique_group_pairs[:, 1]
        ] += counts.astype(np.uint64)

        return similarity_counts
