from typing import List

import numba as nb

from TreeMS2.config.logger_config import get_logger
from TreeMS2.search.similarity_counts import SimilarityCounts

logger = get_logger(__name__)


class DistanceMatrix:
    @staticmethod
    def create_mega(
        path: str,
        similarity_threshold: float,
        precursor_mz_window: float,
        similarity_sets: SimilarityCounts,
    ):
        lines: List[str] = [
            "#mega",
            f"TITLE: {similarity_sets.groups.spectra_sets_mapping_path} (similarity threshold={similarity_threshold}, precursor m/z window={precursor_mz_window}))",
            "",
        ]
        for group in similarity_sets.groups.get_spectra_sets():
            group_name = group.get_label().replace(" ", "_")
            lines.append(f"#{group_name}")
        lines.extend(["", ""])

        # construct Lower-left triangular matrix
        for j in range(1, similarity_sets.groups.count_spectra_sets()):
            distances = []
            b = similarity_sets.groups.get_spectra_set(j).get_stats()[1].high_quality
            for i in range(j):
                a = (
                    similarity_sets.groups.get_spectra_set(i)
                    .get_stats()[1]
                    .high_quality
                )
                s_a = similarity_sets.similarity_sets.item((i, j))
                s_b = similarity_sets.similarity_sets.item((j, i))
                global_similarity = _global_similarity(a, b, s_a, s_b)
                global_distance = _global_distance(a, b, global_similarity)
                distances.append(global_distance)
            line = "\t".join(f"{x:.4f}" for x in distances)
            lines.append(line)
        text = "\r\n".join(lines)

        with open(path, "w") as f:
            f.write(text)
        return


@nb.njit("f4(u4, u4, u4, u4)", cache=True)
def _global_similarity(a: int, b: int, s_a: int, s_b: int) -> float:
    """
    Computes the global similarity between two sets of tandem mass spectra.
    :param a: the number of spectra in set a
    :param b: the number of spectra in set b
    :param s_a: the number of spectra in set a that have at least one similar ingestion in set b
    :param s_b: the number of spectra in set b that have at least one similar ingestion in set a
    :return: the global similarity between set a and set b
    """
    return (s_a / (2 * a)) + (s_b / (2 * b))


@nb.njit("f4(u4, u4, f4)", cache=True)
def _global_distance(a: int, b: int, global_similarity: float) -> float:
    """
    Computes the global distance between two sets of tandem mass spectra.
    :param a: the number of spectra in set a
    :param b: the number of spectra in set b
    :param global_similarity: the global similarity between set a and set b
    :return: the global distance between set a and set b
    """
    if global_similarity > 0.0:
        return (1 / global_similarity) - 1
    elif global_similarity == 0.0:
        return ((4 * a * b) / (a + b)) - 1
    else:
        return -1.0
