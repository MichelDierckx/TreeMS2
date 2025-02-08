from typing import List

import numba as nb

from TreeMS2.logger_config import get_logger
from TreeMS2.similarity_sets import SimilaritySets

logger = get_logger(__name__)


class Distances:
    def __init__(self, similarity_sets: SimilaritySets):
        self.similarity_sets = similarity_sets
        self.groups = self.similarity_sets.groups

    def create_mega(self, path: str, similarity_threshold: float):
        lines: List[str] = ["#mega",
                            f"TITLE: {self.groups.filename} (similarity_threshold={similarity_threshold}))",
                            ""]
        for group in self.groups.get_groups():
            lines.append(f"#{group.get_group_name()}")
        lines.extend(["", ""])

        # construct Lower-left triangular matrix
        for j in range(1, self.groups.get_size()):
            distances = []
            b = self.groups.get_group(j).total_spectra
            for i in range(j):
                a = self.groups.get_group(i).total_spectra
                s_a = self.similarity_sets.similarity_sets.item((i, j))
                s_b = self.similarity_sets.similarity_sets.item((j, i))
                global_similarity = _global_similarity(a, b, s_a, s_b)
                global_distance = _global_distance(a, b, global_similarity)
                distances.append(global_distance)
            line = "\t".join(f"{x:.4f}" for x in distances)
            lines.append(line)
        text = "\r\n".join(lines)

        logger.info(f"Writing distance matrix to '{path}'.")
        with open(path, 'w') as f:
            f.write(text)
        return


@nb.njit("f4(u4, u4, u4, u4)", cache=True)
def _global_similarity(a: int, b: int, s_a: int, s_b: int) -> float:
    """
    Computes the global similarity between two sets of tandem mass spectra.
    :param a: the number of spectra in set a
    :param b: the number of spectra in set b
    :param s_a: the number of spectra in set a that have at least one similar spectrum in set b
    :param s_b: the number of spectra in set b that have at least one similar spectrum in set a
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
