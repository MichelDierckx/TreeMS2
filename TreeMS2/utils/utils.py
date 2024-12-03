from typing import List

import numpy as np
import numpy.typing as npt


def get_sorted_unique_from_coordinates(coordinates: np.ndarray) -> np.ndarray:
    # Step 1: Extract x and y coordinates
    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    # Step 2: Combine x and y coordinates into a single array
    combined_coords = np.concatenate((rows, cols))

    # Step 3: Get unique coordinates and sort them
    unique_sorted_coords = np.unique(combined_coords)

    return unique_sorted_coords


def partition_pylist(sorted_list: List[int], partition_limits: List[int]) -> List[List[int]]:
    """
    Partitions a sorted list of integers into intervals based on specified partition limits and normalizes
    the values in each partition by subtracting the minimum value of the respective partition.

    The function divides the sorted input list into partitions where each partition is defined by a maximum
    value from `partition_limits`. Each partition contains integers that are within the range
    `[partition_limits[i-1] + 1, partition_limits[i]]` (for partition i, with the first partition starting at 0).
    The minimum value for each partition is subtracted from all integers in that partition.

    Args:
        sorted_list (List[int]): A sorted list of positive integers to be partitioned.
        partition_limits (List[int]): A sorted list of maximum values that define the upper bounds of each partition.
                                      The minimum value for partition 0 is assumed to be 0.
                                      Each partition `i` includes values between `partition_limits[i-1] + 1`
                                      and `partition_limits[i]`.

    Returns:
        List[List[int]]: A list of partitions, where each partition is represented by a list of integers.
                         Each partition contains the normalized values, which are the original values minus the
                         minimum value of the respective partition. The partitions are ordered according to
                         the partition limits.

    Example:
        sorted_list = [3, 8, 15, 20, 25, 30, 40]
        partition_limits = [10, 20, 30]

        result = partition_and_normalize(sorted_list, partition_limits)

        # Output:
        [
            [3, 8],         # Partition for [0-10]
            [4, 9],         # Partition for [11-20]
            [4, 9],         # Partition for [21-30]
            [9]             # Partition for [31+]
        ]
    """
    partitions = []
    partition_index = 0
    partition_min = partition_limits[partition_index - 1] + 1 if partition_index > 0 else 0
    partition_max = partition_limits[partition_index]

    number_of_partitions = len(partition_limits)
    for i in range(number_of_partitions):
        partitions.append([])

    for value in sorted_list:
        while value > partition_max:
            partition_index += 1
            if partition_index > len(partition_limits) - 1:
                return partitions

            partition_min = partition_limits[partition_index - 1] + 1 if partition_index > 0 else 0
            partition_max = partition_limits[partition_index]

        partitions[partition_index].append(value - partition_min)

    return partitions


def partition_numpy(
        sorted_array: npt.NDArray[np.int64],
        partition_limits: npt.NDArray[np.int64]
) -> List[npt.NDArray[np.uint32]]:
    """
    Partitions a sorted numpy array of unsigned integers into intervals based on specified partition limits
    and normalizes the values in each partition by subtracting the minimum value of the respective partition.

    The function divides the sorted input array into partitions where each partition is defined by a maximum
    value from `partition_limits`. Each partition contains integers that are within the range
    `[partition_limits[i-1] + 1, partition_limits[i]]` (for partition i, with the first partition starting at 0).
    The minimum value for each partition is subtracted from all integers in that partition.

    Args:
        sorted_array (npt.NDArray[np.uint32]): A sorted numpy array of positive integers to be partitioned.
        partition_limits (npt.NDArray[np.uint32]): A sorted numpy array defining the upper bounds of each partition.

    Returns:
        List[npt.NDArray[np.uint32]]: A list of numpy arrays, where each array represents a partition.
                                      Each partition contains the normalized values, which are the original
                                      values minus the minimum value of the respective partition.
    """
    partitions = []

    partition_index = 0

    for _ in partition_limits:
        partition_min = partition_limits[partition_index - 1] + 1 if partition_index > 0 else 0
        partition_max = partition_limits[partition_index]

        # Extract values within the current partition range
        start_idx = np.searchsorted(sorted_array, partition_min, side='left')
        end_idx = np.searchsorted(sorted_array, partition_max, side='right')
        mask = np.arange(len(sorted_array))[start_idx:end_idx]

        # mask = (sorted_array >= partition_min) & (sorted_array <= partition_max)
        partition = sorted_array[mask] - partition_min  # Normalize to start at 0
        partitions.append(partition)

        # Update the lower bound for the next partition
        partition_index += 1

    return partitions
