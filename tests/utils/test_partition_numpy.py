import unittest

import numpy as np
from numpy.testing import assert_array_equal

from TreeMS2.utils.utils import partition_numpy


class TestPartitionAndNormalizeNumpy(unittest.TestCase):

    def test_basic_partitioning(self):
        sorted_array = np.array([3, 8, 15, 20, 25, 30, 40], dtype=np.uint32)
        partition_limits = np.array([10, 20, 30], dtype=np.uint32)
        expected_result = [
            np.array([3, 8], dtype=np.uint32),  # Partition for [0-10]
            np.array([4, 9], dtype=np.uint32),  # Partition for [11-20]
            np.array([4, 9], dtype=np.uint32),  # Partition for [21-30]
        ]
        result = partition_numpy(sorted_array, partition_limits)
        print(result)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)

    def test_single_partition(self):
        sorted_array = np.array([5, 6, 7, 8, 9], dtype=np.uint32)
        partition_limits = np.array([10], dtype=np.uint32)
        expected_result = [
            np.array([5, 6, 7, 8, 9], dtype=np.uint32)  # All values in the first partition
        ]
        result = partition_numpy(sorted_array, partition_limits)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)

    def test_no_values_in_partition(self):
        sorted_array = np.array([15, 20, 25, 30, 35], dtype=np.uint32)
        partition_limits = np.array([10, 20, 30], dtype=np.uint32)
        expected_result = [
            np.array([], dtype=np.uint32),  # Empty partition for [0-10]
            np.array([4, 9], dtype=np.uint32),  # Partition for [11-20]
            np.array([4, 9], dtype=np.uint32),  # Partition for [21-30]
        ]
        result = partition_numpy(sorted_array, partition_limits)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)

    def test_values_beyond_last_limit(self):
        sorted_array = np.array([5, 10, 15, 20, 30, 40, 50], dtype=np.uint32)
        partition_limits = np.array([10, 20, 30], dtype=np.uint32)
        expected_result = [
            np.array([5, 10], dtype=np.uint32),  # Partition for [0-10]
            np.array([4, 9], dtype=np.uint32),  # Partition for [11-20]
            np.array([9], dtype=np.uint32),  # Partition for [21-30]
            np.array([10, 20], dtype=np.uint32)  # Values beyond the last partition
        ]
        result = partition_numpy(sorted_array, partition_limits)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)

    def test_empty_array(self):
        sorted_array = np.array([], dtype=np.uint32)
        partition_limits = np.array([10, 20, 30], dtype=np.uint32)
        expected_result = [
            np.array([], dtype=np.uint32),  # Empty partition for [0-10]
            np.array([], dtype=np.uint32),  # Empty partition for [11-20]
            np.array([], dtype=np.uint32),  # Empty partition for [21-30]
        ]
        result = partition_numpy(sorted_array, partition_limits)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)

    def test_single_element_partition(self):
        sorted_array = np.array([5], dtype=np.uint32)
        partition_limits = np.array([10], dtype=np.uint32)
        expected_result = [
            np.array([5], dtype=np.uint32)  # The only value in the first partition
        ]
        result = partition_numpy(sorted_array, partition_limits)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)

    def test_partition_with_all_elements_in_last_range(self):
        sorted_array = np.array([35, 40, 45], dtype=np.uint32)
        partition_limits = np.array([10, 20, 30], dtype=np.uint32)
        expected_result = [
            np.array([], dtype=np.uint32),  # Empty partition for [0-10]
            np.array([], dtype=np.uint32),  # Empty partition for [11-20]
            np.array([], dtype=np.uint32),  # Empty partition for [21-30]
            np.array([4, 9, 14], dtype=np.uint32)  # Normalized values in the last partition
        ]
        result = partition_numpy(sorted_array, partition_limits)
        for r, e in zip(result, expected_result):
            assert_array_equal(r, e)


if __name__ == '__main__':
    unittest.main()
