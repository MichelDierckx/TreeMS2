import unittest

from TreeMS2.vector_store.vector_store import _partition_integers


# 10 10 10

# 0-9 10-19 20-29


class TestPartitionAndNormalize(unittest.TestCase):

    def test_basic_partitioning(self):
        sorted_list = [3, 8, 15, 20, 25, 30, 40]
        partition_limits = [10, 20, 30]
        expected_result = [
            [3, 8],  # Partition for [0-10]
            [4, 9],  # Partition for [11-20]
            [4, 9],  # Partition for [21-30]
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)

    def test_single_partition(self):
        sorted_list = [5, 6, 7, 8, 9]
        partition_limits = [10]  # All values are in the range [0-10]
        expected_result = [
            [5, 6, 7, 8, 9]  # No normalization, as everything is in the first partition
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)

    def test_no_values_in_partition(self):
        sorted_list = [15, 20, 25, 30, 35]
        partition_limits = [10, 20, 30]  # Partitioning with limits, but no values in the first partition
        expected_result = [
            [],  # Empty partition [0-10]
            [4, 9],  # Partition for [11-20]
            [4, 9],  # Partition for [21-30] (normalized)
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)

    def test_values_beyond_last_limit(self):
        sorted_list = [5, 10, 15, 20, 30, 40, 50]
        partition_limits = [10, 20, 30]  # Defines partitions: [0-10], [11-20], [21-30]
        expected_result = [
            [5, 10],  # Partition for [0-10]
            [4, 9],  # Partition for [11-20] (normalized)
            [9],  # Partition for [21-30] (normalized)
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)

    def test_empty_list(self):
        sorted_list = []
        partition_limits = [10, 20, 30]  # Defines partitions: [0-10], [11-20], [21-30]
        expected_result = [
            [],  # Empty partitions
            [],
            [],
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)

    def test_single_element_partition(self):
        sorted_list = [5]
        partition_limits = [10]
        expected_result = [
            [5]  # The only value falls in the [0-10] range with no adjustment needed.
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)

    def test_partition_with_all_elements_in_last_range(self):
        sorted_list = [35, 40, 45]
        partition_limits = [10, 20, 30]
        expected_result = [
            [],  # Empty partition for [0-10]
            [],  # Empty partition for [11-20]
            [],  # Empty partition for [21-30]
        ]
        result = _partition_integers(sorted_list, partition_limits)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
