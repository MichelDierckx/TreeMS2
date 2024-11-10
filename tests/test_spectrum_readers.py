import unittest

from TreeMS2.spectrum_readers.mgf_spectrum_reader import MGFSpectrumReader
# Import the necessary classes from the TreeMS2.spectrum_readers package
from TreeMS2.spectrum_readers.spectrum_reader_manager import SpectrumReaderManager


class TestUniqueExtensions(unittest.TestCase):
    def test_unique_extensions(self):
        """
        Test that there are no duplicate extensions across all readers.
        Each file extension should be unique across all SpectrumReader subclasses.
        """
        # List to collect all the extensions
        all_extensions = []

        # Iterate over all registered readers and collect their extensions
        for reader_cls in SpectrumReaderManager._reader_types:
            all_extensions.extend(reader_cls.VALID_EXTENSIONS)

        # Use a set to check for duplicates
        unique_extensions = set(all_extensions)

        # Assert that the number of unique extensions matches the number of total extensions
        self.assertEqual(len(all_extensions), len(unique_extensions),
                         "There are duplicate extensions across different readers.")

    def test_reader_registration(self):
        """
        Test that SpectrumReaderManager properly registers the reader for a file extension.
        """
        # Set up the manager
        manager = SpectrumReaderManager()

        # Register a reader for a known extension
        reader = manager.get_reader(".mgf")
        self.assertIsInstance(reader, MGFSpectrumReader)

        # Check that requesting a non-registered extension raises an error
        with self.assertRaises(ValueError):
            manager.get_reader(".txt")


if __name__ == "__main__":
    unittest.main()
