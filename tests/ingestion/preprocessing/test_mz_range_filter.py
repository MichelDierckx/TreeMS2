import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.filters import MZRangeFilter
from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats
from TreeMS2.ingestion.preprocessing.validators import SpectrumValidator


class TestMZRangeFilterProcessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy ingestion with identifier, precursor_mz, charge, mz, and intensity values
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=float),
            intensity=np.array([10.0, 50.0, 300.0, 20.0, 150.0], dtype=float),
            retention_time=10.0,
        )

        # Initialize the MZRangeFilterProcessor with a validator
        self.mz_min = 150.0  # Minimum m/z value
        self.mz_max = 450.0  # Maximum m/z value
        self.validator = SpectrumValidator(min_peaks=1, min_mz_range=50)
        self.processor = MZRangeFilter(
            mz_min=self.mz_min,
            mz_max=self.mz_max,
        )

    def test_process(self):
        # Apply the m/z range filtering
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expected filtered results
        expected_mz = np.array([200.0, 300.0, 400.0], dtype=float)
        expected_intensity = np.array([50.0, 300.0, 20.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered m/z values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )

    def test_no_mz_min(self):
        # Test when mz_min is None (only mz_max applied)
        self.processor.mz_min = None
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expected filtered results
        expected_mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=float)
        expected_intensity = np.array([10.0, 50.0, 300.0, 20.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered m/z values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )

    def test_no_mz_max(self):
        # Test when mz_max is None (only mz_min applied)
        self.processor.mz_max = None
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expected filtered results
        expected_mz = np.array([200.0, 300.0, 400.0, 500.0], dtype=float)
        expected_intensity = np.array([50.0, 300.0, 20.0, 150.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered m/z values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )

    def test_invalid_range(self):
        # Test when mz_min > mz_max
        self.processor.mz_min = 450.0
        self.processor.mz_max = 150.0
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expected behavior: inverted range is handled
        expected_mz = np.array([200.0, 300.0, 400.0], dtype=float)
        expected_intensity = np.array([50.0, 300.0, 20.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered m/z values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )

    def test_no_filtering(self):
        # Test when mz_min and mz_max are both None (no filtering)
        self.processor.mz_min = None
        self.processor.mz_max = None
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expect the original ingestion
        np.testing.assert_array_equal(
            filtered_spectrum.mz,
            self.spectrum.mz,
            "Filtered m/z values do not match the original.",
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            self.spectrum.intensity,
            "Filtered intensity values do not match the original.",
        )

    def test_all_peaks_removed(self):
        # Test when mz range excludes all peaks
        self.processor.mz_min = 600.0  # Higher than any m/z value
        self.processor.mz_max = 700.0
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Validate that the ingestion is invalid
        quality_stats = QualityStats()
        valid = self.validator.validate(filtered_spectrum, quality_stats)
        self.assertEqual(False, valid, "Spectrum should be invalidated.")


if __name__ == "__main__":
    unittest.main()
