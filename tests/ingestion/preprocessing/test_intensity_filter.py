import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.filters import IntensityFilter
from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats
from TreeMS2.ingestion.preprocessing.validators import SpectrumValidator


class TestIntensityFilterProcessor(unittest.TestCase):
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

        # Initialize the IntensityFilterProcessor with parameters
        self.min_intensity = (
            0.10  # Minimum intensity threshold (10% of base peak -> 300/10 = 30)
        )
        self.max_peaks_used = 2  # Maximum number of peaks to retain
        self.validator = SpectrumValidator(min_peaks=1, min_mz_range=50)
        self.processor = IntensityFilter(
            min_intensity=self.min_intensity,
            max_peaks_used=self.max_peaks_used,
        )

    def test_change(self):
        # Apply the change (filtering) method
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expected filtered results
        expected_mz = np.array(
            [300.0, 500.0], dtype=float
        )  # Corresponding to intensities [50.0, 300.0, 150.0]
        expected_intensity = np.array([300.0, 150.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered mz values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )
        self.assertEqual(
            len(filtered_spectrum.mz),
            self.max_peaks_used,
            "Number of peaks retained is incorrect.",
        )

    def test_change_no_max_peaks(self):
        # Test when max_peaks_used is None (all peaks above intensity threshold retained)
        self.processor.max_peaks_used = None
        filtered_spectrum = self.processor.transform(self.spectrum)

        # Expected filtered results
        expected_mz = np.array([200.0, 300.0, 500.0], dtype=float)
        expected_intensity = np.array([50.0, 300.0, 150.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered mz values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )
        self.assertEqual(
            len(filtered_spectrum.mz),
            len(expected_mz),
            "Number of peaks retained is incorrect.",
        )

    def test_change_all_peaks_removed(self):
        # Test when min_intensity is set too high to retain any peaks
        self.processor.min_intensity = (
            1.1  # Set higher than any intensity in the ingestion
        )
        filtered_spectrum = self.processor.transform(self.spectrum)
        quality_stats = QualityStats()
        valid = self.validator.validate(filtered_spectrum, quality_stats)

        # Spectrum should be invalid
        self.assertEqual(valid, False)


if __name__ == "__main__":
    unittest.main()
