import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_processing.processors.precursor_peak_remover_processor import (
    PrecursorPeakRemoverProcessor,
)
from TreeMS2.spectrum.spectrum_processing.processors.spectrum_validator import (
    SpectrumValidator,
)


class TestPrecursorPeakRemoverProcessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy spectrum with identifier, precursor_mz, charge, mz, and intensity values
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,  # Precursor m/z
            precursor_charge=2,  # Precursor charge
            mz=np.array([100.0, 200.0, 300.0, 499.5, 500.0, 501.0, 600.0], dtype=float),
            intensity=np.array(
                [10.0, 50.0, 300.0, 20.0, 150.0, 30.0, 200.0], dtype=float
            ),
            retention_time=10.0,
        )

        # Initialize the PrecursorPeakRemoverProcessor with a tolerance
        self.remove_precursor_tolerance = 1.0  # Tolerance in Da
        self.validator = SpectrumValidator(min_peaks=1, min_mz_range=50)
        self.processor = PrecursorPeakRemoverProcessor(
            remove_precursor_tolerance=self.remove_precursor_tolerance,
            validator=self.validator,
        )

    def test_process(self):
        # Apply the precursor peak removal
        filtered_spectrum = self.processor.process(self.spectrum)

        # Expected filtered results
        expected_mz = np.array(
            [100.0, 200.0, 300.0, 600.0], dtype=float
        )  # Peaks near 499.5, 500.0, 501.0 are removed
        expected_intensity = np.array([10.0, 50.0, 300.0, 200.0], dtype=float)

        # Validate the results
        np.testing.assert_array_equal(
            filtered_spectrum.mz, expected_mz, "Filtered m/z values do not match."
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            expected_intensity,
            "Filtered intensity values do not match.",
        )

    def test_no_precursor_peak_removed(self):
        # Test when no peaks fall within the precursor tolerance
        self.processor.remove_precursor_tolerance = 0.1  # Tighter tolerance
        filtered_spectrum = self.processor.process(self.spectrum)

        # Expected results: No peaks removed
        np.testing.assert_array_equal(
            filtered_spectrum.mz,
            self.spectrum.mz,
            "Filtered m/z values do not match original.",
        )
        np.testing.assert_array_equal(
            filtered_spectrum.intensity,
            self.spectrum.intensity,
            "Filtered intensity values do not match original.",
        )

    def test_all_peaks_removed(self):
        # Test behavior when tolerance is set to a very large value
        self.processor.remove_precursor_tolerance = 1000.0  # Very large tolerance
        filtered_spectrum = self.processor.process(self.spectrum)

        # Expected behavior: all peaks removed except those far from precursor
        self.assertEqual(None, filtered_spectrum, "Spectrum should be invalidated.")


if __name__ == "__main__":
    unittest.main()
