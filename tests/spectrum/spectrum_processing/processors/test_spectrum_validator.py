import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_processing.processors.spectrum_validator import (
    SpectrumValidator,
)


class TestSpectrumValidator(unittest.TestCase):
    def setUp(self):
        # Create a dummy spectrum with mz values for testing
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=float),
            intensity=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float),
            retention_time=10.0,
        )
        self.validator = SpectrumValidator(min_peaks=3, min_mz_range=200.0)

    def test_validate_valid_spectrum(self):
        # The spectrum meets the minimum peaks and mz range criteria
        is_valid = self.validator.validate(self.spectrum)
        self.assertTrue(is_valid, "Spectrum should be valid but was not.")

    def test_validate_too_few_peaks(self):
        # Create a spectrum with fewer peaks
        invalid_spectrum = sus.MsmsSpectrum(
            identifier="invalid_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0], dtype=float),
            intensity=np.array([10.0, 20.0], dtype=float),
            retention_time=10.0,
        )
        is_valid = self.validator.validate(invalid_spectrum)
        self.assertFalse(is_valid, "Spectrum should be invalid due to too few peaks.")

    def test_validate_too_small_mz_range(self):
        # Create a spectrum with a small mz range
        invalid_spectrum = sus.MsmsSpectrum(
            identifier="invalid_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([300.0, 310.0, 320.0], dtype=float),
            intensity=np.array([10.0, 20.0, 30.0], dtype=float),
            retention_time=10.0,
        )
        is_valid = self.validator.validate(invalid_spectrum)
        self.assertFalse(
            is_valid, "Spectrum should be invalid due to insufficient m/z range."
        )

    def test_validate_edge_case(self):
        # Create a spectrum that exactly meets the minimum requirements
        edge_case_spectrum = sus.MsmsSpectrum(
            identifier="edge_case_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0], dtype=float),
            intensity=np.array([10.0, 20.0, 30.0], dtype=float),
            retention_time=10.0,
        )
        edge_validator = SpectrumValidator(min_peaks=3, min_mz_range=200.0)
        is_valid = edge_validator.validate(edge_case_spectrum)
        self.assertTrue(
            is_valid,
            "Spectrum meeting exactly the minimum requirements should be valid.",
        )


if __name__ == "__main__":
    unittest.main()
