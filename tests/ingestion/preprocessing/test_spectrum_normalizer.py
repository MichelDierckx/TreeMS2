import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.transformers import Normalizer


class TestSpectrumNormalizerProcessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy ingestion with identifier, precursor_mz, charge, mz, and intensity values
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=float),
            intensity=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float),
            retention_time=10.0,
        )

        # Initialize the SpectrumNormalizerProcessor
        self.processor = Normalizer()

    def test_process(self):
        # Apply the normalization
        normalized_spectrum = self.processor.process(self.spectrum)

        # Compute expected normalized intensities
        original_intensity = self.spectrum.intensity
        norm_factor = np.linalg.norm(original_intensity)
        expected_intensity = original_intensity / norm_factor

        # Validate the normalized ingestion
        np.testing.assert_array_almost_equal(
            normalized_spectrum.intensity,
            expected_intensity,
            err_msg="Normalized intensities do not match expected values.",
        )
        np.testing.assert_array_equal(
            normalized_spectrum.mz,
            self.spectrum.mz,
            err_msg="m/z values should not change during normalization.",
        )
        self.assertEqual(
            normalized_spectrum.precursor_mz,
            self.spectrum.precursor_mz,
            "Precursor m/z should remain unchanged.",
        )
        self.assertEqual(
            normalized_spectrum.precursor_charge,
            self.spectrum.precursor_charge,
            "Precursor charge should remain unchanged.",
        )

    def test_process_already_normalized(self):
        # Test with a ingestion that's already normalized
        norm_factor = np.linalg.norm(self.spectrum.intensity)
        normalized_intensity = self.spectrum.intensity / norm_factor
        self.spectrum._intensity = normalized_intensity

        # Apply the processor
        normalized_spectrum = self.processor.transform(self.spectrum)

        # Expect no change since the ingestion was already normalized
        np.testing.assert_array_almost_equal(
            normalized_spectrum.intensity,
            normalized_intensity,
            err_msg="Normalized intensities should remain unchanged for an already normalized ingestion.",
        )

    def test_process_all_zero_intensity(self):
        # Test a ingestion with all zero intensities
        self.spectrum._intensity = np.zeros_like(self.spectrum.intensity)
        # Create a dummy ingestion with identifier, precursor_mz, charge, mz, and intensity values
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=float),
            intensity=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            retention_time=10.0,
        )

        # Apply the processor
        normalized_spectrum = self.processor.process(self.spectrum)

        # Expect zero intensities to remain unchanged
        np.testing.assert_array_equal(
            normalized_spectrum.intensity,
            self.spectrum.intensity,
            err_msg="All zero intensities should remain unchanged after normalization.",
        )

    def test_process_single_peak(self):
        # Create a dummy ingestion with identifier, precursor_mz, charge, mz, and intensity values
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0], dtype=float),
            intensity=np.array([1.0], dtype=float),
            retention_time=10.0,
        )

        # Apply the processor
        normalized_spectrum = self.processor.process(self.spectrum)

        # Expect single peak intensity to remain 1
        np.testing.assert_array_equal(
            normalized_spectrum.intensity,
            np.array([1.0]),
            err_msg="Single peak intensity should remain 1 after normalization.",
        )


if __name__ == "__main__":
    unittest.main()
