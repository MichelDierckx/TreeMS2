import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import IntensityScalingProcessor
from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import ScalingMethod


class TestIntensityScalingProcessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy spectrum with identifier, precursor_mz, charge, mz, and intensity values
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=float),
            intensity=np.array([10.0, 50.0, 300.0, 20.0, 150.0], dtype=float),
            retention_time=10.0,
        )

    def test_root_scaling(self):
        # Initialize the processor with root scaling
        processor = IntensityScalingProcessor(
            scaling=ScalingMethod.ROOT,
            max_rank=None  # Not used in root scaling
        )

        # Expected intensity values (square root of original intensities)
        expected_intensity = np.sqrt(self.spectrum.intensity).astype(np.float32)

        # Apply the scaling method
        scaled_spectrum = processor.process(self.spectrum)

        # Validate the results
        np.testing.assert_array_almost_equal(
            scaled_spectrum.intensity, expected_intensity, decimal=6,
            err_msg="Root scaling failed."
        )

    def test_log_scaling(self):
        # Initialize the processor with log scaling
        processor = IntensityScalingProcessor(
            scaling=ScalingMethod.LOG,
            max_rank=None  # Not used in log scaling
        )

        # Expected intensity values (logarithm base 2 of original intensities + 1)
        expected_intensity = (np.log1p(self.spectrum.intensity) / np.log(2)).astype(np.float32)

        # Apply the scaling method
        scaled_spectrum = processor.process(self.spectrum)

        # Validate the results
        np.testing.assert_array_almost_equal(
            scaled_spectrum.intensity, expected_intensity, decimal=6,
            err_msg="Log scaling failed."
        )

    def test_rank_scaling(self):
        # Initialize the processor with rank scaling
        max_rank = 5  # Define a maximum rank
        processor = IntensityScalingProcessor(
            scaling=ScalingMethod.RANK,
            max_rank=max_rank
        )

        # Expected rank values (inverse rank of intensities)
        expected_rank = max_rank - np.argsort(np.argsort(self.spectrum.intensity)[::-1])
        expected_intensity = expected_rank.astype(np.float32)

        # Apply the scaling method
        scaled_spectrum = processor.process(self.spectrum)

        # Validate the results
        np.testing.assert_array_equal(
            scaled_spectrum.intensity, expected_intensity,
            err_msg="Rank scaling failed."
        )

    def test_invalid_max_rank(self):
        # Initialize the processor with rank scaling but an invalid max_rank
        max_rank = 3  # Less than the number of peaks in the spectrum
        processor = IntensityScalingProcessor(
            scaling=ScalingMethod.RANK,
            max_rank=max_rank
        )

        # Expect a ValueError due to invalid max_rank
        with self.assertRaises(ValueError):
            processor.process(self.spectrum)


if __name__ == "__main__":
    unittest.main()
