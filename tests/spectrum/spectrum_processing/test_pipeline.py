import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_processing.pipeline import ProcessingPipelineFactory
from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import (
    ScalingMethod,
)


class TestSpectrumProcessingPipeline(unittest.TestCase):
    def setUp(self):
        # Create a dummy spectrum
        self.spectrum = sus.MsmsSpectrum(
            identifier="test_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0, 350.0, 400.0, 500.0], dtype=float),
            intensity=np.array([70.0, 50.0, 300.0, 250.0, 20.0, 150.0], dtype=float),
            retention_time=10.0,
        )
        # Parameters for the pipeline
        self.min_mz = 150.0
        self.max_mz = 500.0

        # Create the processing pipeline
        self.pipeline = ProcessingPipelineFactory.create_pipeline(
            min_peaks=2,
            min_mz_range=100.0,
            remove_precursor_tol=0.1,
            min_intensity=0.1,
            max_peaks_used=3,
            scaling=ScalingMethod.RANK,
            min_mz=self.min_mz,
            max_mz=self.max_mz,
        )

    def test_pipeline_valid_spectrum(self):
        # Test that a valid spectrum is correctly processed
        processed_spectrum = self.pipeline.process(self.spectrum)

        # Expected results after the pipeline (manually calculated)
        expected_mz = np.array([200.0, 300.0, 350.0], dtype=float)
        expected_intensity = np.array([50.0, 300.0, 250.0], dtype=float)
        expected_intensity_normalized = expected_intensity / np.linalg.norm(
            expected_intensity
        )

        # Validate the processed spectrum
        self.assertIsNotNone(
            processed_spectrum, "Pipeline should return a processed spectrum."
        )
        np.testing.assert_array_equal(
            processed_spectrum.mz,
            expected_mz,
            "Processed spectrum mz values do not match.",
        )
        np.testing.assert_array_almost_equal(
            processed_spectrum.intensity,
            expected_intensity_normalized,
            err_msg="Processed spectrum intensity values do not match.",
        )
        self.assertEqual(
            len(processed_spectrum.mz), 3, "Incorrect number of peaks retained."
        )

    def test_pipeline_invalid_spectrum(self):
        # Create a spectrum that will fail validation (not enough peaks)
        invalid_spectrum = sus.MsmsSpectrum(
            identifier="invalid_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0], dtype=float),
            intensity=np.array([10.0], dtype=float),
            retention_time=10.0,
        )

        processed_spectrum = self.pipeline.process(invalid_spectrum)

        # The pipeline should invalidate the spectrum
        self.assertIsNone(
            processed_spectrum,
            "Pipeline should invalidate a spectrum with insufficient peaks.",
        )

    def test_pipeline_edge_case(self):
        # Create a spectrum that barely passes validation
        edge_case_spectrum = sus.MsmsSpectrum(
            identifier="edge_case_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([150.0, 250.0, 500], dtype=float),
            intensity=np.array([60.0, 60.0, 5.0], dtype=float),
            retention_time=10.0,
        )

        processed_spectrum = self.pipeline.process(edge_case_spectrum)

        # Validate that the spectrum passes the pipeline
        self.assertIsNotNone(
            processed_spectrum,
            "Pipeline should process a spectrum meeting minimum criteria.",
        )


if __name__ == "__main__":
    unittest.main()
