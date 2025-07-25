import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats
from TreeMS2.ingestion.preprocessing.spectrum_preprocessor import SpectrumPreprocessor
from TreeMS2.ingestion.preprocessing.transformers import ScalingMethod


class TestSpectrumPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy ingestion
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
        self.spectrum_prepocessor = SpectrumPreprocessor(
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
        # Test that a valid ingestion is correctly processed
        quality_stats = QualityStats()
        processed_spectrum = self.spectrum_prepocessor.process(
            spec_id=1, spectrum=self.spectrum, quality_stats=quality_stats
        )

        # Expected results after the pipeline (manually calculated)
        expected_mz = np.array([200.0, 300.0, 350.0], dtype=float)
        expected_intensity = np.array([50.0, 300.0, 250.0], dtype=float)
        expected_intensity_normalized = expected_intensity / np.linalg.norm(
            expected_intensity
        )

        # Validate the processed ingestion
        self.assertIsNotNone(
            processed_spectrum, "Pipeline should return a processed ingestion."
        )
        np.testing.assert_array_equal(
            processed_spectrum.mz,
            expected_mz,
            "Processed ingestion mz values do not match.",
        )
        np.testing.assert_array_almost_equal(
            processed_spectrum.intensity,
            expected_intensity_normalized,
            err_msg="Processed ingestion intensity values do not match.",
        )
        self.assertEqual(
            len(processed_spectrum.mz), 3, "Incorrect number of peaks retained."
        )

    def test_pipeline_invalid_spectrum(self):
        # Create a ingestion that will fail validation (not enough peaks)
        invalid_spectrum = sus.MsmsSpectrum(
            identifier="invalid_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([100.0], dtype=float),
            intensity=np.array([10.0], dtype=float),
            retention_time=10.0,
        )

        quality_stats = QualityStats()
        processed_spectrum = self.spectrum_prepocessor.process(
            spec_id=1, spectrum=invalid_spectrum, quality_stats=quality_stats
        )

        # The pipeline should invalidate the ingestion
        self.assertIsNone(
            processed_spectrum,
            "Pipeline should invalidate a ingestion with insufficient peaks.",
        )

    def test_pipeline_edge_case(self):
        # Create a ingestion that barely passes validation
        edge_case_spectrum = sus.MsmsSpectrum(
            identifier="edge_case_spectrum",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([150.0, 250.0, 500], dtype=float),
            intensity=np.array([60.0, 60.0, 5.0], dtype=float),
            retention_time=10.0,
        )

        quality_stats = QualityStats()
        processed_spectrum = self.spectrum_prepocessor.process(
            spec_id=1, spectrum=edge_case_spectrum, quality_stats=quality_stats
        )

        # Validate that the ingestion passes the pipeline
        self.assertIsNotNone(
            processed_spectrum,
            "Pipeline should process a ingestion meeting minimum criteria.",
        )


if __name__ == "__main__":
    unittest.main()
