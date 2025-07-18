import unittest

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_vectorization.spectrum_binner import SpectrumBinner


class TestSpectrumBinner(unittest.TestCase):
    def setUp(self):
        # Define parameters for the SpectrumBinner
        self.max_mz = 10
        self.min_mz = 3.5
        self.bin_size = 1.5
        self.binner = SpectrumBinner(self.min_mz, self.max_mz, self.bin_size)

        # Define some test spectra
        self.spectra = [
            sus.MsmsSpectrum(
                identifier="spectrum_1",
                precursor_mz=450.0,
                precursor_charge=2,
                mz=np.array([3.5, 4, 6, 8, 8.5, 10, 10.5], dtype=float),
                intensity=np.array(
                    [100.0, 80.0, 200.0, 300.0, 50.0, 400.0, 20.0], dtype=float
                ),
                retention_time=10.0,
            ),
            sus.MsmsSpectrum(
                identifier="spectrum_2",
                precursor_mz=550.0,
                precursor_charge=2,
                mz=np.array([4, 5, 6, 7, 8, 9, 10], dtype=float),
                intensity=np.array(
                    [120.0, 110.0, 180.0, 310.0, 70.0, 390.0, 25.0], dtype=float
                ),
                retention_time=12.0,
            ),
        ]

    def test_bin_dimensions(self):
        # Verify the calculated dimensions of the binner
        self.assertEqual(
            self.binner.dim, 5, "Binner dimension calculation is incorrect."
        )
        self.assertEqual(
            self.binner.min_mz, 3, "Binner minimum mz calculation is incorrect."
        )
        self.assertEqual(
            self.binner.max_mz, 10.5, "Binner maximum mz calculation is incorrect."
        )
        self.assertEqual(self.binner.bin_size, 1.5, "Binner bin size is incorrect.")

    def test_bin_single_spectrum(self):
        # Test binning a single spectrum
        spectrum = self.spectra[0]
        binned_matrix = self.binner.bin([spectrum])

        # Verify the shape of the binned matrix
        self.assertEqual(
            binned_matrix.shape,
            (1, self.binner.dim),
            "Binned matrix shape is incorrect for a single spectrum.",
        )

        # Expected bin indices and intensities (manually calculated)
        # [3.5, 4, 6, 8, 8.5, 10, 10.5]
        # Bins: [3, 4.5), [4.5, 6), [6, 7.5), [7.5, 9), [9, 10.5), [10.5, 12)
        expected_indices = np.array([0, 0, 2, 3, 3, 4, 5], dtype=int)
        expected_intensities = np.array(
            [100.0, 80.0, 200.0, 300.0, 50.0, 400.0, 20.0], dtype=float
        )

        # Validate non-zero entries
        self.assertTrue(
            np.array_equal(binned_matrix.indices, expected_indices),
            "Binned indices do not match expected values for a single spectrum.",
        )
        self.assertTrue(
            np.array_equal(binned_matrix.data, expected_intensities),
            "Binned intensities do not match expected values for a single spectrum.",
        )

    def test_bin_multiple_spectra(self):
        # Test binning multiple spectra
        binned_matrix = self.binner.bin(self.spectra)

        # Verify the shape of the binned matrix
        self.assertEqual(
            binned_matrix.shape,
            (len(self.spectra), self.binner.dim),
            "Binned matrix shape is incorrect for multiple spectra.",
        )

        # Expected values for each spectrum
        # Spectrum 1: Bins: [3.5, 5.0), [5.0, 6.5), [6.5, 8.0), [8.0, 9.5), [9.5, 11.0)
        expected_indices_1 = np.array([0, 0, 2, 3, 3, 4, 5], dtype=int)
        expected_intensities_1 = np.array(
            [100.0, 80.0, 200.0, 300.0, 50.0, 400.0, 20.0], dtype=float
        )

        expected_indices_2 = np.array([0, 1, 2, 2, 3, 4, 4], dtype=int)
        expected_intensities_2 = np.array(
            [120.0, 110.0, 180.0, 310.0, 70.0, 390.0, 25.0], dtype=float
        )

        # Verify the non-zero entries for each spectrum
        for i, (expected_indices, expected_intensities) in enumerate(
            [
                (expected_indices_1, expected_intensities_1),
                (expected_indices_2, expected_intensities_2),
            ]
        ):
            row_start = binned_matrix.indptr[i]
            row_end = binned_matrix.indptr[i + 1]
            indices = binned_matrix.indices[row_start:row_end]
            data = binned_matrix.data[row_start:row_end]

            self.assertTrue(
                np.array_equal(indices, expected_indices),
                f"Binned indices for spectrum {i + 1} do not match expected values.",
            )
            self.assertTrue(
                np.array_equal(data, expected_intensities),
                f"Binned intensities for spectrum {i + 1} do not match expected values.",
            )

    def test_empty_spectrum(self):
        # Test binning an empty spectrum
        empty_spectrum = sus.MsmsSpectrum(
            identifier="empty",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array([], dtype=float),
            intensity=np.array([], dtype=float),
            retention_time=10.0,
        )
        binned_matrix = self.binner.bin([empty_spectrum])

        # Verify the binned matrix has no non-zero entries
        self.assertEqual(
            binned_matrix.nnz,
            0,
            "Binned matrix should have no non-zero entries for an empty spectrum.",
        )

    def test_invalid_bin_size(self):
        # Test creating a SpectrumBinner with an invalid bin size
        with self.assertRaises(ValueError):
            SpectrumBinner(self.min_mz, self.max_mz, bin_size=0.0)

    def test_invalid_mz_range(self):
        # Test creating a SpectrumBinner with invalid mz range
        with self.assertRaises(ValueError):
            SpectrumBinner(self.max_mz, self.min_mz, bin_size=self.bin_size)


if __name__ == "__main__":
    unittest.main()
