import numba as nb
import numpy as np
import spectrum_utils.spectrum as sus

from ..spectrum_processor import SpectrumProcessor


class SpectrumNormalizerProcessor(SpectrumProcessor):
    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        spectrum._intensity = _norm_intensity(spectrum.intensity)
        new_spectrum = sus.MsmsSpectrum(
            # A unique identifier or title for the spectrum (often representing the filename or a descriptor of the experiment).
            spectrum.identifier,
            # The mass-to-charge ratio (m/z) of the precursor ion (the peptide ion before fragmentation).
            spectrum.precursor_mz,
            # The charge state of the precursor ion (usually represented as a positive integer).
            spectrum.precursor_charge,
            # Peak data (column 1) : The m/z values, which represent the mass-to-charge ratio of the detected ions (peptide fragments).
            spectrum.mz,
            # Peak data (column 2) The intensity values, which represent the relative abundance or intensity of the corresponding ion.
            _norm_intensity(spectrum.intensity),
            # the retention time of the precursor ion in the chromatographic separation step before mass spectrometry analysis
            spectrum.retention_time,
        )
        return new_spectrum


@nb.njit(cache=True)
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize cluster peak intensities by their vector norm.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The cluster peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)