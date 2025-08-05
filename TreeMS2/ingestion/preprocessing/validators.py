import numba as nb
import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats


class SpectrumValidator:
    def __init__(self, min_peaks: int, min_mz_range: float):
        self.min_peaks = min_peaks
        self.min_mz_range = min_mz_range

    def validate(self, spectrum: sus.MsmsSpectrum, quality_stats: QualityStats) -> bool:
        valid = True
        if not _check_enough_peaks(spectrum.mz, self.min_peaks):
            quality_stats.too_few_peaks += 1
            valid = False
        if not _check_spectrum_coverage(spectrum.mz, self.min_mz_range):
            quality_stats.too_small_mz_range += 1
            valid = False
        return valid


@nb.njit(cache=True)
def _check_enough_peaks(spectrum_mz: np.ndarray, min_peaks: int) -> bool:
    """
    Check whether a cluster has enough peaks.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the cluster whose quality is checked.
    min_peaks : int
        Minimum number of peaks the cluster has to contain.

    Returns
    -------
    bool
        True if the cluster has enough peaks, False otherwise.
    """
    return len(spectrum_mz) >= min_peaks


@nb.njit(cache=True)
def _check_spectrum_coverage(spectrum_mz: np.ndarray, min_mz_range: float) -> bool:
    """
    Check whether a cluster covers a wide enough mass range.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the cluster whose quality is checked.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover.

    Returns
    -------
    bool
        True if the cluster covers a wide enough mass
        range, False otherwise.
    """
    return spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range
