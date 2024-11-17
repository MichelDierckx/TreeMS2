import numba as nb
import numpy as np
import spectrum_utils.spectrum as sus


class SpectrumValidator:
    def __init__(self, min_peaks: int, min_mz_range: float):
        self.min_peaks = min_peaks
        self.min_mz_range = min_mz_range

    def validate(self, spectrum: sus.MsmsSpectrum) -> bool:
        return _check_spectrum_valid(spectrum.mz, self.min_peaks, self.min_mz_range)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_peaks={self.min_peaks}, min_mz_range={self.min_mz_range:.3f})"


@nb.njit(cache=True)
def _check_spectrum_valid(
        spectrum_mz: np.ndarray, min_peaks: int, min_mz_range: float
) -> bool:
    """
    Check whether a cluster is of good enough quality to be used.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the cluster whose quality is checked.
    min_peaks : int
        Minimum number of peaks the cluster has to contain.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover.

    Returns
    -------
    bool
        True if the cluster has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (
            len(spectrum_mz) >= min_peaks
            and spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range
    )
