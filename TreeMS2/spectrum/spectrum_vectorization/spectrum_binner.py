import math
from typing import List, Tuple

import numba as nb
import numpy as np
import scipy.sparse as ss
import spectrum_utils.spectrum as sus


class SpectrumBinner:
    """
    Bins spectra into high-dimensional sparse vectors.
    """

    def __init__(self, min_mz: float, max_mz: float, bin_size: float):
        if bin_size <= 0:
            raise ValueError("bin_size must be greater than 0.")
        if min_mz >= max_mz:
            raise ValueError("min_mz must be less than max_mz.")

        self.bin_size = bin_size
        self.dim, self.min_mz, self.max_mz = _get_dim(min_mz, max_mz, bin_size)

    def bin(self, spectra: List[sus.MsmsSpectrum]) -> ss.csr_matrix:
        """
        Convert spectra into sparse binned vectors.
        """
        mzs = [spec.mz for spec in spectra]
        intensities = [spec.intensity for spec in spectra]
        data, indices, indptr = _to_vector(mzs, intensities, self.min_mz, self.bin_size)
        return ss.csr_matrix(
            (data, indices, indptr), shape=(len(spectra), self.dim), dtype=np.float32
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bin_size={self.bin_size:.3f}, dim={self.dim}, min_mz={self.min_mz:.3f}, max_mz={self.max_mz:.3f})"


@nb.njit(cache=True)
def _to_vector(
        mzs: List[np.ndarray],
        intensities: List[np.ndarray],
        min_mz: float,
        bin_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spectra to sparse binned vectors for input into a vectorizer.
    """
    n_spectra = len(mzs)
    n_peaks = 0
    for mz in mzs:
        n_peaks += len(mz)

    data = np.zeros(n_peaks, dtype=np.float32)
    indices = np.zeros(n_peaks, dtype=np.int32)
    indptr = np.zeros(n_spectra + 1, dtype=np.int32)

    peak_idx = 0
    for spec_idx, (mz, intensity) in enumerate(zip(mzs, intensities)):
        n_peaks_spectrum = len(mz)
        data[peak_idx:peak_idx + n_peaks_spectrum] = intensity
        indices[peak_idx:peak_idx + n_peaks_spectrum] = np.floor((mz - min_mz) / bin_size).astype(np.int32)
        indptr[spec_idx + 1] = indptr[spec_idx] + n_peaks_spectrum
        peak_idx += n_peaks_spectrum

    return data, indices, indptr


@nb.njit("Tuple((u4, f4, f4))(f4, f4, f4)", cache=True)
def _get_dim(
        min_mz: float, max_mz: float, bin_size: float
) -> Tuple[int, float, float]:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum mass in the mass range (inclusive).
    max_mz : float
        The maximum mass in the mass range (inclusive).
    bin_size : float
        The bin size (in Da).

    Returns
    -------
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim
