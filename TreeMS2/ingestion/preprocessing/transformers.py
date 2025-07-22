from enum import Enum
import numba as nb
from typing import Optional, Literal

import numpy as np
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.pipeline_step import PipelineStep


class ScalingMethod(Enum):
    OFF = "off"
    ROOT = "root"
    LOG = "log"
    RANK = "rank"


class IntensityScaler(PipelineStep):
    def __init__(
        self,
        scaling: Literal[ScalingMethod.ROOT, ScalingMethod.LOG, ScalingMethod.RANK],
        max_rank: Optional[int],
    ):
        super().__init__()
        self.scaling = scaling
        self.max_rank = max_rank

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        return spectrum.scale_intensity(self.scaling.value, max_rank=self.max_rank)


class Normalizer(PipelineStep):
    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        spectrum._intensity = _norm_intensity(spectrum.intensity)
        new_spectrum = sus.MsmsSpectrum(
            # A unique identifier or title for the spectrum (often representing the filename or a descriptor of the experiment).
            identifier=spectrum.identifier,
            # The mass-to-charge ratio (m/z) of the precursor ion (the peptide ion before fragmentation).
            precursor_mz=spectrum.precursor_mz,
            # The charge state of the precursor ion (usually represented as a positive integer).
            precursor_charge=spectrum.precursor_charge,
            # Peak data (column 1) : The m/z values, which represent the mass-to-charge ratio of the detected ions (peptide fragments).
            mz=spectrum.mz,
            # Peak data (column 2) The intensity values, which represent the relative abundance or intensity of the corresponding ion.
            intensity=_norm_intensity(spectrum.intensity),
            # the retention time of the precursor ion in the chromatographic separation step before mass spectrometry analysis
            retention_time=spectrum.retention_time,
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
    norm = np.linalg.norm(spectrum_intensity)
    if norm == 0:
        return spectrum_intensity  # Return the same array if all intensities are zero
    return spectrum_intensity / norm
