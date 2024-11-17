from typing import Optional

import spectrum_utils.spectrum as sus

from .spectrum_validator import SpectrumValidator
from ..spectrum_processor import SpectrumProcessor


class IntensityFilterProcessor(SpectrumProcessor):
    def __init__(self, min_intensity: Optional[float], max_peaks_used: Optional[int], validator: SpectrumValidator):
        super().__init__(validator)
        self.min_intensity: float = min_intensity or 0.0
        self.max_peaks_used = max_peaks_used

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        return spectrum.filter_intensity(self.min_intensity, self.max_peaks_used)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_intensity={self.min_intensity:.3f}, max_peaks_used={self.max_peaks_used}, validator={self.validator})"
