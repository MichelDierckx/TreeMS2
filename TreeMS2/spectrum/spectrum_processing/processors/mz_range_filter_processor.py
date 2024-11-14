from typing import Optional

import spectrum_utils.spectrum as sus

from .spectrum_validator import SpectrumValidator
from ..spectrum_processor import SpectrumProcessor


class MZRangeFilterProcessor(SpectrumProcessor):
    def __init__(self, mz_min: Optional[float], mz_max: Optional[float], validator: SpectrumValidator):
        super().__init__(validator)
        self.mz_min = mz_min
        self.mz_max = mz_max

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        spectrum = spectrum.set_mz_range(self.mz_min, self.mz_max)
        return spectrum