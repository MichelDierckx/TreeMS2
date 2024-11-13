from typing import Optional

from ..spectrum_processor import SpectrumProcessor
from ...spectrum import Spectrum


class IntensityFilterProcessor(SpectrumProcessor):
    def __init__(self, min_intensity: Optional[float], max_peaks_used: Optional[int]):
        self.min_intensity = min_intensity or 0.0
        self.max_peaks_used = max_peaks_used

    def process(self, spectrum: Spectrum) -> Spectrum:
        return spectrum.filter_intensity(self.min_intensity, self.max_peaks_used)
