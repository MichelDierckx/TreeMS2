from typing import Optional

from ...spectrum import Spectrum


class SpectrumValidatorProcessor:
    def __init__(self, min_peaks: int, min_mz_range: float):
        self.min_peaks = min_peaks
        self.min_mz_range = min_mz_range

    def process(self, spectrum: Spectrum) -> Optional[Spectrum]:
        if not _check_spectrum_valid(spectrum.mz, self.min_peaks, self.min_mz_range):
            return None
        return spectrum
