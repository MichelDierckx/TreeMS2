from typing import Optional

from ...spectrum import Spectrum


class PrecursorPeakRemoverProcessor:
    def __init__(self, remove_precursor_tolerance: Optional[float]):
        self.remove_precursor_tolerance = remove_precursor_tolerance

    def process(self, spectrum: Spectrum) -> Spectrum:
        if self.remove_precursor_tolerance is not None:
            spectrum = spectrum.remove_precursor_peak(self.remove_precursor_tolerance, "Da", 0)
        return spectrum
