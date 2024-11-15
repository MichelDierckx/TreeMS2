import spectrum_utils.spectrum as sus

from .spectrum_validator import SpectrumValidator
from ..spectrum_processor import SpectrumProcessor


class PrecursorPeakRemoverProcessor(SpectrumProcessor):
    def __init__(self, remove_precursor_tolerance: float, validator: SpectrumValidator):
        super().__init__(validator)
        self.remove_precursor_tolerance = remove_precursor_tolerance

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        none_charge = spectrum.precursor_charge is None

        # Temporarily set the precursor charge to 1 to remove the precursor peak
        if none_charge:
            spectrum.precursor_charge = 1

        spectrum = spectrum.remove_precursor_peak(
            self.remove_precursor_tolerance, "Da", 0
        )
        if none_charge:
            spectrum.precursor_charge = None

        if self.remove_precursor_tolerance is not None:
            spectrum = spectrum.remove_precursor_peak(self.remove_precursor_tolerance, "Da", 0)
        return spectrum

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(remove_precursor_tolerance={self.remove_precursor_tolerance}, validator={self.validator})"
