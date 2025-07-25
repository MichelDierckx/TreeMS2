from typing import Optional

import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.preprocessing_step import PreprocessingStep
from TreeMS2.ingestion.preprocessing.validators import SpectrumValidator


class IntensityFilter(PreprocessingStep):
    def __init__(
        self,
        min_intensity: Optional[float],
        max_peaks_used: Optional[int],
        validator: SpectrumValidator,
    ):
        super().__init__(validator)
        self.min_intensity: float = min_intensity or 0.0
        self.max_peaks_used = max_peaks_used

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        return spectrum.filter_intensity(self.min_intensity, self.max_peaks_used)


class MZRangeFilter(PreprocessingStep):
    def __init__(
        self,
        mz_min: Optional[float],
        mz_max: Optional[float],
        validator: SpectrumValidator,
    ):
        super().__init__(validator)
        self.mz_min = mz_min
        self.mz_max = mz_max

    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        spectrum = spectrum.set_mz_range(self.mz_min, self.mz_max)
        return spectrum


class PrecursorPeakFilter(PreprocessingStep):
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
            spectrum = spectrum.remove_precursor_peak(
                self.remove_precursor_tolerance, "Da", 0
            )
        return spectrum
