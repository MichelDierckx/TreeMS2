from typing import Optional

import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.filters import (
    MZRangeFilter,
    PrecursorPeakFilter,
    IntensityFilter,
)
from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats
from TreeMS2.ingestion.preprocessing.transformers import (
    ScalingMethod,
    IntensityScaler,
    Normalizer,
)
from TreeMS2.ingestion.preprocessing.validators import SpectrumValidator


# Processing pipeline that applies each processor in sequence
class SpectrumPreprocessor:
    def __init__(
        self,
        min_peaks: int,
        min_mz_range: float,
        remove_precursor_tol: Optional[float],
        min_intensity: Optional[float],
        max_peaks_used: Optional[int],
        scaling: ScalingMethod,
        min_mz: float,
        max_mz: float,
    ):

        self.validator = SpectrumValidator(
            min_peaks=min_peaks, min_mz_range=min_mz_range
        )
        self.mz_range_filter = MZRangeFilter(mz_min=min_mz, mz_max=max_mz)
        if remove_precursor_tol is not None:
            self.precursor_peak_filter = PrecursorPeakFilter(
                remove_precursor_tolerance=remove_precursor_tol
            )
        else:
            self.precursor_peak_filter = None
        if min_intensity is not None or max_peaks_used is not None:
            self.intensity_filter = IntensityFilter(
                min_intensity=min_intensity,
                max_peaks_used=max_peaks_used,
            )
        else:
            self.intensity_filter = None

        if not scaling != ScalingMethod.OFF:
            self.intensity_scaler = IntensityScaler(
                scaling=scaling, max_rank=max_peaks_used
            )
        else:
            self.intensity_scaler = None
        self.normalizer = Normalizer()

    def process(
        self, spec_id: int, spectrum: sus.MsmsSpectrum, quality_stats: QualityStats
    ) -> Optional[sus.MsmsSpectrum]:

        if not self.validator.validate(spectrum, quality_stats):
            quality_stats.filtered_after_reading += 1
            quality_stats.add_low_quality(spec_id)
            return None

        spectrum = self.mz_range_filter.transform(spectrum)
        if not self.validator.validate(spectrum, quality_stats):
            quality_stats.filtered_after_restricting_mz_range += 1
            quality_stats.add_low_quality(spec_id)
            return None

        if self.precursor_peak_filter is not None:
            spectrum = self.precursor_peak_filter.transform(spectrum)
            if not self.validator.validate(spectrum, quality_stats):
                quality_stats.filtered_after_removing_precursor_peak_noise += 1
                quality_stats.add_low_quality(spec_id)
                return None

        if self.intensity_filter is not None:
            spectrum = self.intensity_filter.transform(spectrum)
            if not self.validator.validate(spectrum, quality_stats):
                quality_stats.filtered_after_removing_low_intensity_peaks += 1
                quality_stats.add_low_quality(spec_id)
                return None

        if self.intensity_scaler is not None:
            spectrum = self.intensity_scaler.transform(spectrum)

        spectrum = self.normalizer.transform(spectrum)
        quality_stats.add_high_quality()
        return spectrum
