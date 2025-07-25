from typing import List, Optional

import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.filters import (
    MZRangeFilter,
    PrecursorPeakFilter,
    IntensityFilter,
)
from TreeMS2.ingestion.preprocessing.preprocessing_step import PreprocessingStep
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

        self.steps: List[PreprocessingStep] = []

        validator = SpectrumValidator(min_peaks=min_peaks, min_mz_range=min_mz_range)
        self.steps.append(
            MZRangeFilter(mz_min=min_mz, mz_max=max_mz, validator=validator)
        )

        if remove_precursor_tol is not None:
            self.steps.append(
                PrecursorPeakFilter(
                    remove_precursor_tolerance=remove_precursor_tol, validator=validator
                )
            )

        if min_intensity is not None or max_peaks_used is not None:
            self.steps.append(
                IntensityFilter(
                    min_intensity=min_intensity,
                    max_peaks_used=max_peaks_used,
                    validator=validator,
                )
            )

        if not scaling != ScalingMethod.OFF:
            self.steps.append(IntensityScaler(scaling=scaling, max_rank=max_peaks_used))

        self.steps.append(Normalizer())

    def process(
        self, spec_id: int, spectrum: sus.MsmsSpectrum, quality_stats: QualityStats
    ) -> Optional[sus.MsmsSpectrum]:
        for processor in self.steps:
            spectrum = processor.process(spectrum)
            if spectrum is None:
                quality_stats.add_low_quality(spec_id)
                return None
        quality_stats.add_high_quality()
        return spectrum
