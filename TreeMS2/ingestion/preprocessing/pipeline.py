from typing import List, Optional
import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.filters import MZRangeFilter, PrecursorPeakFilter, IntensityFilter
from TreeMS2.ingestion.preprocessing.pipeline_step import PipelineStep
from TreeMS2.ingestion.preprocessing.transformers import ScalingMethod, IntensityScaler, Normalizer
from TreeMS2.ingestion.preprocessing.validators import SpectrumValidator


# Processing pipeline that applies each processor in sequence
class Pipeline:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def process(self, spectrum: sus.MsmsSpectrum) -> Optional[sus.MsmsSpectrum]:
        for processor in self.steps:
            spectrum = processor.process(spectrum)
            if spectrum is None:
                return None
        return spectrum


# Pipeline Factory that creates the processing pipeline based on configuration
class PipelineFactory:
    @staticmethod
    def create_pipeline(
        min_peaks: int,
        min_mz_range: float,
        remove_precursor_tol: Optional[float],
        min_intensity: Optional[float],
        max_peaks_used: Optional[int],
        scaling: ScalingMethod,
        min_mz: float,
        max_mz: float,
    ) -> Pipeline:
        steps = []
        validator = SpectrumValidator(min_peaks=min_peaks, min_mz_range=min_mz_range)
        steps.append(
            MZRangeFilter(mz_min=min_mz, mz_max=max_mz, validator=validator)
        )

        if remove_precursor_tol is not None:
            steps.append(
                PrecursorPeakFilter(
                    remove_precursor_tolerance=remove_precursor_tol, validator=validator
                )
            )

        if min_intensity is not None or max_peaks_used is not None:
            steps.append(
                IntensityFilter(
                    min_intensity=min_intensity,
                    max_peaks_used=max_peaks_used,
                    validator=validator,
                )
            )

        if not scaling != ScalingMethod.OFF:
            steps.append(
                IntensityScaler(scaling=scaling, max_rank=max_peaks_used)
            )

        steps.append(Normalizer())

        pipeline = Pipeline(steps=steps)

        return pipeline
