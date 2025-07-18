from typing import List, Optional

import spectrum_utils.spectrum as sus

from TreeMS2.spectrum.spectrum_processing.processors.intensity_filter_processor import (
    IntensityFilterProcessor,
)
from TreeMS2.spectrum.spectrum_processing.processors.intensity_scaling_processor import (
    IntensityScalingProcessor,
    ScalingMethod,
)
from TreeMS2.spectrum.spectrum_processing.processors.mz_range_filter_processor import (
    MZRangeFilterProcessor,
)
from TreeMS2.spectrum.spectrum_processing.processors.precursor_peak_remover_processor import (
    PrecursorPeakRemoverProcessor,
)
from TreeMS2.spectrum.spectrum_processing.processors.spectrum_normalizer_processor import (
    SpectrumNormalizerProcessor,
)
from TreeMS2.spectrum.spectrum_processing.processors.spectrum_validator import (
    SpectrumValidator,
)
from TreeMS2.spectrum.spectrum_processing.spectrum_processor import SpectrumProcessor
from ...logger_config import get_logger

logger = get_logger(__name__)


# Processing pipeline that applies each processor in sequence
class SpectrumProcessingPipeline:
    def __init__(self, processors: List[SpectrumProcessor]):
        self.processors = processors

    def process(self, spectrum: sus.MsmsSpectrum) -> Optional[sus.MsmsSpectrum]:
        for processor in self.processors:  # Iterate over the list of processors
            spectrum = processor.process(spectrum)  # Apply each processor
            if spectrum is None:
                return None  # spectrum got invalidated
        return spectrum


# Pipeline Factory that creates the processing pipeline based on configuration
class ProcessingPipelineFactory:
    @staticmethod
    def create_pipeline(
        min_peaks: int,
        min_mz_range: float,
        remove_precursor_tol: float,
        min_intensity: float,
        max_peaks_used: int,
        scaling: ScalingMethod,
        min_mz: float,
        max_mz: float,
    ) -> SpectrumProcessingPipeline:
        processors = []
        validator = SpectrumValidator(min_peaks=min_peaks, min_mz_range=min_mz_range)
        processors.append(
            MZRangeFilterProcessor(mz_min=min_mz, mz_max=max_mz, validator=validator)
        )

        if remove_precursor_tol is not None:
            processors.append(
                PrecursorPeakRemoverProcessor(
                    remove_precursor_tolerance=remove_precursor_tol, validator=validator
                )
            )

        if min_intensity is not None or max_peaks_used is not None:
            processors.append(
                IntensityFilterProcessor(
                    min_intensity=min_intensity,
                    max_peaks_used=max_peaks_used,
                    validator=validator,
                )
            )

        if not scaling != ScalingMethod.OFF:
            processors.append(
                IntensityScalingProcessor(scaling=scaling, max_rank=max_peaks_used)
            )

        processors.append(SpectrumNormalizerProcessor())

        pipeline = SpectrumProcessingPipeline(processors=processors)

        return pipeline
