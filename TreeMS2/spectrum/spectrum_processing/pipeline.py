from typing import List, Optional

import spectrum_utils.spectrum as sus

from .processors.intensity_filter_processor import IntensityFilterProcessor
from .processors.intensity_scaling_processor import IntensityScalingProcessor
from .processors.mz_range_filter_processor import MZRangeFilterProcessor
from .processors.precursor_peak_remove_processor import PrecursorPeakRemoverProcessor
from .processors.spectrum_normalizer_processor import SpectrumNormalizerProcessor
from .processors.spectrum_validator import SpectrumValidator
from .spectrum_processor import SpectrumProcessor
from ...config.spectrum_processing_config import SpectrumProcessingConfig, ScalingMethod
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

    def __repr__(self) -> str:
        """Provide a textual representation of the pipeline and its processors."""
        processors_repr = "\n  ".join([repr(processor) for processor in self.processors])
        return f"SpectrumProcessingPipeline with processors:\n  {processors_repr}"


# Pipeline Factory that creates the processing pipeline based on configuration
class ProcessingPipelineFactory:
    @staticmethod
    def create_pipeline(config: SpectrumProcessingConfig, min_mz, max_mz) -> SpectrumProcessingPipeline:
        processors = []

        validator = SpectrumValidator(min_peaks=config.min_peaks, min_mz_range=config.min_mz_range)
        processors.append(MZRangeFilterProcessor(mz_min=min_mz, mz_max=max_mz, validator=validator))

        if config.remove_precursor_tol is not None:
            processors.append(PrecursorPeakRemoverProcessor(remove_precursor_tolerance=config.remove_precursor_tol,
                                                            validator=validator))

        if config.min_intensity is not None or config.max_peaks_used is not None:
            processors.append(
                IntensityFilterProcessor(min_intensity=config.min_intensity, max_peaks_used=config.max_peaks_used,
                                         validator=validator))

        if not config.scaling != ScalingMethod.OFF:
            processors.append(IntensityScalingProcessor(scaling=config.scaling, max_rank=config.max_peaks_used))

        processors.append(SpectrumNormalizerProcessor())

        pipeline = SpectrumProcessingPipeline(processors=processors)
        logger.info(f"Created processing pipeline:\n{pipeline}")

        return pipeline
