import math
from typing import Tuple

import numba as nb
import spectrum_utils.spectrum as sus

from .processors.intensity_filter_processor import IntensityFilterProcessor
from .processors.intensity_scaling_processor import IntensityScalingProcessor
from .processors.mz_range_filter_processor import MZRangeFilterProcessor
from .processors.precursor_peak_remove_processor import PrecursorPeakRemoverProcessor
from .processors.spectrum_normalizer_processor import SpectrumNormalizerProcessor
from .processors.spectrum_validator import SpectrumValidator
from ...config.spectrum_processing_config import SpectrumProcessingConfig, ScalingMethod


# Processing pipeline that applies each processor in sequence
class SpectrumProcessingPipeline:
    def __init__(self, processors: list):
        self.processors = processors

    def process(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        for processor in self.processors:  # Iterate over the list of processors
            spectrum = processor.change(spectrum)  # Apply each processor
        return spectrum


# Pipeline Factory that creates the processing pipeline based on configuration
class ProcessingPipelineFactory:
    @staticmethod
    def create_pipeline(config: SpectrumProcessingConfig) -> SpectrumProcessingPipeline:
        processors = []

        vec_len, min_mz, max_mz = get_dim(
            config.min_mz, config.max_mz, config.fragment_tol
        )

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

        return SpectrumProcessingPipeline(processors)


@nb.njit("Tuple((u4, f4, f4))(f4, f4, f4)", cache=True)
def get_dim(
        min_mz: float, max_mz: float, bin_size: float
) -> Tuple[int, float, float]:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum mass in the mass range (inclusive).
    max_mz : float
        The maximum mass in the mass range (inclusive).
    bin_size : float
        The bin size (in Da).

    Returns
    -------
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim
