import math
from typing import Tuple

import numba as nb

from ..spectrum import Spectrum
from ...config.spectrum_processing_config import SpectrumProcessingConfig


# Processing pipeline that applies each processor in sequence
class SpectrumProcessingPipeline:
    def __init__(self, processors: list):
        self.processors = processors

    def process(self, spectrum: Spectrum) -> Spectrum:
        for processor in self.processors:  # Iterate over the list of processors
            spectrum = processor.process(spectrum)  # Apply each processor
        return spectrum


# Pipeline Factory that creates the processing pipeline based on configuration
class ProcessingPipelineFactory:
    @staticmethod
    def create_pipeline(config: SpectrumProcessingConfig) -> SpectrumProcessingPipeline:
        processors = []

        vec_len, min_mz, max_mz = get_dim(
            config.min_mz, config.max_mz, config.fragment_tol
        )

        config.min_mz, config.max_mz, config.fragment_tol

        # Dynamically add processors based on config settings
        if config is not None and "processing" in config:
            processing_steps = config["processing"]

            if processing_steps.get("normalize_intensity", False):
                processors.append(IntensityNormalizationProcessor())  # Add to the list

            if "filter_low_intensity" in processing_steps:
                threshold = processing_steps["filter_low_intensity"].get("threshold", 0.1)
                processors.append(LowIntensityFilterProcessor(threshold))

            # Add more processors here as needed based on the config options
            # For example:
            # if processing_steps.get("other_processing", False):
            #     processors.append(OtherProcessor(...))

        # Return the constructed pipeline with all relevant processors
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
