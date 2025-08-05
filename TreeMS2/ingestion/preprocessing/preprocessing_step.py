# Base class for a processing step in the pipeline
from abc import ABC, abstractmethod

import spectrum_utils.spectrum as sus


class PreprocessingStep(ABC):
    @abstractmethod
    def transform(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        pass
