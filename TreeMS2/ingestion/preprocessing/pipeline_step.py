# Base class for a processing step in the pipeline
from abc import ABC, abstractmethod
from typing import Optional

import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.preprocessing.validators import SpectrumValidator


class PipelineStep(ABC):

    def __init__(self, validator: Optional[SpectrumValidator] = None):
        self.validator: Optional[SpectrumValidator] = validator

    @abstractmethod
    def change(self, spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
        pass

    def validate(self, spectrum: sus.MsmsSpectrum) -> bool:
        if self.validator is not None:
            return self.validator.validate(spectrum)
        return True

    def process(self, spectrum: sus.MsmsSpectrum) -> Optional[sus.MsmsSpectrum]:
        spectrum = self.change(spectrum)
        if self.validate(spectrum):
            return spectrum
        return None