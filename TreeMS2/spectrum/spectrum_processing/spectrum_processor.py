# Base class for a processing step in the pipeline
from abc import ABC, abstractmethod

from ..spectrum import Spectrum


class SpectrumProcessor(ABC):
    @abstractmethod
    def process(self, spectrum: Spectrum) -> Spectrum:
        pass
