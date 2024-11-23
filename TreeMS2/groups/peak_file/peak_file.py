from abc import ABC, abstractmethod
from typing import Iterable

import spectrum_utils.spectrum as sus

from ...spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline


class PeakFile(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._id = None
        self._group_id = None

        self.total_spectra = 0
        self.failed_parsed = 0
        self.failed_processed = 0

        self.begin = 0
        self.end = 0

    @abstractmethod
    def get_spectra(self, processing_pipeline: SpectrumProcessingPipeline) -> Iterable[sus.MsmsSpectrum]:
        """
        Abstract method to read spectra from the file.
        Each subclass must implement this method.
        """
        pass

    def set_id(self, file_id: int):
        self._id = file_id

    def get_id(self):
        return self._id

    def set_group_id(self, group_id: int):
        self._group_id = group_id

    def get_group_id(self):
        return self._group_id

    def compute_spectrum_range(self, begin_id: int):
        self.begin = begin_id
        self.end = begin_id + self.total_spectra
        return self.end

    def get_global_id(self, spectrum_id: int) -> int:
        return self.begin + spectrum_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id}, filepath={self.file_path})"
