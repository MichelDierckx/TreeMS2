from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, List

from ...spectrum.group_spectrum import GroupSpectrum
from ...spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline


class PeakFile(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._id = None
        self._group_id = None

        self.total_spectra = 0
        self.failed_parsed = 0
        self.failed_processed = 0

        self.filtered: List[int] = []

    @abstractmethod
    def get_spectra(self, processing_pipeline: SpectrumProcessingPipeline) -> Iterable[GroupSpectrum]:
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

    def total_valid_spectra(self) -> int:
        return self.total_spectra - self.failed_parsed - self.failed_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "filename": self.file_path,
            "total_spectra": self.total_spectra,
            "failed_parsed": self.failed_parsed,
            "failed_processed": self.failed_processed,
            "filtered": self.filtered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeakFile":
        peak_file = cls(data["filename"])
        peak_file._id = data["id"]
        peak_file.total_spectra = data["total_spectra"]
        peak_file.failed_parsed = data["failed_parsed"]
        peak_file.failed_processed = data["failed_processed"]
        peak_file.filtered = data["filtered"]
        return peak_file
