from abc import ABC, abstractmethod
from typing import Iterable

import spectrum_utils.spectrum as sus


class PeakFile(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._id = None
        self._group_id = None

    @abstractmethod
    def get_spectra(self) -> Iterable[sus.MsmsSpectrum]:
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