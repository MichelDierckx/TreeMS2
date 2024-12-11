from typing import Dict

import spectrum_utils.spectrum as sus


class GroupSpectrum:
    def __init__(self, spectrum: sus.MsmsSpectrum):
        self.spectrum = spectrum
        self._spectrum_id = None
        self._file_id = None
        self._group_id = None
        self.vector = None

    def set_id(self, spectrum_id: int):
        self._spectrum_id = spectrum_id

    def get_id(self):
        return self._spectrum_id

    def set_file_id(self, file_id: int):
        self._file_id = file_id

    def get_file_id(self):
        return self._file_id

    def set_group_id(self, group_id: int):
        self._group_id = group_id

    def get_group_id(self):
        return self._group_id

    def to_dict(self) -> Dict:
        return {
            "spectrum_id": self._spectrum_id,
            "file_id": self._file_id,
            "group_id": self._group_id,
            "identifier": self.spectrum.identifier,
            "precursor_mz": self.spectrum.precursor_mz,
            "precursor_charge": self.spectrum.precursor_charge,
            "mz": self.spectrum.mz,
            "intensity": self.spectrum.intensity,
            "retention_time": self.spectrum.retention_time,
            "vector": self.vector,
        }
