from typing import Dict

import spectrum_utils.spectrum as sus


class TreeMS2Spectrum:
    def __init__(
        self,
        spectrum_id: int,
        file_id: int,
        group_id: int,
        spectrum: sus.MsmsSpectrum,
        scan_number: int,
        vector=None,
    ):
        self.spectrum_id = spectrum_id
        self.file_id = file_id
        self.group_id = group_id
        self.spectrum = spectrum
        self.vector = vector
        self.scan_number = scan_number

    def to_dict(self) -> Dict:
        return {
            "spectrum_title": self.spectrum.identifier,
            "spectrum_id": self.spectrum_id,
            "file_id": self.file_id,
            "group_id": self.group_id,
            "precursor_mz": self.spectrum.precursor_mz,
            "precursor_charge": self.spectrum.precursor_charge,
            "scan_number": self.scan_number,
            "vector": self.vector,
        }
