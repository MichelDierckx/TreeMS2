import os
from typing import List, Dict, Any

from TreeMS2.groups.peak_file.mgf_file import MgfFile
from TreeMS2.groups.peak_file.peak_file import PeakFile


class Group:
    def __init__(self, group_name: str):
        self._group_name = group_name
        self._id = None
        self._peak_files: List[PeakFile] = []

        self.total_spectra = 0
        self.failed_parsed = 0
        self.failed_processed = 0

    def set_id(self, file_id: int):
        self._id = file_id

    def get_id(self):
        return self._id

    def get_group_name(self):
        return self._group_name

    def get_peak_files(self):
        return self._peak_files

    def get_size(self):
        return len(self._peak_files)

    def add(self, peak_file: PeakFile) -> PeakFile:
        peak_file.set_id(len(self._peak_files))
        peak_file.set_group_id(self._id)
        self._peak_files.append(peak_file)
        return self._peak_files[-1]

    def get_peak_file(self, peak_file_id):
        return self._peak_files[peak_file_id]

    def total_valid_spectra(self) -> int:
        return self.total_spectra - self.failed_parsed - self.failed_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._group_name,
            "total_spectra": self.total_spectra,
            "failed_parsed": self.failed_parsed,
            "failed_processed": self.failed_processed,
            "files": [file.to_dict() for file in self._peak_files],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Group":
        group = cls(data["name"])
        group._id = data["id"]
        group.total_spectra = data["total_spectra"]
        group.failed_parsed = data["failed_parsed"]
        group.failed_processed = data["failed_processed"]
        for file in data["files"]:
            _, file_extension = os.path.splitext(file["filename"])
            match file_extension:
                case ".mgf":
                    MgfFile.from_dict(file)
                case _:
                    continue
        return group
