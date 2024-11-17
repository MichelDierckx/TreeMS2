from typing import List

from TreeMS2.peak_file.peak_file import PeakFile


class Group:
    def __init__(self, group_name: str):
        self._group_name = group_name
        self._id = None
        self._peak_files: List[PeakFile] = []

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
