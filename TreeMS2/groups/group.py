from typing import List

from TreeMS2.groups.peak_file.peak_file import PeakFile


class Group:
    def __init__(self, group_name: str):
        self._group_name = group_name
        self._id = None
        self._peak_files: List[PeakFile] = []

        self.total_spectra = 0
        self.failed_parsed = 0
        self.failed_processed = 0

        self.begin = 0
        self.end = 0

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

    def compute_spectrum_range(self, begin_id):
        self.begin = begin_id
        self.end = begin_id + self.total_spectra
        cur_id = self.begin
        for peak_file in self._peak_files:
            cur_id = peak_file.compute_spectrum_range(cur_id)
        return self.end

    def get_global_id(self, peak_file_id: int, spectrum_id: int) -> int:
        global_id = self._peak_files[peak_file_id].get_global_id(spectrum_id)
        return global_id

    def __repr__(self) -> str:
        files_repr = "\n\t".join([repr(file) for file in self._peak_files])
        return f"{self.__class__.__name__}(id={self._id}):\n\t{files_repr}"
