import spectrum_utils.spectrum as sus


class Spectrum:
    def __init__(self, spectrum: sus.MsmsSpectrum):
        self.spectrum = spectrum
        self._id = None
        self._file_id = None
        self._group_id = None

    def set_id(self, spectrum_id: int):
        self._id = spectrum_id

    def get_id(self):
        return self._id

    def set_file_id(self, file_id: int):
        self._file_id = file_id

    def get_file_id(self):
        return self._file_id

    def set_group_id(self, group_id: int):
        self._group_id = group_id

    def get_group_id(self):
        return self._group_id
