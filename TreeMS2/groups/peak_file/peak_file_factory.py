from pathlib import Path

from TreeMS2.groups.peak_file.mgf_file import MgfFile
from TreeMS2.groups.peak_file.peak_file import PeakFile


class PeakFileFactory:

    def __init__(self):
        self.valid_extensions = [".mgf"]

    def create(self, file_path: str) -> PeakFile:
        path = Path(file_path)
        file_extension = path.suffix.lower()
        match file_extension:
            case ".mgf":
                return MgfFile(str(path))
            case _:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. Supported types: {', '.join(self.valid_extensions)}")
