from pathlib import Path

from TreeMS2.ingestion.spectra_dataset.peak_file.readers.peak_file_reader import PeakFileReader, MGFReader


class ReaderFactory:
    def __init__(self):
        self.valid_extensions = [".mgf"]

    def get_reader(self, file_path: str) -> PeakFileReader:
        path = Path(file_path)
        file_extension = path.suffix.lower()
        match file_extension:
            case ".mgf":
                return MGFReader()
            case _:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. Supported types: {', '.join(self.valid_extensions)}"
                )
