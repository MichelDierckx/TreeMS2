from typing import Type

from .spectrum_reader import SpectrumReader


class SpectrumReaderManager:
    def __init__(self):
        self._readers = {}

    def register_reader(self, spectrum_reader_cls: Type[SpectrumReader]) -> None:
        """
        Register a spectrum reader class.
        """
        self._readers[spectrum_reader_cls.extension] = spectrum_reader_cls()

    def get_reader(self, file_extension: str) -> SpectrumReader:
        """
        Retrieve the spectrum reader class for a specific file extension.
        """
        try:
            return self._readers[file_extension.lower()]
        except KeyError:
            raise ValueError(f"No spectrum reader registered for file extension '{file_extension.lower()}'")
