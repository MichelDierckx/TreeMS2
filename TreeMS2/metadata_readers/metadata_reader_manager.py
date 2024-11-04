from typing import Type

from .metadata_reader import MetaDataReader


class MetadataReaderManager:
    def __init__(self):
        self._readers = {}

    def register_reader(self, spectrum_reader_cls: Type[MetaDataReader]) -> None:
        """
        Register a metadata reader class.
        """
        self._readers[spectrum_reader_cls.extension] = spectrum_reader_cls()

    def get_reader(self, file_extension: str) -> MetaDataReader:
        """
        Retrieve the metadata reader class for a specific file extension.
        """
        try:
            return self._readers[file_extension.lower()]
        except KeyError:
            raise ValueError(f"No metadata reader registered for file extension '{file_extension.lower()}'")
