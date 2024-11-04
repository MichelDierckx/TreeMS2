from enum import Enum

from .csv_metadata_reader import CsvMetaDataReader
from .metadata_reader import MetaDataReader


class MetaDataFileFormat(Enum):
    """
    The supported metadata file formats.
    """
    CSV = 1


class MetaDataReaderFactory:
    """Factory class to create instances of MetaDataReader subclasses."""

    @staticmethod
    def create_reader(file_format: MetaDataFileFormat) -> MetaDataReader:
        """
        Creates an instance of a MetaDataReader based on the file format.

        :param file_format: The file format of the metadata file.
        :return: An instance of a MetaDataReader.
        :raises ValueError: If the file type is not supported.
        """
        # Check the file extension
        match file_format:
            case MetaDataFileFormat.CSV:
                return CsvMetaDataReader()
            case _:
                raise ValueError(f"Unsupported file format: {file_format.value}")
