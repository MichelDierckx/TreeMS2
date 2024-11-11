from abc import ABC, abstractmethod
from typing import List

from .peak_file_group_mapping import PeakFileGroupMapping


class MetaDataReader(ABC):
    """
    Abstract base class for metadata readers.
    """
    VALID_EXTENSIONS: List[str] = []

    @abstractmethod
    def get_metadata(self, source: str) -> PeakFileGroupMapping:
        """
        Abstract method to get metadata for samples from a file.

        :param source: str, the path to the file to read the metadata from.
        :return: SampleGroupMapping, a mapping that maps samples to groups.
        :raises:
        ValueError: The file does not contain data that allows for a mapping from sample files to groups.
        FileNotFoundError: source file does not exist.
        """
        raise NotImplementedError("Subclasses must implement this method.")
