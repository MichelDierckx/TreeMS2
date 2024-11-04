from abc import ABC, abstractmethod
from typing import Union, IO

from .sample_group_mapping import SampleGroupMapping


class MetaDataReader(ABC):
    """
    Abstract base class for metadata readers.
    """
    extension = None

    @abstractmethod
    def get_metadata(self, source: Union[IO, str]) -> SampleGroupMapping:
        """
        Abstract method to get metadata for samples from a file.

        :param source: Union[IO, str], the source (file name or file object) to read the metadata from.
        :return: SampleGroupMapping, a mapping that maps samples to groups.
        :raises:
        ValueError: The file does not contain data that allows for a mapping from sample files to groups.
        FileNotFoundError: source file does not exist.
        """
        raise NotImplementedError("Subclasses must implement this method.")
