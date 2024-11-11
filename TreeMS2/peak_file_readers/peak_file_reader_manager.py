from typing import Type, List, Dict, Tuple

from .mgf_reader import MGFReader
from .peak_file_reader import PeakFileReader


class PeakFileReaderManager:
    _reader_types: List[Type[PeakFileReader]] = [MGFReader]  # List of all file type subclasses

    def __init__(self):
        self._readers: Dict[Tuple[str], PeakFileReader] = {}  # Use tuple of extensions as dictionary key

    def _register_reader_for_extension(self, file_extension: str) -> None:
        """
        Register a reader class instance for a given file extension if not already registered.
        """
        file_extension = file_extension.lower()

        # Check if the reader for this extension is already registered
        for reader_cls in self._reader_types:
            if file_extension in reader_cls.VALID_EXTENSIONS:
                # Use tuple of valid extensions as key to the dictionary
                ext_tuple = tuple(reader_cls.VALID_EXTENSIONS)

                if ext_tuple not in self._readers:
                    # Register reader only if its extension tuple is not already in the dictionary
                    reader_instance = reader_cls()
                    self._readers[ext_tuple] = reader_instance
                return

        raise ValueError(f"No spectrum reader supports the file extension '{file_extension}'")

    def get_reader(self, file_extension: str) -> PeakFileReader:
        """
        Retrieve the reader for a specific file extension, registering it if necessary.
        """
        file_extension = file_extension.lower()
        # If reader is not registered for this extension, register it
        for reader_cls in self._reader_types:
            if file_extension in reader_cls.VALID_EXTENSIONS:
                ext_tuple = tuple(reader_cls.VALID_EXTENSIONS)
                if ext_tuple not in self._readers:
                    self._register_reader_for_extension(file_extension)
                return self._readers[ext_tuple]

        raise ValueError(f"No spectrum reader found for the file extension '{file_extension}'")

    @classmethod
    def get_all_valid_extensions(cls) -> List[str]:
        """
        Get a list of all valid file extensions for available spectrum readers.
        """
        all_extensions = []
        for file_type_cls in cls._reader_types:
            all_extensions.extend(file_type_cls.VALID_EXTENSIONS)
        return all_extensions
