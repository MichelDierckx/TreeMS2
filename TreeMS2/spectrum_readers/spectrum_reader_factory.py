from enum import Enum

from .mgf_spectrum_reader import MGFSpectrumReader
from .spectrum_reader import SpectrumReader


class SpectrumFileFormat(Enum):
    """
    The supported spectrum file formats.
    """
    MGF = 1


class SpectrumReaderFactory:
    """
    Factory for creating spectrum readers based on spectrum file format.
    """

    @staticmethod
    def create_reader(file_format: SpectrumFileFormat) -> SpectrumReader:
        """
        Create a spectrum reader for the specified file format.

        Parameters
        ----------
        file_format : SpectrumFileFormat
            The file format for which to create a reader (e.g., 'SpectrumFileFormat.MGF').

        Returns
        -------
        SpectrumReader
            An concrete instance of a SpectrumReader for the specified format.

        Raises
        ------
        ValueError
            If the file format is not supported.
        """
        match file_format:
            case SpectrumFileFormat.MGF:
                return MGFSpectrumReader()
            case _:
                raise ValueError(f"Unsupported file format: {file_format.value}")
