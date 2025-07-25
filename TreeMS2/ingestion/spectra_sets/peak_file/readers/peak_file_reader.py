from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Any

import pyteomics.mgf

from TreeMS2.config.logger_config import get_logger
from TreeMS2.ingestion.spectra_sets.peak_file.peak_file import PeakFile

logger = get_logger(__name__)


class PeakFileReader(ABC):
    """Abstract base class for reading peak files."""

    @abstractmethod
    def read(self, peak_file: PeakFile) -> Iterable[Tuple[int, Any]]:
        """Read the contents of the peak file and populate its parsing stats."""
        pass


class MGFReader(PeakFileReader):
    def read(self, peak_file: PeakFile) -> Iterable[Tuple[int, Any]]:
        with pyteomics.mgf.MGF(peak_file.file_path) as f_in:
            for spectrum_i, spectrum_dict in enumerate(f_in):
                yield spectrum_i, spectrum_dict
