from abc import ABC, abstractmethod
from typing import IO, Union, Iterable

import spectrum_utils.spectrum as sus


class SpectrumReader(ABC):
    """
    Abstract base class for spectrum readers.
    """
    extension = None

    @abstractmethod
    def get_spectra(self, source: Union[IO, str]) -> Iterable[sus.MsmsSpectrum]:
        """
        Abstract method to get MS/MS spectra from a source.

        Parameters
        ----------
        source : Union[IO, str]
            The source (file name or file object) to read spectra from.

        Returns
        -------
        Iterable[MsmsSpectrum]
            An iterable of MsmsSpectrum objects.
        """
        pass
