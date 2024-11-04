from typing import Dict, IO, Iterable, Union

import pyteomics.mgf
import spectrum_utils.spectrum as sus

from .spectrum_reader import SpectrumReader


class MGFSpectrumReader(SpectrumReader):
    """
    MGF spectrum reader class to handle MGF files for reading and writing MS/MS spectra.
    """

    @classmethod
    def get_spectra(cls, source: Union[IO, str]) -> Iterable[sus.MsmsSpectrum]:
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

        with pyteomics.mgf.MGF(source) as f_in:
            for spectrum_i, spectrum_dict in enumerate(f_in):
                try:
                    yield cls.__parse_spectrum(spectrum_dict)
                except (ValueError, KeyError):
                    pass

    @classmethod
    def __parse_spectrum(cls, spectrum_dict: Dict) -> sus.MsmsSpectrum:
        identifier = spectrum_dict["params"]["title"]
        mz_array = spectrum_dict["m/z array"]
        intensity_array = spectrum_dict["intensity array"]
        retention_time = float(spectrum_dict["params"].get("rtinseconds", -1))
        precursor_mz = float(spectrum_dict["params"]["pepmass"][0])

        precursor_charge = int(spectrum_dict["params"]["charge"][0]) if "charge" in spectrum_dict["params"] else None

        return sus.MsmsSpectrum(
            identifier,
            precursor_mz,
            precursor_charge,
            mz_array,
            intensity_array,
            retention_time,
        )
