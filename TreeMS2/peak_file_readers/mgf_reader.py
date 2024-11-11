"""
This module contains the `MGFSpectrumReader` class for reading MS/MS
spectra in MGF format.

Note:
This code was adapted from the `falcon` repository for mgf io:
https://github.com/bittremieux/falcon

The original structure and approach were used as a reference, with modifications
to fit the specific needs of this project.
"""

from typing import Dict, IO, Iterable, Union

import pyteomics.mgf
import spectrum_utils.spectrum as sus

from .peak_file_reader import PeakFileReader


class MGFReader(PeakFileReader):
    """
    MGF spectrum reader class to handle MGF files for reading and writing MS/MS spectra.
    """
    VALID_EXTENSIONS = ['.mgf']

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
            # A unique identifier or title for the spectrum (often representing the filename or a descriptor of the experiment).
            identifier,
            # The mass-to-charge ratio (m/z) of the precursor ion (the peptide ion before fragmentation).
            precursor_mz,
            # The charge state of the precursor ion (usually represented as a positive integer).
            precursor_charge,
            # Peak data (column 1) : The m/z values, which represent the mass-to-charge ratio of the detected ions (peptide fragments).
            mz_array,
            # Peak data (column 2) The intensity values, which represent the relative abundance or intensity of the corresponding ion.
            intensity_array,
            # the retention time of the precursor ion in the chromatographic separation step before mass spectrometry analysis
            retention_time,
        )
