from typing import Dict, Iterable

import pyteomics.mgf
import spectrum_utils.spectrum as sus

from .peak_file import PeakFile
from ..spectrum.group_spectrum import GroupSpectrum
from ..spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline


class MgfFile(PeakFile):
    def __init__(self, file_path: str):
        # Call the parent class constructor
        super().__init__(file_path)

    def get_spectra(self, processing_pipeline: SpectrumProcessingPipeline) -> Iterable[GroupSpectrum]:
        with pyteomics.mgf.MGF(self.file_path) as f_in:
            for spectrum_i, spectrum_dict in enumerate(f_in):
                self.total_spectra += 1  # Increment the total spectra counter
                try:
                    # Parse the spectrum into an MsmsSpectrum instance
                    msms_spectrum = MgfFile._parse_spectrum(spectrum_dict)
                except (ValueError, KeyError):
                    # If parsing fails, increment failed parsing counter and skip this spectrum
                    self.failed_parsed += 1
                    continue

                # Process the spectrum and skip if it returns None
                processed_spectrum = processing_pipeline.process(msms_spectrum)
                if processed_spectrum is None:
                    self.failed_processed += 1  # Increment failed processing counter
                    continue

                # Create a Spectrum instance and assign the correct file_id and group_id
                spectrum = GroupSpectrum(processed_spectrum)
                spectrum.set_id(spectrum_i)  # Assign the spectrum index as the spectrum id
                spectrum.set_file_id(self._id)  # Set the file_id for this spectrum
                spectrum.set_group_id(self._group_id)  # Set the group_id for this spectrum

                yield spectrum  # Yield the spectrum instance with the correct data

    @staticmethod
    def _parse_spectrum(spectrum_dict: Dict) -> sus.MsmsSpectrum:
        identifier = spectrum_dict["params"]["title"]
        mz_array = spectrum_dict["m/z array"]
        intensity_array = spectrum_dict["intensity array"]
        retention_time = float(spectrum_dict["params"].get("rtinseconds", -1))
        precursor_mz = float(spectrum_dict["params"]["pepmass"][0])
        precursor_charge = int(spectrum_dict["params"]["charge"][0]) if "charge" in spectrum_dict["params"] else 0

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
