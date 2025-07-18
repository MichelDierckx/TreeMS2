from typing import Dict, Iterable

import pyteomics.mgf
import spectrum_utils.spectrum as sus

from TreeMS2.groups.peak_file.peak_file import PeakFile
from TreeMS2.logger_config import get_logger
from TreeMS2.spectrum.group_spectrum import GroupSpectrum
from TreeMS2.spectrum.spectrum_processing.pipeline import SpectrumProcessingPipeline

logger = get_logger(__name__)


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
                except (ValueError, KeyError) as e:
                    # If parsing fails, increment failed parsing counter and skip this spectrum
                    logger.warning(f"Error parsing spectrum {spectrum_i}: {e}")
                    self.failed_parsed += 1
                    self.filtered.append(spectrum_i)
                    continue

                # Process the spectrum and skip if it returns None
                processed_spectrum = processing_pipeline.process(msms_spectrum)
                if processed_spectrum is None:
                    self.failed_processed += 1  # Increment failed processing counter
                    self.filtered.append(spectrum_i)
                    continue

                # Create a Spectrum instance and assign the correct file_id and group_id
                spectrum = GroupSpectrum(spectrum_id=spectrum_i, file_id=self._id, group_id=self._group_id,
                                         spectrum=processed_spectrum)
                yield spectrum  # Yield the spectrum instance with the correct data

    @staticmethod
    def _parse_spectrum(spectrum_dict: Dict) -> sus.MsmsSpectrum:
        identifier = spectrum_dict["params"]["title"]
        mz_array = spectrum_dict["m/z array"]
        intensity_array = spectrum_dict["intensity array"]
        retention_time = float(spectrum_dict["params"].get("rtinseconds", -1))
        precursor_mz = float(spectrum_dict["params"]["pepmass"][0])
        precursor_charge = int(spectrum_dict["params"]["charge"][0]) if "charge" in spectrum_dict["params"] else 404

        return sus.MsmsSpectrum(
            # A unique identifier or title for the spectrum (often representing the filename or a descriptor of the experiment).
            identifier=identifier,
            # The mass-to-charge ratio (m/z) of the precursor ion (the peptide ion before fragmentation).
            precursor_mz=precursor_mz,
            # The charge state of the precursor ion (usually represented as a positive integer).
            precursor_charge=precursor_charge,
            # Peak data (column 1) : The m/z values, which represent the mass-to-charge ratio of the detected ions (peptide fragments).
            mz=mz_array,
            # Peak data (column 2) The intensity values, which represent the relative abundance or intensity of the corresponding ion.
            intensity=intensity_array,
            # the retention time of the precursor ion in the chromatographic separation step before mass spectrometry analysis
            retention_time=retention_time,
        )
