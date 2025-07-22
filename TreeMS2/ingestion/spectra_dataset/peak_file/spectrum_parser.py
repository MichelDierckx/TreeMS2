from typing import Dict, Optional
import spectrum_utils.spectrum as sus


class SpectrumParser:
    @staticmethod
    def _parse_spectrum(spectrum_dict: Dict) -> sus.MsmsSpectrum:
        identifier = spectrum_dict["params"]["title"]
        mz_array = spectrum_dict["m/z array"]
        intensity_array = spectrum_dict["intensity array"]
        retention_time = float(spectrum_dict["params"].get("rtinseconds", -1))
        precursor_mz = float(spectrum_dict["params"]["pepmass"][0])
        precursor_charge = (
            int(spectrum_dict["params"]["charge"][0])
            if "charge" in spectrum_dict["params"]
            else 404
        )

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

    @staticmethod
    def parse(spectrum_dict: Dict) -> Optional[sus.MsmsSpectrum]:
        try:
            # Parse the spectrum into an MsmsSpectrum instance
            msms_spectrum = SpectrumParser._parse_spectrum(spectrum_dict)
            return msms_spectrum
        except (ValueError, KeyError) as e:
            # If parsing fails, increment failed parsing counter and skip this spectrum
            return None