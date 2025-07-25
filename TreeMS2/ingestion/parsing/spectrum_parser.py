from typing import Dict, Optional

import spectrum_utils.spectrum as sus

from TreeMS2.ingestion.parsing.parsing_stats import ParsingStats


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
            # A unique identifier or title for the ingestion (often representing the filename or a descriptor of the experiment).
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
    def parse(
        spec_id: int, spectrum_dict: Dict, parsing_stats: ParsingStats
    ) -> Optional[sus.MsmsSpectrum]:
        try:
            # Parse the ingestion into an MsmsSpectrum instance
            parsed_spectrum = SpectrumParser._parse_spectrum(spectrum_dict)
            parsing_stats.add_valid()
            parsing_stats.add_precursor_charge(parsed_spectrum.precursor_charge, 1)
            return parsed_spectrum
        except (ValueError, KeyError) as e:
            parsing_stats.add_invalid(spec_id)
            return None
