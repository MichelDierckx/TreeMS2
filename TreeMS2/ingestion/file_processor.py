import re
from typing import Optional

from TreeMS2.ingestion.batch_writer import BatchWriter
from TreeMS2.ingestion.parsing.parsing_stats import ParsingStats
from TreeMS2.ingestion.parsing.spectrum_parser import SpectrumParser
from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats
from TreeMS2.ingestion.preprocessing.spectrum_preprocessor import SpectrumPreprocessor
from TreeMS2.ingestion.spectra_sets.peak_file.peak_file import PeakFile
from TreeMS2.ingestion.spectra_sets.peak_file.readers.peak_file_reader import (
    PeakFileReader,
)
from TreeMS2.ingestion.spectra_sets.treems2_spectrum import TreeMS2Spectrum


def map_charge_to_vector_store(charge: Optional[int]) -> str:
    """Maps precursor charge to the name of the corresponding vector store."""
    if charge == 1:
        return "charge_1"
    elif charge == 2:
        return "charge_2"
    elif charge == 3:
        return "charge_3"
    elif charge is not None and charge >= 4:
        return "charge_4plus"
    return "charge_unknown"  # Covers cases where charge is None or not recognized


def parse_scan_number(line: str) -> int:
    match = re.search(r"\bscan=(\d+)\b", line)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Could not parse scan number")


class ProcessingResult:
    def __init__(
        self,
        spectra_set_id: int,
        file_id: int,
        parsing_stats: ParsingStats,
        quality_stats: QualityStats,
    ):
        self.spectra_set_id: int = spectra_set_id
        self.file_id = file_id
        self.parsing_stats = parsing_stats
        self.quality_stats = quality_stats


class FileProcessor:
    def __init__(
        self,
        reader: PeakFileReader,
        spectrum_preprocessor: SpectrumPreprocessor,
        batch_writer: BatchWriter,
    ):

        self.reader = reader
        self.spectrum_parser = SpectrumParser()
        self.spectrum_preprocessor = spectrum_preprocessor
        self.batch_writer = batch_writer

        self.parsing_stats = ParsingStats()
        self.quality_stats = QualityStats()

    def process(self, peak_file: PeakFile) -> ProcessingResult:
        for spec_id, raw_spectrum in self.reader.read(peak_file=peak_file):
            processed_spectrum = self._process_spectrum(spec_id, raw_spectrum)
            if processed_spectrum is not None:
                scan_number = parse_scan_number(processed_spectrum.identifier)

                treems2_spectrum = TreeMS2Spectrum(
                    spectrum_id=spec_id,
                    file_id=peak_file.get_id(),
                    group_id=peak_file.get_spectra_set_id(),
                    spectrum=processed_spectrum,
                    scan_number=scan_number,
                )
                target_store = map_charge_to_vector_store(
                    processed_spectrum.precursor_charge
                )
                self.batch_writer.add(target_store, treems2_spectrum)

        self.batch_writer.flush()
        return ProcessingResult(
            spectra_set_id=peak_file.get_spectra_set_id(),
            file_id=peak_file.get_id(),
            parsing_stats=self.parsing_stats,
            quality_stats=self.quality_stats,
        )

    def _process_spectrum(self, spec_id, raw):
        parsed = SpectrumParser.parse(spec_id, raw, self.parsing_stats)
        if parsed is None:
            return None
        processed = self.spectrum_preprocessor.process(
            spec_id, parsed, self.quality_stats
        )
        if processed is None:
            return None
        return processed
