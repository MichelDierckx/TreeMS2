from typing import Optional
from collections import Counter

from TreeMS2.ingestion.batch_writer import BatchWriter
from TreeMS2.ingestion.spectra_dataset.peak_file.parsing_stats import ParsingStats
from TreeMS2.ingestion.spectra_dataset.peak_file.peak_file import PeakFile
from TreeMS2.ingestion.preprocessing.pipeline import Pipeline
from TreeMS2.ingestion.spectra_dataset.peak_file.quality_stats import QualityStats
from TreeMS2.ingestion.spectra_dataset.peak_file.readers.peak_file_reader import PeakFileReader
from TreeMS2.ingestion.spectra_dataset.peak_file.spectrum_parser import SpectrumParser
from TreeMS2.ingestion.spectra_dataset.treems2_spectrum import TreeMS2Spectrum
from TreeMS2.ingestion.vectorization.spectra_vector_transformer import SpectraVectorTransformer


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


class ProcessingResult:
    def __init__(self, file_id: int, parsing_stats: ParsingStats, quality_stats: QualityStats):
        self.file_id = file_id
        self.parsing_stats = parsing_stats
        self.quality_stats = quality_stats


class FileProcessor:
    def __init__(self, reader: PeakFileReader, pipeline: Pipeline, batch_writer: BatchWriter):

        self.reader = reader
        self.pipeline = pipeline
        self.batch_writer = batch_writer

        self.parsing_stats = ParsingStats()
        self.quality_stats = QualityStats()

    def process(self, peak_file: PeakFile) -> ProcessingResult:
        for spec_id, raw_spectrum in self.reader.read(peak_file=peak_file):
            processed_spectrum = self._process_spectrum(spec_id, raw_spectrum)
            if processed_spectrum is not None:
                treems2_spectrum = TreeMS2Spectrum(
                    spectrum_id=spec_id,
                    file_id=peak_file.get_id(),
                    group_id=peak_file.get_spectra_set_id(),
                    spectrum=processed_spectrum
                )
                target_store = map_charge_to_vector_store(processed_spectrum.precursor_charge)
                self.batch_writer.add(target_store, treems2_spectrum)

        self.batch_writer.flush()
        return ProcessingResult(
            file_id=peak_file.get_id(),
            parsing_stats=self.parsing_stats,
            quality_stats=self.quality_stats
        )

    def _process_spectrum(self, spec_id, raw):
        parsed = SpectrumParser.parse(raw)
        if parsed is None:
            self.parsing_stats.add_invalid(spec_id)
            return None
        self.parsing_stats.add_valid()
        self.parsing_stats.add_precursor_charge(parsed.precursor_charge, 1)
        processed = self.pipeline.process(parsed)
        if processed is None:
            self.quality_stats.add_low_quality(spec_id)
            return None
        self.quality_stats.add_high_quality()
        return processed