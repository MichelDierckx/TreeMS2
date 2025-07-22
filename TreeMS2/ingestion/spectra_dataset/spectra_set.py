import os
from typing import List, Dict, Any, Tuple

from TreeMS2.ingestion.spectra_dataset.peak_file import parsing_stats
from TreeMS2.ingestion.spectra_dataset.peak_file.parsing_stats import ParsingStats, ParsingStatsCounts
from TreeMS2.ingestion.spectra_dataset.peak_file.peak_file import PeakFile
from TreeMS2.ingestion.spectra_dataset.peak_file.quality_stats import QualityStats, QualityStatsCounts


class SpectraSet:
    def __init__(self, label: str):
        self.label = label
        self._id = None
        self._peak_files: List[PeakFile] = []

    def set_id(self, file_id: int):
        self._id = file_id

    def get_id(self):
        return self._id

    def get_label(self):
        return self.label

    def get_peak_files(self):
        return self._peak_files

    def count_peak_files(self):
        return len(self._peak_files)

    def add(self, peak_file: PeakFile) -> PeakFile:
        peak_file.set_id(len(self._peak_files))
        peak_file.assign_to_spectra_set(self._id)
        self._peak_files.append(peak_file)
        return self._peak_files[-1]

    def get_peak_file(self, peak_file_id):
        return self._peak_files[peak_file_id]

    def get_stats(self) -> Tuple[ParsingStatsCounts, QualityStatsCounts]:
        parsing_stats_list = [pf.parsing_stats for pf in self._peak_files if pf.parsing_stats is not None]
        quality_stats_list = [pf.quality_stats for pf in self._peak_files if pf.quality_stats is not None]

        aggregated_parsing_stats = ParsingStatsCounts.from_parsing_stats_list(parsing_stats_list)
        aggregated_quality_stats = QualityStatsCounts.from_quality_stats_list(quality_stats_list)

        return aggregated_parsing_stats, aggregated_quality_stats


    def to_dict(self) -> Dict[str, Any]:
        parsing_stats_counts, quality_stats_counts = self.get_stats()

        return {
            "id": self._id,
            "label": self.label,
            "valid": {
                "count": parsing_stats_counts.valid,
                "high_quality": quality_stats_counts.high_quality,
                "low_quality": quality_stats_counts.low_quality,
            },
            "invalid": {
                "count": parsing_stats_counts.invalid,
            },
            "files": [file.to_dict() for file in self._peak_files],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectraSet":
        spectra_set = cls(data["label"])
        spectra_set._id = data["id"]
        for file in data["files"]:
            peak_file: PeakFile = PeakFile.from_dict(file)
            spectra_set._peak_files.append(peak_file)
        return spectra_set
