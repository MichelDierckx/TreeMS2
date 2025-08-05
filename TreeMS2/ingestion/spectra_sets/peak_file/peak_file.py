from typing import Dict, Any, Optional

from TreeMS2.ingestion.parsing.parsing_stats import ParsingStats
from TreeMS2.ingestion.preprocessing.quality_stats import QualityStats


class PeakFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._id = None
        self._spectra_set_id = None

        self.parsing_stats: Optional[ParsingStats] = None
        self.quality_stats: Optional[QualityStats] = None

    def set_id(self, file_id: int):
        self._id = file_id

    def get_id(self):
        return self._id

    def assign_to_spectra_set(self, spectra_set_id: int):
        self._spectra_set_id = spectra_set_id

    def get_spectra_set_id(self):
        return self._spectra_set_id

    def mark_processed(self, parsing_stats: ParsingStats, quality_stats: QualityStats):
        self.parsing_stats = parsing_stats
        self.quality_stats = quality_stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "file_path": self.file_path,
            "spectra_set_id": self._spectra_set_id,
            "valid": {
                "count": self.parsing_stats.valid if self.parsing_stats else 0,
                "high_quality": {
                    "count": (
                        self.quality_stats.high_quality if self.quality_stats else 0
                    )
                },
                "low_quality": {
                    "count": (
                        self.quality_stats.low_quality_count
                        if self.quality_stats
                        else 0
                    ),
                    "too_few_peaks": (
                        self.quality_stats.too_few_peaks if self.quality_stats else 0
                    ),
                    "too_small_mz_range": (
                        self.quality_stats.too_small_mz_range
                        if self.quality_stats
                        else 0
                    ),
                    "low_quality_ids": (
                        self.quality_stats.low_quality_ids if self.quality_stats else []
                    ),
                    "filtered_after_reading": (
                        self.quality_stats.filtered_after_reading
                        if self.quality_stats
                        else 0
                    ),
                    "filtered_after_restricting_mz_range": (
                        self.quality_stats.filtered_after_restricting_mz_range
                        if self.quality_stats
                        else 0
                    ),
                    "filtered_after_removing_precursor_peak_noise": (
                        self.quality_stats.filtered_after_removing_precursor_peak_noise
                        if self.quality_stats
                        else 0
                    ),
                    "filtered_after_removing_low_intensity_peaks": (
                        self.quality_stats.filtered_after_removing_low_intensity_peaks
                        if self.quality_stats
                        else 0
                    ),
                },
            },
            "invalid": {
                "count": self.parsing_stats.invalid_count if self.parsing_stats else 0,
                "invalid_ids": (
                    self.parsing_stats.invalid_ids if self.parsing_stats else []
                ),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeakFile":
        peak_file = cls(data["file_path"])
        peak_file._id = data.get("id")
        peak_file._spectra_set_id = data.get("spectra_set_id")

        # Restore ParsingStats
        parsing_stats = ParsingStats()
        invalid_data = data.get("invalid", {})
        parsing_stats.valid = data.get("valid", {}).get("count", 0)
        parsing_stats.invalid_ids = invalid_data.get("invalid_ids", [])
        peak_file.parsing_stats = parsing_stats

        # Restore QualityStats
        quality_stats = QualityStats()
        valid_data = data.get("valid", {})
        high_quality_data = valid_data.get("high_quality", {})
        low_quality_data = valid_data.get("low_quality", {})

        quality_stats.high_quality = high_quality_data.get("count", 0)
        quality_stats.low_quality_ids = low_quality_data.get("low_quality_ids", [])
        quality_stats.filtered_after_reading = low_quality_data.get(
            "filtered_after_reading", 0
        )
        quality_stats.filtered_after_restricting_mz_range = low_quality_data.get(
            "filtered_after_restricting_mz_range", 0
        )
        quality_stats.filtered_after_removing_precursor_peak_noise = (
            low_quality_data.get("filtered_after_removing_precursor_peak_noise", 0)
        )
        quality_stats.filtered_after_removing_low_intensity_peaks = (
            low_quality_data.get("filtered_after_removing_low_intensity_peaks", 0)
        )
        quality_stats.too_few_peaks = low_quality_data.get("too_few_peaks", 0)
        quality_stats.too_small_mz_range = low_quality_data.get("too_small_mz_range", 0)

        peak_file.quality_stats = quality_stats
        return peak_file
