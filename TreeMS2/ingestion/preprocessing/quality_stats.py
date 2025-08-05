from typing import List


class QualityStats:
    def __init__(self):
        self.high_quality = 0
        self.low_quality_ids = []

        self.filtered_after_reading = 0
        self.filtered_after_restricting_mz_range = 0
        self.filtered_after_removing_precursor_peak_noise = 0
        self.filtered_after_removing_low_intensity_peaks = 0

        self.too_few_peaks = 0
        self.too_small_mz_range = 0

    def add_high_quality(self, count=1):
        self.high_quality += count

    def add_low_quality(self, spec_id):
        self.low_quality_ids.append(spec_id)

    @property
    def low_quality_count(self):
        return len(self.low_quality_ids)


class QualityStatsCounts:
    def __init__(
        self,
        high_quality: int = 0,
        low_quality: int = 0,
        filtered_after_reading: int = 0,
        filtered_after_restricting_mz_range: int = 0,
        filtered_after_removing_precursor_peak_noise: int = 0,
        filtered_after_removing_low_intensity_peaks: int = 0,
        too_few_peaks: int = 0,
        too_small_mz_range: int = 0,
    ):
        self.high_quality = high_quality
        self.low_quality = low_quality
        self.filtered_after_reading = filtered_after_reading
        self.filtered_after_restricting_mz_range = filtered_after_restricting_mz_range
        self.filtered_after_removing_precursor_peak_noise = (
            filtered_after_removing_precursor_peak_noise
        )
        self.filtered_after_removing_low_intensity_peaks = (
            filtered_after_removing_low_intensity_peaks
        )
        self.too_few_peaks = too_few_peaks
        self.too_small_mz_range = too_small_mz_range

    @classmethod
    def from_quality_stats_list(
        cls, stats_list: List["QualityStats"]
    ) -> "QualityStatsCounts":
        total_high_quality = 0
        total_low_quality = 0
        filtered_after_reading = 0
        filtered_after_restricting_mz_range = 0
        filtered_after_removing_precursor_peak_noise = 0
        filtered_after_removing_low_intensity_peaks = 0
        too_few_peaks = 0
        too_small_mz_range = 0

        for stats in stats_list:
            total_high_quality += stats.high_quality
            total_low_quality += stats.low_quality_count
            filtered_after_reading += stats.filtered_after_reading
            filtered_after_restricting_mz_range += (
                stats.filtered_after_restricting_mz_range
            )
            filtered_after_removing_precursor_peak_noise += (
                stats.filtered_after_removing_precursor_peak_noise
            )
            filtered_after_removing_low_intensity_peaks += (
                stats.filtered_after_removing_low_intensity_peaks
            )
            too_few_peaks += stats.too_few_peaks
            too_small_mz_range += stats.too_small_mz_range

        return cls(
            total_high_quality,
            total_low_quality,
            filtered_after_reading,
            filtered_after_restricting_mz_range,
            filtered_after_removing_precursor_peak_noise,
            filtered_after_removing_low_intensity_peaks,
            too_few_peaks,
            too_small_mz_range,
        )
