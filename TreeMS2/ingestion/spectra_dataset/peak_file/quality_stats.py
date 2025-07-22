from typing import List

class QualityStats:
    def __init__(self):
        self.high_quality = 0
        self.low_quality_ids = []

    def add_high_quality(self, count=1):
        self.high_quality += count

    def add_low_quality(self, spec_id):
        self.low_quality_ids.append(spec_id)

    @property
    def low_quality_count(self):
        return len(self.low_quality_ids)




class QualityStatsCounts:
    def __init__(self, high_quality: int = 0, low_quality: int = 0):
        self.high_quality = high_quality
        self.low_quality = low_quality

    @classmethod
    def from_quality_stats_list(cls, stats_list: List["QualityStats"]) -> "QualityStatsCounts":
        total_high_quality = 0
        total_low_quality = 0

        for stats in stats_list:
            total_high_quality += stats.high_quality
            total_low_quality += stats.low_quality_count

        return cls(total_high_quality, total_low_quality)