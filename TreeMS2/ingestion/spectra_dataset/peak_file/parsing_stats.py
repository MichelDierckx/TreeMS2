from typing import List, Dict
from collections import defaultdict

class ParsingStats:
    def __init__(self):
        self.valid = 0
        self.invalid_ids = []
        self.precursor_charge_counts = defaultdict(int)  # key: charge (int), value: count

    def add_valid(self, count=1):
        self.valid += count

    def add_invalid(self, spec_id):
        self.invalid_ids.append(spec_id)

    def add_precursor_charge(self, charge: int, count=1):
        self.precursor_charge_counts[charge] += count

    @property
    def invalid_count(self):
        return len(self.invalid_ids)

    def merge(self, other: "ParsingStats"):
        self.valid += other.valid
        self.invalid_ids.extend(other.invalid_ids)
        for charge, count in other.precursor_charge_counts.items():
            self.precursor_charge_counts[charge] += count

class ParsingStatsCounts:
    def __init__(self, valid: int = 0, invalid: int = 0, precursor_charge_counts: Dict[int, int] = None):
        self.valid = valid
        self.invalid = invalid
        self.precursor_charge_counts = precursor_charge_counts or defaultdict(int)

    @classmethod
    def from_parsing_stats_list(cls, stats_list: List["ParsingStats"]) -> "ParsingStatsCounts":
        total_valid = 0
        total_invalid = 0
        total_charge_counts = defaultdict(int)

        for stats in stats_list:
            total_valid += stats.valid
            total_invalid += stats.invalid_count
            for charge, count in stats.precursor_charge_counts.items():
                total_charge_counts[charge] += count

        return cls(total_valid, total_invalid, total_charge_counts)

