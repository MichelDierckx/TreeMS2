import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from TreeMS2.config.logger_config import get_logger
from TreeMS2.ingestion.parsing.parsing_stats import ParsingStatsCounts
from TreeMS2.ingestion.preprocessing.quality_stats import QualityStatsCounts
from TreeMS2.ingestion.spectra_sets.peak_file.peak_file import PeakFile
from TreeMS2.ingestion.spectra_sets.spectra_set import SpectraSet

logger = get_logger(__name__)


class SpectraSets:
    def __init__(self):
        self.spectra_sets_mapping_path = ""

        self._spectra_sets: List[SpectraSet] = []

    def add(self, spectra_set: SpectraSet) -> SpectraSet:
        for spec_set in self._spectra_sets:
            if spec_set.get_label() == spectra_set.get_label():
                return spec_set
        spectra_set.set_id(len(self._spectra_sets))
        self._spectra_sets.append(spectra_set)
        return self._spectra_sets[-1]

    def get_spectra_sets(self) -> List[SpectraSet]:
        return self._spectra_sets

    def count_spectra_sets(self) -> int:
        return len(self._spectra_sets)

    def count_peak_files(self) -> int:
        nr_files = 0
        for spectra_set in self._spectra_sets:
            nr_files += spectra_set.count_peak_files()
        return nr_files

    def get_spectra_set(self, spectra_set_id: int) -> SpectraSet:
        return self._spectra_sets[spectra_set_id]

    def get_spectra_set_ids(self) -> List[int]:
        return [spectra_set.get_id() for spectra_set in self._spectra_sets]

    def get_stats(self) -> Tuple[ParsingStatsCounts, QualityStatsCounts]:
        # Collect all ParsingStats from all SpectraSets
        all_parsing_stats = []
        # Collect all QualityStats from all SpectraSets
        all_quality_stats = []

        for spectra_set in self._spectra_sets:
            # Get lists of stats from the peak files within each spectra set
            parsing_stats_list = [
                pf.parsing_stats
                for pf in spectra_set.get_peak_files()
                if pf.parsing_stats is not None
            ]
            quality_stats_list = [
                pf.quality_stats
                for pf in spectra_set.get_peak_files()
                if pf.quality_stats is not None
            ]

            all_parsing_stats.extend(parsing_stats_list)
            all_quality_stats.extend(quality_stats_list)

        # Aggregate parsing stats across all peak files in all spectra sets
        aggregated_parsing_stats = ParsingStatsCounts.from_parsing_stats_list(
            all_parsing_stats
        )
        # Aggregate quality stats across all peak files in all spectra sets
        aggregated_quality_stats = QualityStatsCounts.from_quality_stats_list(
            all_quality_stats
        )

        return aggregated_parsing_stats, aggregated_quality_stats

    @classmethod
    def read(cls, file_path: str):
        path = Path(file_path)
        # Check if the file exists
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"The file at {path} does not exist or is not a valid file."
            )
        valid_extensions = [".csv", ".tsv"]
        file_extension = path.suffix.lower()
        match file_extension:
            case ".csv" | ".tsv":
                groups = cls._parse_spectra_sets_mapping(path)
            case _:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. Supported types: {', '.join(valid_extensions)}"
                )
        return groups

    @staticmethod
    def _parse_spectra_sets_mapping(path: Path) -> "SpectraSets":
        """
        Helper function to read the file and create Groups instances.
        The file must contain the columns 'file' and 'group'.
        It also checks that each 'file' exists and that the data rows are valid.
        """
        spectra_dataset = SpectraSets()

        path = path.resolve()
        spectra_dataset.spectra_sets_mapping_path = str(path)

        # Get the directory of the CSV/TSV file to resolve relative paths
        file_dir = path.parent

        with path.open(newline="", mode="r") as f:
            # Determine the delimiter based on file extension
            delimiter = "," if path.suffix.lower() == ".csv" else "\t"

            reader = csv.DictReader(f, delimiter=delimiter)

            # Check if the necessary headers are present
            if "file" not in reader.fieldnames or "group" not in reader.fieldnames:
                raise ValueError(f"The file must contain 'file' and 'group' headers.")

            data_rows_found = False  # Track if we find any valid data rows

            # Process each row and create Group objects
            for row_number, row in enumerate(
                reader, start=2
            ):  # start=2 because row 1 is header
                # Ensure both 'file' and 'group' are present and not empty
                if not row.get("file") or not row.get("group"):
                    raise ValueError(
                        f"Row {row_number}: Each row must contain both a valid 'file' and 'group' entry."
                    )

                file_value = row["file"]
                group_value = row["group"]

                absolute_file_path = (file_dir / file_value).resolve()

                # Check if the file path exists (the file should exist on the filesystem)
                if not absolute_file_path.is_file():
                    raise FileNotFoundError(
                        f"Row {row_number}: The file specified in the 'file' column, {absolute_file_path}, does not exist."
                    )

                peak_file = PeakFile(file_path=str(absolute_file_path))
                spectra_set = SpectraSet(group_value)
                spectra_dataset.add(spectra_set).add(peak_file)
                data_rows_found = True  # At least one valid data row was found

            # If no valid data rows were found
            if not data_rows_found:
                raise ValueError("The file does not contain any valid data rows.")

        return spectra_dataset

    def to_dict(self) -> Dict[str, Any]:
        parsing_stats_counts, quality_stats_counts = self.get_stats()
        return {
            "spectra_sets_mapping_path": self.spectra_sets_mapping_path,
            "valid": {
                "count": parsing_stats_counts.valid,
                "high_quality": quality_stats_counts.high_quality,
                "low_quality": quality_stats_counts.low_quality,
            },
            "invalid": {
                "count": parsing_stats_counts.invalid,
            },
            "spectra_sets": [
                spectra_set.to_dict() for spectra_set in self._spectra_sets
            ],
        }

    def write_to_file(self, path: str):
        """
        Write the SpectraDataset to a JSON-file.
        :param path: The path to which the JSON file will be written.
        :return:
        """
        with open(path, "w") as json_file:
            json.dump(self.to_dict(), json_file, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["SpectraSets"]:
        try:
            spectra_dataset = cls()
            spectra_dataset.spectra_sets_mapping_path = data[
                "spectra_sets_mapping_path"
            ]
            spectra_dataset._spectra_sets = [
                SpectraSet.from_dict(spectra_set)
                for spectra_set in data["spectra_sets"]
            ]
            return spectra_dataset
        except (KeyError, TypeError, AttributeError):
            return None  # Return None if the data structure is incorrect

    @classmethod
    def load(cls, path: str) -> Optional["SpectraSets"]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
            return cls.from_dict(data)
        except (json.JSONDecodeError, OSError, PermissionError):
            return None  # Return None if the file is unreadable or corrupted
