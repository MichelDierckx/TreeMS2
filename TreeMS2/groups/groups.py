import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from TreeMS2.groups.group import Group
from TreeMS2.groups.peak_file.peak_file_factory import PeakFileFactory
from TreeMS2.logger_config import get_logger

logger = get_logger(__name__)


class Groups:
    def __init__(self):
        self.filename = ""

        self._groups: List[Group] = []

        self.total_spectra = 0
        self.failed_parsed = 0
        self.failed_processed = 0

    def add(self, group: Group) -> Group:
        for g in self._groups:
            if g.get_group_name() == group.get_group_name():
                return g
        group.set_id(len(self._groups))
        self._groups.append(group)
        return self._groups[-1]

    def get_groups(self) -> List[Group]:
        return self._groups

    def get_size(self) -> int:
        return len(self._groups)

    def get_nr_files(self) -> int:
        nr_files = 0
        for group in self._groups:
            nr_files += group.get_size()
        return nr_files

    def get_group(self, group_id: int) -> Group:
        return self._groups[group_id]

    def get_group_ids(self) -> List[int]:
        return [group.get_id() for group in self._groups]

    def total_valid_spectra(self) -> int:
        return self.total_spectra - self.failed_parsed - self.failed_processed

    @classmethod
    def read(cls, file_path: str):
        path = Path(file_path)
        # Check if the file exists
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"The file at {path} does not exist or is not a valid file.")
        valid_extensions = [".csv", ".tsv"]
        file_extension = path.suffix.lower()
        match file_extension:
            case ".csv" | ".tsv":
                groups = cls._read_groups_from_file(path)
            case _:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. Supported types: {', '.join(valid_extensions)}")
        return groups

    @staticmethod
    def _read_groups_from_file(path: Path) -> "Groups":
        """
        Helper function to read the file and create Groups instances.
        The file must contain the columns 'file' and 'group'.
        It also checks that each 'file' exists and that the data rows are valid.
        """
        groups = Groups()
        file_factory = PeakFileFactory()

        # Get the directory of the CSV/TSV file to resolve relative paths
        file_dir = path.parent
        file_name = path.name
        groups.filename = file_name

        with path.open(newline='', mode='r') as f:
            # Determine the delimiter based on file extension
            delimiter = ',' if path.suffix.lower() == '.csv' else '\t'

            reader = csv.DictReader(f, delimiter=delimiter)

            # Check if the necessary headers are present
            if "file" not in reader.fieldnames or "group" not in reader.fieldnames:
                raise ValueError(f"The file must contain 'file' and 'group' headers.")

            data_rows_found = False  # Track if we find any valid data rows

            # Process each row and create Group objects
            for row_number, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                # Ensure both 'file' and 'group' are present and not empty
                if not row.get("file") or not row.get("group"):
                    raise ValueError(f"Row {row_number}: Each row must contain both a valid 'file' and 'group' entry.")

                file_value = row["file"]
                group_value = row["group"]

                absolute_file_path = file_dir / file_value

                # Check if the file path exists (the file should exist on the filesystem)
                if not absolute_file_path.is_file():
                    raise FileNotFoundError(
                        f"Row {row_number}: The file specified in the 'file' column, {absolute_file_path}, does not exist.")

                peak_file = file_factory.create(absolute_file_path)
                group = Group(group_value)
                groups.add(group).add(peak_file)
                data_rows_found = True  # At least one valid data row was found

            # If no valid data rows were found
            if not data_rows_found:
                raise ValueError("The file does not contain any valid data rows.")

        return groups

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "total_spectra": self.total_spectra,
            "failed_parsed": self.failed_parsed,
            "failed_processed": self.failed_processed,
            "groups": [group.to_dict() for group in self._groups],
        }

    def write_to_file(self, path: str):
        """
        Write the groups to a JSON-file.
        :param path: The path to which the JSON file will be written.
        :return:
        """
        with open(path, "w") as json_file:
            json.dump(self.to_dict(), json_file, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["Groups"]:
        try:
            groups = cls()
            groups.filename = data["filename"]
            groups.total_spectra = data["total_spectra"]
            groups.failed_parsed = data["failed_parsed"]
            groups.failed_processed = data["failed_processed"]

            groups._groups = [Group.from_dict(group) for group in data["groups"]]
            return groups
        except (KeyError, TypeError, AttributeError):
            return None  # Return None if the data structure is incorrect

    @classmethod
    def load(cls, path: str) -> Optional["Groups"]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
            return cls.from_dict(data)
        except (json.JSONDecodeError, OSError, PermissionError):
            return None  # Return None if the file is unreadable or corrupted
