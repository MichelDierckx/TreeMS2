import csv
from pathlib import Path
from typing import List

from .group import Group
from ..logger_config import get_logger
from ..peak_file.peak_file_factory import PeakFileFactory

logger = get_logger(__name__)


class Groups:
    def __init__(self):
        self._groups: List[Group] = []

    def add(self, group: Group) -> Group:
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

    @classmethod
    def from_file(cls, file_path: str):
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
        logger.info(f"Found {groups.get_size()} groups and {groups.get_nr_files()} files in '{file_path}'")
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

    def __repr__(self) -> str:
        groups_repr = "\n\t".join([repr(group) for group in self._groups])
        return f"{self.__class__.__name__}({self.get_size()} groups, {self.get_nr_files()} files):\n\t{groups_repr}"
