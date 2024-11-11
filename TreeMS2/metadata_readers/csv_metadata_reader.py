import csv
import os
from typing import Union, IO, List

from .metadata_reader import MetaDataReader
from .peak_file_group_mapping import PeakFileGroupMapping


class CsvMetaDataReader(MetaDataReader):
    """
    Metadata reader class to handle csv files for reading metadata for peak files.
    """
    VALID_EXTENSIONS: List[str] = ['.csv', '.tsv']

    @classmethod
    def get_metadata(cls, source: Union[IO, str]) -> PeakFileGroupMapping:
        """
        Reads metadata for peak files from a csv file.
        :param source: Union[IO, str], the source (file name or file object) to read the metadata from.
        :return: PeakFileGroupMapping, a mapping that maps peak files to groups.
        :raises:
        ValueError: CSV file does not contain 'PeakFile' and/or 'Group' columns.
        FileNotFoundError: CSV file does not exist.
        """
        # Used to store mapping that associates sample files with group names/ids
        peak_file_group_mapping = PeakFileGroupMapping()

        # Open the CSV file if not passed as object
        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"The file '{source}' does not exist.")
            file = open(source, mode='r', newline='', encoding='utf-8')
            close_file = True
        else:
            file = source
            close_file = False

        try:
            # Check the file extension and set delimiter accordingly
            file_extension = os.path.splitext(file.name if isinstance(file, str) else source.name)[1].lower()
            if file_extension == '.csv':
                delimiter = ','  # Comma for CSV
            elif file_extension == '.tsv':
                delimiter = '\t'  # Tab for TSV
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}. Only '.csv' and '.tsv' are supported.")

            reader = csv.DictReader(file, delimiter=delimiter)

            # columns PeakFile and Group should be present
            if 'PeakFile' not in reader.fieldnames or 'Group' not in reader.fieldnames:
                raise ValueError("CSV file must contain 'PeakFile' and 'Group' columns.")

            # loop over entries
            for row in reader:
                peak_file = row['PeakFile']
                group_name = row['Group']
                peak_file_group_mapping.add(peak_file, group_name)  # add PeakFile file to mapping

        finally:
            # if we had to open a file, close it
            if close_file:
                file.close()

        return peak_file_group_mapping
