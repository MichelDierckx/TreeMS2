import csv
import os
from typing import Union, IO, List

from .metadata_reader import MetaDataReader
from .sample_group_mapping import SampleGroupMapping


class CsvMetaDataReader(MetaDataReader):
    """
    Metadata reader class to handle csv files for reading metadata for sample files.
    """
    VALID_EXTENSIONS: List[str] = ['.csv']

    @classmethod
    def get_metadata(cls, source: Union[IO, str]) -> SampleGroupMapping:
        """
        Reads metadata for samples from a csv file.
        :param source: Union[IO, str], the source (file name or file object) to read the metadata from.
        :return: SampleGroupMapping, a mapping that maps samples to groups.
        :raises:
        ValueError: CSV file does not contain 'sample_file' and/or 'group' columns.
        FileNotFoundError: CSV file does not exist.
        """
        # Used to store mapping that associates sample files with group names/ids
        sample_group_mapping = SampleGroupMapping()

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
            reader = csv.DictReader(file)

            # columns sample_file and group should be present
            if 'sample_file' not in reader.fieldnames or 'group' not in reader.fieldnames:
                raise ValueError("CSV file must contain 'sample_file' and 'group' columns.")

            # loop over entries
            for row in reader:
                sample_file = row['sample_file']
                group_name = row['group']
                sample_group_mapping.add(sample_file, group_name)  # add sample file to mapping

        finally:
            # if we had to open a file, close it
            if close_file:
                file.close()

        return sample_group_mapping
