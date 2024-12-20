class GroupsConfig:
    def __init__(self, sample_to_group_file: str):
        """
        Configuration for sample-to-group file mapping.

        Parameters:
        ----------
        sample_to_group_file : str
            Path to the file containing a mapping from sample filename to group.
            Supported formats are .csv and .tsv. The file must contain columns
            named 'sample_file' and 'group'.
        """
        self.sample_to_group_file = sample_to_group_file
