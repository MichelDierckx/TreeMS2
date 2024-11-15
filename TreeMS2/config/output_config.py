class OutputConfig:
    def __init__(self, work_dir: str):
        """
        Configuration for sample-to-group file mapping.

        Parameters:
        ----------
        sample_to_group_file : str
            Path to the file containing a mapping from sample filename to group.
            Supported formats are .csv and .tsv. The file must contain columns
            named 'sample_file' and 'group'.
        """
        self.work_dir = work_dir

    @classmethod
    def from_parser(cls, parser) -> "OutputConfig":
        return cls(
            work_dir=parser.get("work_dir")
        )
