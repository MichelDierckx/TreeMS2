import os
from typing import Optional, Any, Union

import configargparse

from TreeMS2.logger_config import get_logger, log_section_title, log_parameter

# Create a logger for this module
logger = get_logger(__name__)


class Config:
    def __init__(self):
        self._parser = configargparse.ArgParser(
            description="TreeMS2: An efficient tool for phylogenetic analysis of MS/MS spectra using ANN indexing.",
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
        self._define_arguments()
        self._namespace = None

    def get(self, option: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieve configuration options with a default value if the option does not exist.
        :param option: str, the configuration option to retrieve.
        :param default: Optional[str], the default value to return if the option does not exist. Defaults to None.
        :return: Optional[str], The value of the configuration option or the default value.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized. Call `parse()` first.")
        return self._namespace.get(option, default)

    def parse(self, args_str: Optional[str] = None) -> None:
        """
        Parse the configuration settings.

        Parameters
        ----------
        args_str : Optional[str]
            If None, arguments are taken from sys.argv; otherwise, a string of arguments.
            Arguments not specified on the command line are taken from the config file.
        """
        if self._namespace is not None:
            return  # Skip parsing if already parsed

        # Parse the arguments
        self._namespace = vars(self._parser.parse_args(args_str))
        self._validate_file_path("sample_to_group_file")
        self._validate_directory_path("work_dir")

        self._validate_positive_number("fragment_tol", True)
        self._validate_positive_number("min_peaks")
        self._validate_positive_number("min_mz_range")
        self._validate_positive_number("min_mz")
        self._validate_positive_number("max_mz")
        self._validate_positive_number("remove_precursor_tol")
        self._validate_number_range("min_intensity", min_value=0.0, max_value=1.0)
        self._validate_positive_number("low_dim", True)
        self._validate_positive_number("batch_size", True)
        self._validate_positive_number("num_neighbours", True)
        self._validate_positive_number("num_probe", True)
        self._validate_number_range("similarity", min_value=0.0, max_value=1.0)
        self._validate_positive_number("precursor_mz_window", True)

        self._validate_num_neighbours_num_probe()

    def _define_arguments(self) -> None:
        """
        Define the command-line and config file arguments.
        """
        # IO arguments
        self._parser.add_argument(
            "--sample_to_group_file",
            required=True,
            help=(
                "File containing a mapping from sample filename to group.\n"
                "Supported formats("
                ".csv, .tsv):\n"
                "  - .csv: Must contain 'sample_file' and 'group' columns.\n"
            ),
            type=str,
            action='store',
            dest="sample_to_group_file",
            metavar="<path>",
        )
        self._parser.add_argument(
            "--work_dir",
            required=True,
            help="Working directory (used to save lance datasets)",
            type=str,
            action='store',
            dest="work_dir",
            metavar="<path>",
        )
        self._parser.add_argument(
            "--overwrite",
            help="Overwrite existing results if they already exist.",
            action="store_true",
            dest="overwrite",
        )

        # Whether to enable temporary compaction during vector store writes
        self._parser.add_argument(
            "--incremental_compaction",
            help="Whether to enable incremental compaction during writes to Lance vector stores. When enabled, compaction is performed periodically during ingestion rather than only at the end.",
            action="store_true",
            dest="incremental_compaction",
        )

        # Processing
        self._parser.add_argument(
            "--fragment_tol",
            type=float,
            default=0.05,
            help="Fragment mass tolerance in m/z (default: %(default)s m/z).",
            dest="fragment_tol",
        )
        self._parser.add_argument(
            "--min_peaks",
            default=5,
            type=int,
            help="Discard spectra with fewer than this number of peaks "
                 "(default: %(default)s).",
            dest="min_peaks",
        )
        self._parser.add_argument(
            "--min_mz_range",
            default=250.0,
            type=float,
            help="Discard spectra with a smaller mass range "
                 "(default: %(default)s m/z).",
            dest="min_mz_range",
        )
        self._parser.add_argument(
            "--min_mz",
            default=101.0,
            type=float,
            help="Minimum peak m/z value (inclusive, "
                 "default: %(default)s m/z).",
            dest="min_mz",
        )
        self._parser.add_argument(
            "--max_mz",
            default=1500.0,
            type=float,
            help="Maximum peak m/z value (inclusive, "
                 "default: %(default)s m/z).",
            dest="max_mz",
        )
        self._parser.add_argument(
            "--remove_precursor_tol",
            default=1.5,
            type=float,
            help="Window around the precursor mass to remove peaks "
                 "(default: %(default)s m/z).",
            dest="remove_precursor_tol",
        )
        self._parser.add_argument(
            "--min_intensity",
            default=0.01,
            type=float,
            help="Remove peaks with a lower intensity relative to the base "
                 "intensity (default: %(default)s).",
            dest="min_intensity",
        )
        self._parser.add_argument(
            "--max_peaks_used",
            default=50,
            type=int,
            help="Only use the specified most intense peaks in the spectra "
                 "(default: %(default)s).",
            dest="max_peaks_used",
        )
        self._parser.add_argument(
            "--scaling",
            default="root",
            type=str,
            choices=["off", "root", "log", "rank"],
            help="Peak scaling method used to reduce the influence of very "
                 "intense peaks (default: %(default)s).",
            dest="scaling",
        )

        # Vectorization
        self._parser.add_argument(
            "--low_dim",
            default=400,
            type=int,
            help="Low-dimensional vector length (default: %(default)s).",
            dest="low_dim",
        )

        # Whether to use GPU for training
        self._parser.add_argument(
            "--use_gpu",
            help="Whether to use GPU for training (default: False).",
            action="store_true",
            dest="use_gpu",
        )

        self._parser.add_argument(
            "--batch_size",
            default=16384, type=int,
            help="Number of query spectra to process simultaneously "
                 "(default: %(default)s)",
            dest="batch_size", )

        self._parser.add_argument(
            "--num_neighbours",
            default=1024,
            type=int,
            help="The number of neighbours to retrieve for each query during ANN search. "
                 "(default: %(default)s). maximum 2048 when using GPU",
            dest="num_neighbours",
        )

        self._parser.add_argument(
            "--num_probe",
            default=128,
            type=int,
            help="Number of clusters to probe during ANN search. "
                 "(default: %(default)s). maximum 2048 when using GPU",
            dest="num_probe",
        )

        # Similarity threshold
        self._parser.add_argument(
            "--similarity",
            default=0.8,
            type=float,
            help="Minimum cosine similarity score for 2 spectra to be considered similar (default: %(default)s).",
            dest="similarity",
        )

        # Precursor MZ filtering on search results
        self._parser.add_argument(
            "--precursor_mz_window",
            default=2.05,
            type=float,
            help="Maximum difference in precursor m/z for two spectra to be considered similar (default: %(default)s).",
            dest="precursor_mz_window",
        )

        # Logging
        self._parser.add_argument(
            "--log_level",
            default="INFO",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help=(
                "Set the logging level for console output "
                "(default: %(default)s). Log file will always capture all levels."
            ),
            dest="log_level",
        )

    def _validate_directory_path(self, param: str, required_extensions: Optional[list] = None) -> None:
        """
        Validate that a directory path exists and optionally check if it has one of the required extensions.

        Parameters:
        ----------
        param : str
            The name of the parameter to validate.

        required_extensions : Optional[list], optional
            A list of file extensions to validate against. If provided, checks that at least one file
            in the directory matches one of the extensions.
        """
        path = self.get(param)
        if not os.path.isdir(path):  # Check if path is a valid directory
            raise NotADirectoryError(f"--{param}: Path '{path}' is not a valid directory.")

        if required_extensions:
            # Check if any file in the directory matches the required extensions
            files_with_valid_extension = [
                file for file in os.listdir(path) if any(file.endswith(ext) for ext in required_extensions)
            ]
            if not files_with_valid_extension:
                valid_extensions = ', '.join(required_extensions)
                raise ValueError(
                    f"--{param}: No files with extensions {valid_extensions} found in the directory '{path}'."
                )

    def _validate_file_path(self, param: str, required_extensions: Optional[list] = None) -> None:
        """
        Validate that a file path exists and optionally check if it has one of the required extensions.

        Parameters:
        ----------
        param : str
            The name of the parameter to validate.

        required_extensions : Optional[list], optional
            A list of valid extensions to check the file against. If provided, ensures that the file ends with one of them.
        """
        path = self.get(param)
        if not os.path.isfile(path):  # Check if the path is a valid file
            raise FileNotFoundError(f"--{param}: Path '{path}' is not a valid file.")

        if required_extensions:
            if not any(path.endswith(ext) for ext in required_extensions):
                valid_extensions = ', '.join(required_extensions)
                raise ValueError(
                    f"--{param}: Path '{path}' does not end with one of the following extensions: {valid_extensions}"
                )

    def _validate_positive_number(self, param: str, strict: bool = False) -> None:
        value = self.get(param)
        float_value = float(value)
        if (strict and float_value <= 0) or (not strict and float_value < 0):
            comparison = "greater than 0" if strict else "0 or greater"
            raise ValueError(f"--{param}: {value} is not a positive integer ({comparison}).")

    def _validate_number_range(self, param: str, min_value: Union[int, float], max_value: Union[int, float]) -> None:
        """
        Validate that the value of a configuration parameter is in the range (inclusive) specified by the user.
        :param param: The name of the parameter to be validated
        :param min_value: The minimum value of the range
        :param max_value: The maximum value of the range
        :return: None, raises ValueError if the value is not within the specified range
        """
        value = self.get(param)
        if value < min_value:
            raise ValueError(f"--{param}: value can not be less than {min_value:.4f}. Got {value:.4f}.")
        if value > max_value:
            raise ValueError(f"--{param}: value can not be greater than {max_value:.4f}. Got {value:.4f}.")

    def _validate_num_neighbours_num_probe(self) -> None:
        """
        Checks whether num_neighbours and num_probes exceed 2048 when using GPU.
        """
        num_neighbours: int = self.get("num_neighbours")
        num_probe: int = self.get("num_probe")
        use_gpu: bool = self.get("use_gpu")
        if use_gpu:
            if num_neighbours > 2048:
                raise ValueError(
                    f"--num_neighbours: value can not be larger than 2048 when using GPU (--use_gpu = true).")
            if num_probe > 2048:
                raise ValueError(
                    f"--num_probe: value can not be larger than 2048 when using GPU (see --use_gpu = true).")

    def log_parameters(self) -> None:
        """Log all chosen parameters."""
        log_section_title(logger=logger, title="[ CONFIGURATION ]")
        for key, value in self._namespace.items():
            log_parameter(logger=logger, parameter_name=key, parameter_value=value)

    def __getattr__(self, option):
        """
        Retrieve configuration options as attributes.

        Raises a KeyError with a helpful message if the option does not exist.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized. Call `parse()` first.")
        if option not in self._namespace:
            raise KeyError(f"The configuration option '{option}' does not exist.")
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)
