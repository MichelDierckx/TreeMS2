"""
This module contains the `Config` class for parsing arguments from a config file or the command line.

Note:
This code was adapted from the `falcon` repository for argument parsing:
https://github.com/bittremieux/falcon

The original structure and approach were used as a reference, with modifications
to fit the specific needs of this project.
"""

import logging
import os
from typing import Optional, Any

import configargparse

from .metadata_readers.metadata_reader_manager import MetadataReaderManager
from .peak_file_readers.peak_file_reader_manager import PeakFileReaderManager


class Config:
    """
    Command-line and file-based configuration handler. Ensures only one instance (singleton pattern).
    """

    _instance = None  # To ensure only one instance (singleton)

    # declare the names for the different parameters once
    SAMPLE_TO_GROUP_FILE = "sample_to_group_file"
    MS2_DIR = "ms2_dir"
    FRAGMENT_TOL = "fragment_tol"
    MIN_PEAKS = "min_peaks"
    MIN_MZ_RANGE = "min_mz_range"
    MIN_MZ = "min_mz"
    MAX_MZ = "max_mz"
    REMOVE_PRECURSOR_TOL = "remove_precursor_tol"
    MIN_INTENSITY = "min_intensity"
    MAX_PEAKS_USED = "max_peaks_used"
    SCALING = "scaling"

    def __new__(cls) -> "Config":
        """
        Ensure only one instance of Config exists (singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the configuration parser and provide default values.
        Ensures the configuration is only initialized once.
        """
        if self.__dict__.get("_initialized", False):
            return  # An instance already exists, so return

        self._parser = configargparse.ArgParser(
            description="TreeMS2: An efficient tool for phylogenetic analysis of MS/MS spectra using ANN indexing.",
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

        self.logger = logging.getLogger("config")  # Naming the logger as "config"
        self._setup_logger()
        self._define_arguments()  # Add argument definitions
        self._namespace = None
        self._initialized = True  # Mark the configuration as initialized

    def _setup_logger(self):
        """
        Set up the logger specifically for the configuration module.
        """
        # Set the logger level (it can be adjusted based on your needs)
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler (optional, can log to file too)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter and apply it to the handler
        formatter = logging.Formatter(
            "{asctime} {levelname} [{name}] {message}",
            style="{"
        )
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(console_handler)

    def _define_arguments(self) -> None:
        """
        Define the command-line and config file arguments.
        """
        # IO arguments
        self._parser.add_argument(
            f"--{self.SAMPLE_TO_GROUP_FILE}",
            required=True,
            help=(
                "File containing a mapping from sample filename to group.\n"
                "Supported formats("
                f"{', '.join(MetadataReaderManager.get_all_valid_extensions())}):\n"
                "  - .csv: Must contain 'sample_file' and 'group' columns.\n"
            ),
            type=str,
            action='store',  # This action means the value will be stored in the `self._namespace`
            dest=f"{self.SAMPLE_TO_GROUP_FILE}",
            # The key under which the value will be stored in the parsed arguments
            metavar="<path>",  # The placeholder text shown in the help for this argument
        )
        self._parser.add_argument(
            f"--{self.MS2_DIR}",
            required=True,
            help=(
                f"Directory containing MS/MS files (supported formats: {', '.join(PeakFileReaderManager.get_all_valid_extensions())})."
            ),
            type=str,
            action='store',
            dest=f"{self.MS2_DIR}",
            metavar="<path>",
        )

        self._parser.add_argument(
            f"--{self.FRAGMENT_TOL}",
            type=float,
            default=0.05,
            help="Fragment mass tolerance in m/z (default: %(default)s m/z).",
            dest=self.FRAGMENT_TOL,
        )
        self._parser.add_argument(
            f"--{self.MIN_PEAKS}",
            default=5,
            type=int,
            help="Discard spectra with fewer than this number of peaks "
                 "(default: %(default)s).",
            dest=self.MIN_PEAKS,
        )
        self._parser.add_argument(
            f"--{self.MIN_MZ_RANGE}",
            default=250.0,
            type=float,
            help="Discard spectra with a smaller mass range "
                 "(default: %(default)s m/z).",
            dest=self.MIN_MZ_RANGE,
        )
        self._parser.add_argument(
            f"--{self.MIN_MZ}",
            default=101.0,
            type=float,
            help="Minimum peak m/z value (inclusive, "
                 "default: %(default)s m/z).",
            dest=self.MIN_MZ,
        )
        self._parser.add_argument(
            f"--{self.MAX_MZ}",
            default=1500.0,
            type=float,
            help="Maximum peak m/z value (inclusive, "
                 "default: %(default)s m/z).",
            dest=self.MAX_MZ,
        )
        self._parser.add_argument(
            f"--{self.REMOVE_PRECURSOR_TOL}",
            default=1.5,
            type=float,
            help="Window around the precursor mass to remove peaks "
                 "(default: %(default)s m/z).",
            dest=self.REMOVE_PRECURSOR_TOL,
        )
        self._parser.add_argument(
            f"--{self.MIN_INTENSITY}",
            default=0.01,
            type=float,
            help="Remove peaks with a lower intensity relative to the base "
                 "intensity (default: %(default)s).",
            dest=self.MIN_INTENSITY,
        )
        self._parser.add_argument(
            f"--{self.MAX_PEAKS_USED}",
            default=50,
            type=int,
            help="Only use the specified most intense peaks in the spectra "
                 "(default: %(default)s).",
            dest=self.MAX_PEAKS_USED,
        )
        self._parser.add_argument(
            f"--{self.SCALING}",
            default="off",
            type=str,
            choices=["off", "root", "log", "rank"],
            help="Peak scaling method used to reduce the influence of very "
                 "intense peaks (default: %(default)s).",
            dest=self.SCALING,
        )

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
        self._validate_path(self._namespace.get(f"{self.MS2_DIR}"))
        self._validate_path(self._namespace.get(f"{self.SAMPLE_TO_GROUP_FILE}"))
        self._log_parameters()

    def _validate_choice(self, param: str, valid_options: list) -> None:
        """
        Validate that the value of a configuration parameter is in the list of valid options.
        """
        value = self.get(param)
        if value not in valid_options:
            raise ValueError(f"Invalid value '{value}' for '{param}'. Valid options are: {', '.join(valid_options)}")

    @staticmethod
    def _validate_path(path: str, required_extensions: Optional[list] = None) -> None:
        """
        Validate that a file path exists and optionally check if it has one of the required extensions.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")
        if required_extensions:
            if not any(path.endswith(ext) for ext in required_extensions):
                valid_extensions = ', '.join(required_extensions)
                raise ValueError(f"Path '{path}' must end with one of the following extensions: {valid_extensions}")

    def _log_parameters(self) -> None:
        """Log all chosen parameters."""
        for key, value in self._namespace.items():
            self.logger.debug(f"  {key}: {value}")

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


# Instantiate a shared configuration object for global access. Only one can exist.
config = Config()
