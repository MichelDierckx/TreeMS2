# TreeMS2/config/config_factory.py
import logging
import os
from typing import Optional, Any

import configargparse

from .groups_config import GroupsConfig
from .spectrum_processing_config import SpectrumProcessingConfig


class ConfigFactory:
    def __init__(self):
        self._parser = configargparse.ArgParser(
            description="TreeMS2: An efficient tool for phylogenetic analysis of MS/MS spectra using ANN indexing.",
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
        self.logger = logging.getLogger("config")  # Naming the logger as "config"
        self._setup_logger()
        self._define_arguments()
        self._namespace = None

    def create_peak_file_config(self) -> GroupsConfig:
        return GroupsConfig.from_parser(self.parser)

    def create_spectrum_processing_config(self) -> SpectrumProcessingConfig:
        return SpectrumProcessingConfig.from_parser(self.parser)

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
        self._validate_path("ms2_dir")
        self._validate_path("sample_to_group")
        self._validate_path("work_dir")
        self._validate_positive_int("low_dim", True)
        self._log_parameters()

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
        self.logger.propagate = False

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
            "--ms2_dir",
            required=True,
            help=(
                f"Directory containing MS/MS files (supported formats: .mgf)."
            ),
            type=str,
            action='store',
            dest="ms2_dir",
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
            default="off",
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

    def _validate_choice(self, param: str, valid_options: list) -> None:
        """
        Validate that the value of a configuration parameter is in the list of valid options.
        """
        value = self.get(param)
        if value not in valid_options:
            raise ValueError(f"--{param}: Invalid value '{value}'. Valid options are: {', '.join(valid_options)}")

    def _validate_path(self, param: str, required_extensions: Optional[list] = None) -> None:
        """
        Validate that a file path exists and optionally check if it has one of the required extensions.
        """
        path = self.get(param)
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{param}: Path '{path}' does not exist.")
        if required_extensions:
            if not any(path.endswith(ext) for ext in required_extensions):
                valid_extensions = ', '.join(required_extensions)
                raise ValueError(
                    f"--{param}: Path '{path}' does not end with one of the following extensions: {valid_extensions}")

    def _validate_positive_int(self, param: str, strict: bool = False) -> None:
        value = self.get(param)
        int_value = int(value)
        if (strict and int_value <= 0) or (not strict and int_value < 0):
            comparison = "greater than 0" if strict else "0 or greater"
            raise ValueError(f"--{param}: {value} is not a positive integer ({comparison}).")

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
