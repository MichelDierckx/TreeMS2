"""
This module contains the `Config` class for parsing arguments from a config file or the command line.

Note:
This code was adapted from the `falcon` repository for argument parsing:
https://github.com/bittremieux/falcon

The original structure and approach were used as a reference, with modifications
to fit the specific needs of this project.
"""

import os
from typing import Optional, Any

import configargparse

from metadata_readers.metadata_reader_manager import MetadataReaderManager
from spectrum_readers.spectrum_reader_manager import SpectrumReaderManager


class Config:
    """
    Command-line and file-based configuration handler. Ensures only one instance (singleton pattern).
    """

    _instance = None  # To ensure only one instance (singleton)

    # declare the names for the different parameters once
    _SAMPLE_TO_GROUP_FILE = "sample_to_group_file"
    _MS2_DIR = "ms2_dir"

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
        if hasattr(self, "_initialized") and self._initialized:
            return  # Prevent re-initializing the singleton instance

        self._parser = configargparse.ArgParser(
            description="TreeMS2: An efficient tool for phylogenetic analysis of MS/MS spectra using ANN indexing.",
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

        self._define_arguments()  # Add argument definitions
        self._namespace = None
        self._initialized = True  # Mark the configuration as initialized

    def _define_arguments(self) -> None:
        """
        Define the command-line and config file arguments.
        """
        # IO arguments
        self._parser.add_argument(
            f"--{self._SAMPLE_TO_GROUP_FILE}",
            required=True,
            help=(
                "File containing a mapping from sample filename to group.\n"
                "Supported formats("
                f"{', '.join(MetadataReaderManager.get_all_valid_extensions())}):\n"
                "  - .csv: Must contain 'sample_file' and 'group' columns.\n"
            ),
            type=str,
            action='store',  # This action means the value will be stored in the `self._namespace`
            dest=f"{self._SAMPLE_TO_GROUP_FILE}",
            # The key under which the value will be stored in the parsed arguments
            metavar="<path>",  # The placeholder text shown in the help for this argument
        )
        self._parser.add_argument(
            f"--{self._MS2_DIR}",
            required=True,
            help=(
                f"Directory containing MS/MS files (supported formats: {', '.join(SpectrumReaderManager.get_all_valid_extensions())})."
            ),
            type=str,
            action='store',  # This action means the value will be stored in the `self._namespace`
            dest=f"{self._MS2_DIR}",  # The key under which the value will be stored in the parsed arguments
            metavar="<path>",  # The placeholder text shown in the help for this argument
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
        self._validate_path(self._namespace.get(f"{self._MS2_DIR}"))
        self._validate_path(self._namespace.get(f"{self._SAMPLE_TO_GROUP_FILE}"))

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

    def __getattr__(self, option: str) -> Any:
        """
        Retrieve configuration options as attributes.

        Raises a KeyError with a helpful message if the option does not exist.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized. Call `parse()` first.")
        if option not in self._namespace:
            raise KeyError(f"The configuration option '{option}' does not exist.")
        return self._namespace[option]

    def __getitem__(self, item: str) -> Any:
        """
        Allow dictionary-like access to configuration options.
        """
        return self.__getattr__(item)

    def get(self, option: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieve configuration options with a default value if the option does not exist.

        Parameters
        ----------
        option : str
            The configuration option to retrieve.
        default : Optional[Any]
            The default value to return if the option does not exist.

        Returns
        -------
        Optional[Any]
            The value of the configuration option or the default value.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized. Call `parse()` first.")
        return self._namespace.get(option, default)


# Instantiate a shared configuration object for global access. Only one can exist.
config = Config()
