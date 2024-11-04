"""
This module contains the `Config` class for parsing arguments from a config file or the command line.

Note:
This code was adapted from the `falcon` repository for argument parsing:
https://github.com/bittremieux/falcon

The original structure and approach were used as a reference, with modifications
to fit the specific needs of this project.
"""

from typing import Optional, Any

import configargparse


class Config:
    """
    Command-line and file-based configuration handler. Makes use of the singleton pattern.

    Configuration settings can be specified in a config.ini file (by default in
    the working directory), or as command-line arguments.
    """

    _instance = None  # To ensure only one instance (singleton)

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
        """
        if hasattr(self, "_initialized") and self._initialized:
            return  # Prevent re-initialization in singleton

        self._parser = configargparse.ArgParser(
            description="TreeMS2: An efficient tool for phylogenetic analysis of MS/MS spectra using ANN indexing.",
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

        # Add argument definitions
        self._define_arguments()
        self._namespace = None
        self._initialized = True  # Mark as initialized to avoid re-running init

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
                "Supported formats:\n"
                "  - .csv: Must contain 'sample_file' and 'group' columns.\n"
            ),
        )
        self._parser.add_argument(
            "--ms2_dir",
            required=True,
            help="Directory containing MS/MS files (supported formats: .mgf)",
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
        self._namespace = vars(self._parser.parse_args(args_str))

    def get(self, option: str, default: Any = None) -> Any:
        """
        Safely get a configuration option with a default value.

        Parameters
        ----------
        option : str
            The configuration option to retrieve.
        default : Any
            The default value to return if the option is not found.

        Returns
        -------
        Any
            The option value or the default if the option is not set.
        """
        if self._namespace is None:
            raise RuntimeError("Configuration has not been parsed. Call `parse()` first.")
        return self._namespace.get(option, default)

    def __getattr__(self, option: str) -> Any:
        """
        Allows accessing configuration options as attributes.
        """
        return self.get(option)

    def __getitem__(self, option: str) -> Any:
        """
        Allows accessing configuration options via dictionary-like syntax.
        """
        return self.get(option)


# Instantiate a shared configuration object for global access. Only one can exist.
config = Config()
