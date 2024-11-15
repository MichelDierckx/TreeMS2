from typing import Union, List

from .config.config_factory import ConfigFactory


def main(args: Union[str, List[str]] = None) -> int:
    config_factory = ConfigFactory()
    config_factory.parse(args)  # Parse arguments from config file or command-line

    return 0