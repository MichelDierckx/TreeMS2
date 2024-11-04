from typing import Union, List

from .config import config


def main(args: Union[str, List[str]] = None) -> int:
    config.parse(args)  # Parse arguments from config file or command-line
    return 0
