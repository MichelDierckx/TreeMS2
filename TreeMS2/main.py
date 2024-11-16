from typing import Union, List

from .config.config_factory import ConfigFactory
from .tree_ms2 import TreeMS2


def main(args: Union[str, List[str]] = None) -> int:
    config_factory = ConfigFactory()
    config_factory.parse(args)  # Parse arguments from config file or command-line
    app = TreeMS2(config_factory)
    app.run()
    return 0