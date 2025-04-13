from typing import Union, List

from .config.config import Config
from .logger_config import setup_logging
from .tree_ms2 import TreeMS2


def main(args: Union[str, List[str]] = None) -> int:
    config = Config()
    config.parse(args)  # Parse arguments from config file or command-line

    console_level = config.log_level
    setup_logging(console_level)

    config.log_parameters()

    app = TreeMS2(config)
    app.run()
    return 0
