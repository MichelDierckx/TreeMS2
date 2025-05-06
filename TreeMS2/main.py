from typing import Union, List

import faiss

from .config.config import Config
from .logger_config import setup_logging, get_logger
from .tree_ms2 import TreeMS2


def main(args: Union[str, List[str]] = None) -> int:
    # setup up config
    config = Config()
    config.parse(args)  # Parse arguments from config file or command-line

    # setup logging
    console_level = config.log_level
    setup_logging(work_dir=config.work_dir, console_level=console_level)

    # log config parameters
    config.log_parameters()

    # set FAISS number of threads
    faiss.omp_set_num_threads(8)
    logger = get_logger(__name__)
    logger.debug("FAISS threads:", faiss.omp_get_max_threads())

    # run application
    app = TreeMS2(config)
    app.run()
    return 0
