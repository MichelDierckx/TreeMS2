import os
from typing import Union, List

import faiss

from TreeMS2.environment_variables import TREEMS2_NUM_CPUS, TREEMS2_MEM_PER_CPU
from .config.config import Config
from .logger_config import setup_logging, get_logger, log_parameter, log_section_title
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

    logger = get_logger(__name__)
    # log environment variables
    log_section_title(logger=logger, title="[ ENVIRONMENT VARIABLES (TreeMS2) ]")
    log_parameter(logger=logger, parameter_name=TREEMS2_NUM_CPUS,
                  parameter_value=os.environ.get(TREEMS2_NUM_CPUS, 'Not Set'))
    log_parameter(logger=logger, parameter_name=TREEMS2_MEM_PER_CPU,
                  parameter_value=os.environ.get(TREEMS2_MEM_PER_CPU, 'Not Set'))

    # get FAISS number of threads
    logger.debug(f"FAISS threads: {faiss.omp_get_max_threads()}")

    # run application
    app = TreeMS2(config)
    app.run()
    return 0
