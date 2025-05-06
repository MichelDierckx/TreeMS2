import time

from TreeMS2.config.config import Config
from TreeMS2.utils.utils import format_execution_time
from .logger_config import get_logger
from .states.context import Context
from .states.process_spectra_state import ProcessSpectraState

logger = get_logger(__name__)

LOOP_LIMIT = 999


class TreeMS2:
    def __init__(self, config: Config):
        self.context = Context(config=config)

    def run(self):
        start_time = time.time()  # record start time
        self.context.push_state(state=ProcessSpectraState(context=self.context))
        loop_nr = 0
        while self.context.states:
            if loop_nr > LOOP_LIMIT:
                logger.error("Program exited with pending tasks.")
                break
            self.context.next()
            loop_nr += 1
        execution_time = time.time() - start_time  # calculate execution time
        formatted_time = format_execution_time(execution_time)
        logger.info(f"TreeMS2 finished in {formatted_time}")
