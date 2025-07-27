import time

from TreeMS2.config.logger_config import get_logger, format_execution_time
from TreeMS2.config.treems2_config import Config
from TreeMS2.ingestion.ingestion_state import IngestionState
from TreeMS2.states.context import Context

logger = get_logger(__name__)

LOOP_LIMIT = 999


class TreeMS2:
    def __init__(self, config: Config):
        self.context = Context(config=config)

    def run(self):
        start_time = time.time()  # record start time
        self.context.push_state(state=IngestionState(context=self.context))
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
        return self.context.groups.get_stats()[1].high_quality
