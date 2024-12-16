from TreeMS2.config.config import Config
from .logger_config import get_logger
from .states.context import Context
from .states.process_spectra_state import ProcessSpectraState

logger = get_logger(__name__)

LOOP_LIMIT = 1000


class TreeMS2:
    def __init__(self, config: Config):
        self.context = Context(config=config)

    def run(self):
        self.context.push_state(state=ProcessSpectraState(context=self.context))
        loop_nr = 0
        while self.context.states:
            if loop_nr > LOOP_LIMIT:
                logger.error("Program exited with pending tasks.")
                break
            self.context.get_state().run(overwrite=self.context.config.overwrite)
            loop_nr += 1
