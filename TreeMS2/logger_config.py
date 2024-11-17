import logging

LOG_FILE = "logs/app.log"


def setup_logging():
    # Create a rotating file handler (logs to file with size limit and backups)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Set up a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Disable dependency non-critical log messages.
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    Each logger is prefixed with the module's name for clarity.
    """
    return logging.getLogger(module_name)
