"""
Module related to logging
"""

import logging
import os
from datetime import datetime

# ANSI escape codes for colors
LOG_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
    "SECTION": "\033[35m",  # Magenta for section titles
    "RESET": "\033[0m",  # Reset color
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, LOG_COLORS["RESET"])
        reset = LOG_COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def format_execution_time(seconds: float) -> str:
    """
    Pretty print execution time.
    :return:
    """
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    parts = []

    days, seconds = divmod(seconds, 86400)
    if days:
        parts.append(f"{days}d")

    hours, seconds = divmod(seconds, 3600)
    if hours:
        parts.append(f"{hours}h")

    minutes, seconds = divmod(seconds, 60)
    if minutes:
        parts.append(f"{minutes}m")

    if seconds:
        parts.append(f"{seconds}s")

    if millis:
        parts.append(f"{millis}ms")

    return " ".join(parts) if parts else "0ms"


def setup_logging(work_dir: str, console_level: str):
    # Create a "logs" subdirectory inside the working directory
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)  # Create it if it doesn't exist

    # Create a timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    # File handler (no colors here)
    file_handler = logging.FileHandler(filename=log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (with color)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    color_formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(color_formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Optional: reduce noise from dependencies
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    """
    return logging.getLogger(module_name)


def log_section_title(
    logger: logging.Logger, title: str, symbol: str = "=", width: int = 80
):
    """
    Logs a section title.
    """
    section_color = LOG_COLORS["SECTION"]
    reset_color = LOG_COLORS["RESET"]

    section_title = f"{section_color}{symbol * ((width - len(title) - 2) // 2)} {title} {symbol * ((width - len(title) - 2) // 2)}{reset_color}"

    logger.info(section_title)


def log_parameter(logger: logging.Logger, parameter_name, parameter_value):
    """
    Logs a parameter value.
    """
    log_content = f"  {parameter_name}: {parameter_value}"
    logger.info(log_content)
