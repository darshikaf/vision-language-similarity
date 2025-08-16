import logging
import sys
from typing import Any

from service.constants import APP_NAME


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging with consistent formatting.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for the given module.

    Returns:
        Configured logger instance
    """
    if name is None:
        name = APP_NAME
    return logging.getLogger(name)


# Create default application logger
logger = get_logger(APP_NAME)


def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    logger.error(msg, *args, **kwargs)


def exception(msg: str, *args: Any, **kwargs: Any) -> None:
    logger.exception(msg, *args, **kwargs)


def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    logger.critical(msg, *args, **kwargs)
