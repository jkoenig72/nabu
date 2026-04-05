"""Unified logging setup for Nabu. Call setup_logging() once at startup."""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "nabu.log")
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Third-party loggers that are too noisy at DEBUG
_QUIET_LOGGERS = ["httpcore", "httpx", "asyncio", "piper.voice"]


def setup_logging():
    """Configure root logger with console + daily rotating file output."""
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", backupCount=7, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
