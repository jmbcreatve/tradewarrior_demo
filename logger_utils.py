import logging
import os
from logging.handlers import RotatingFileHandler

from config import Config

_LOGGER_CACHE = {}


def _ensure_log_dir(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)


def _build_file_handler(config: Config) -> RotatingFileHandler:
    """Create a safe file handler that will not attempt rollover in demo mode."""
    _ensure_log_dir(config.log_dir)
    path = os.path.join(config.log_dir, "tradewarrior.log")
    # Disable rollover (maxBytes=0) to avoid rename/locking issues on Windows.
    handler = RotatingFileHandler(path, maxBytes=0, backupCount=0, delay=True)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger configured for console + file (no rollover)."""
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = _build_file_handler(Config())
        logger.addHandler(fh)

    _LOGGER_CACHE[name] = logger
    return logger
