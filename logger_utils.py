import logging
import os
from logging.handlers import RotatingFileHandler

from config import Config

_LOGGER_CACHE = {}


def _ensure_log_dir(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger configured for console + rotating file."""
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # File handler
        config = Config()  # default DEMO config
        _ensure_log_dir(config.log_dir)
        fh = RotatingFileHandler(
            os.path.join(config.log_dir, "tradewarrior.log"),
            maxBytes=1_000_000,
            backupCount=5,
        )
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    _LOGGER_CACHE[name] = logger
    return logger
