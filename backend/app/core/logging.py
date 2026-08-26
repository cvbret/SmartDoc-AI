"""Application-wide logging configuration."""

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
)


def setup_logging(level: str | None = None) -> None:
    """Configure one console handler for the application.

    When no level is supplied, the value is read from application settings.
    ``force=True`` makes repeated calls idempotent and prevents a previously
    configured root handler from causing duplicate output.
    """
    if level is None:
        from app.core.config import settings

        level = settings.log_level

    level_name = level.upper()
    numeric_level = logging.getLevelName(level_name)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid LOG_LEVEL {level!r}; use a standard logging level"
        )

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )

    for logger_name in NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
