from pathlib import Path
import sys

from loguru import logger

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def init_logger(level: str = "INFO"):
    """Configure loguru to output to stderr and a rotating file.

    The function is idempotent – calling it multiple times will not create
    duplicate handlers.  Call it once at application start-up and import the
    configured ``logger`` from loguru everywhere else.
    """
    if not logger._core.handlers:  # type: ignore[attr-defined]
        # Remove the default handler added by loguru.
        logger.remove()
        # Human-readable stderr handler.
        logger.add(sys.stderr, level=level)
        # File handler with automatic rotation.
        logger.add(
            _LOG_DIR / "app.log",
            rotation="5 MB",
            enqueue=True,
            encoding="utf-8",
            level="DEBUG",
        )
    return logger 