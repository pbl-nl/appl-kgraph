from __future__ import annotations

import logging
from pathlib import Path


def configure_file_logger(
    name: str,
    *,
    log_file: Path,
    level: str = "INFO",
    enabled: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    logger.propagate = False

    handler_key = "appl_kgraph_managed"
    for handler in list(logger.handlers):
        if getattr(handler, handler_key, False):
            logger.removeHandler(handler)
            handler.close()

    if enabled:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        setattr(handler, handler_key, True)
        logger.addHandler(handler)

    return logger
