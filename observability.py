"""Centralised logging for MasteryLab.

Writes structured JSON-ish log lines to logs/app.log (rotated) and to
stderr. Every significant pipeline step is logged: model load, prediction
latency, recommender output, DB writes. The goal is to be able to answer
"what happened in this session" from the log file alone, without re-running
anything.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import time
from contextlib import contextmanager
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def _build() -> logging.Logger:
    logger = logging.getLogger("masterylab")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


log = _build()


@contextmanager
def timed(stage: str, **fields: object):
    """Log elapsed time for a stage, with extra context fields."""
    start = time.perf_counter()
    log.info("stage=%s start extra=%s", stage, fields)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log.info("stage=%s done elapsed_ms=%.2f extra=%s", stage, elapsed_ms, fields)


def get_logger() -> logging.Logger:
    return log
