"""Application logging: readable stderr and bounded UTF-8 JSONL files."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
from pathlib import Path
import sys
import time

from .data import ROOT

MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = dict(getattr(record, 'fields', {}))
        payload.update(
            time=datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
        )
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, default=str)


class ConsoleFormatter(logging.Formatter):
    converter = time.gmtime

    def __init__(self):
        super().__init__('%(asctime)s %(levelname)-7s %(name)s: %(message)s', '%Y-%m-%dT%H:%M:%SZ')

    def format(self, record):
        message = super().format(record)
        fields = getattr(record, 'fields', None)
        if fields:
            message += ' ' + json.dumps(_json_safe(fields), ensure_ascii=False, allow_nan=False, default=str)
        return message


def _file_handler(path, level):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUPS,
                                  encoding='utf-8', delay=True)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    return handler


def configure_logging(level='INFO', log_file=None, *, stream=None):
    """Configure only our namespace, leaving third-party/root handlers alone."""
    logger = logging.getLogger('captcha_lab')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)
    logger.propagate = False
    console = logging.StreamHandler(sys.stderr if stream is None else stream)
    console.setLevel(level)
    console.setFormatter(ConsoleFormatter())
    logger.addHandler(console)
    path = Path(log_file) if log_file is not None else ROOT/'runs/app.jsonl'
    logger.addHandler(_file_handler(path, level))
    return path


@contextmanager
def session_logging(path):
    """Use logging handlers for the phone session too; always close the file."""
    logger = logging.getLogger('captcha_lab')
    handler = _file_handler(path, logger.getEffectiveLevel())
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
