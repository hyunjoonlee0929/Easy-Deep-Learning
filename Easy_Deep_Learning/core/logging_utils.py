"""Logging setup utilities for Easy Deep Learning."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the application."""
    if logging.getLogger().handlers:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_mode = os.getenv("EASY_DL_LOG_JSON", "1") == "1"

    stream = logging.StreamHandler()
    file_handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    if json_mode:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root.addHandler(stream)
    root.addHandler(file_handler)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured event log line."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))
