"""Logging setup utilities for OmniInsight."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the application."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
