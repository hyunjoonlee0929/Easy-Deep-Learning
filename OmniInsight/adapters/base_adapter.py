"""Base adapter defining the OmniInsight workflow interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from OmniInsight.core.model_engine import ModelResult


class BaseAdapter(ABC):
    """Abstract interface for domain-specific analysis adapters."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Validate and preprocess data."""

    @abstractmethod
    def train(self, processed: dict[str, Any]) -> ModelResult:
        """Train model on processed data."""

    @abstractmethod
    def interpret(self, processed: dict[str, Any], model_result: ModelResult) -> dict[str, Any]:
        """Interpret model outputs."""

    @abstractmethod
    def generate_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate final insight report payload."""
