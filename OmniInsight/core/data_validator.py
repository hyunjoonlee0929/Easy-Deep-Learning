"""Data validation utilities for tabular and multi-omics data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ValidationReport:
    """Structured output of dataset validation."""

    n_rows: int
    n_columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_by_column: dict[str, int]
    target_column: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a serializable dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "missing_by_column": self.missing_by_column,
            "target_column": self.target_column,
        }


class DataValidator:
    """Validate tabular datasets and provide a concise summary."""

    def validate(self, df: pd.DataFrame, target_column: str | None = None) -> ValidationReport:
        """Run basic checks and return a validation report.

        Args:
            df: Input dataset.
            target_column: Optional target name.

        Returns:
            ValidationReport with column type and missing-value summary.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [col for col in df.columns if col not in numeric_columns]
        missing_by_column = df.isna().sum().astype(int).to_dict()

        if target_column is not None and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' was not found.")

        return ValidationReport(
            n_rows=len(df),
            n_columns=len(df.columns),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            missing_by_column=missing_by_column,
            target_column=target_column,
        )
