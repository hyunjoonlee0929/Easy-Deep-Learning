"""Data quality checks for tabular datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _zscore_outliers(series: pd.Series, z_thresh: float = 3.0) -> int:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        return 0
    mean = vals.mean()
    std = vals.std()
    if std == 0 or np.isnan(std):
        return 0
    z = (vals - mean) / std
    return int((np.abs(z) > z_thresh).sum())


def compute_data_quality(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    missing = df.isna().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    outliers = {col: _zscore_outliers(df[col]) for col in numeric_cols}

    target_summary = {}
    if target_column and target_column in df.columns:
        target = df[target_column]
        if pd.api.types.is_numeric_dtype(target):
            target_summary = {
                "type": "numeric",
                "skew": float(pd.to_numeric(target, errors="coerce").skew()),
            }
        else:
            counts = target.value_counts()
            if len(counts) > 0:
                ratio = float(counts.max() / max(counts.min(), 1))
            else:
                ratio = 0.0
            target_summary = {
                "type": "categorical",
                "class_counts": counts.to_dict(),
                "imbalance_ratio": ratio,
            }

    warnings = []
    missing_total = int(sum(missing.values()))
    missing_rate = float(missing_total / max(1, df.size))
    if duplicate_rows > 0:
        warnings.append({"level": "warning", "message": f"Found {duplicate_rows} duplicate rows."})
    if missing_total > 0:
        level = "critical" if missing_rate >= 0.1 else "warning"
        warnings.append({"level": level, "message": f"Missing values detected ({missing_rate:.2%})."})
    if target_summary.get("type") == "categorical":
        if target_summary.get("imbalance_ratio", 0) >= 10:
            warnings.append({"level": "warning", "message": "Target class imbalance is high."})

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing": missing,
        "duplicates": duplicate_rows,
        "outliers": outliers,
        "target_summary": target_summary,
        "missing_rate": missing_rate,
        "warnings": warnings,
    }
