"""Simple drift detection for tabular datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if expected.size == 0 or actual.size == 0:
        return 0.0
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_drift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in train_df.columns if c not in numeric_cols]

    numeric_drift = {}
    for col in numeric_cols:
        tr = pd.to_numeric(train_df[col], errors="coerce").dropna().to_numpy()
        te = pd.to_numeric(test_df[col], errors="coerce").dropna().to_numpy()
        numeric_drift[col] = _psi(tr, te)

    cat_drift = {}
    for col in cat_cols:
        tr = train_df[col].astype(str).fillna("NA")
        te = test_df[col].astype(str).fillna("NA")
        tr_counts = tr.value_counts(normalize=True)
        te_counts = te.value_counts(normalize=True)
        keys = set(tr_counts.index).union(set(te_counts.index))
        psi = 0.0
        for k in keys:
            e = tr_counts.get(k, 1e-6)
            a = te_counts.get(k, 1e-6)
            psi += (a - e) * np.log(a / e)
        cat_drift[col] = float(psi)

    warnings = []
    high_numeric = [k for k, v in numeric_drift.items() if v >= 0.2]
    high_cat = [k for k, v in cat_drift.items() if v >= 0.2]
    if high_numeric:
        warnings.append({"level": "warning", "message": f"High numeric drift: {', '.join(high_numeric[:5])}"})
    if high_cat:
        warnings.append({"level": "warning", "message": f"High categorical drift: {', '.join(high_cat[:5])}"})

    return {
        "numeric_psi": numeric_drift,
        "categorical_psi": cat_drift,
        "warnings": warnings,
    }
