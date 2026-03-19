from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from Easy_Deep_Learning.core.workflows import test_from_run as run_test_from_run, train_and_save


def test_imbalance_calibration_threshold_artifacts(project_root: Path, tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        weights=[0.9, 0.1],
        random_state=7,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    data_path = tmp_path / "imbalance.csv"
    df.to_csv(data_path, index=False)

    run = train_and_save(
        data_path=data_path,
        config_path=project_root / "config" / "model_config.yaml",
        target_column="target",
        task_type="classification",
        model_type="rf",
        seed=7,
        reuse_if_exists=False,
    )
    assert (run.run_path / "imbalance_report.json").exists()
    assert (run.run_path / "calibration_report.json").exists()
    assert (run.run_path / "threshold_report.json").exists()

    info = json.loads((run.run_path / "model_info.json").read_text(encoding="utf-8"))
    assert "decision_threshold" in info


def test_regression_prediction_interval_artifacts(project_root: Path, tmp_path: Path) -> None:
    X, y = make_regression(n_samples=160, n_features=6, noise=3.5, random_state=11)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["target"] = y
    data_path = tmp_path / "reg.csv"
    df.to_csv(data_path, index=False)

    run = train_and_save(
        data_path=data_path,
        config_path=project_root / "config" / "model_config.yaml",
        target_column="target",
        task_type="regression",
        model_type="rf",
        seed=11,
        reuse_if_exists=False,
    )
    assert (run.run_path / "prediction_interval.json").exists()

    payload = run_test_from_run(
        run_id=run.run_id,
        test_data_path=data_path,
        target_column="target",
        save_artifacts=False,
    )
    preview = payload["prediction_preview"]
    assert "lower" in preview and "upper" in preview
