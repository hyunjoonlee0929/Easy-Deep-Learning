from __future__ import annotations

from pathlib import Path

from Easy_Deep_Learning.core.workflows import (
    cross_validate_and_report,
    test_from_run as run_test_from_run,
    train_and_save,
)


def test_tabular_train_test_cv_smoke(project_root: Path) -> None:
    data = project_root / "data" / "example_dataset.csv"
    config = project_root / "config" / "model_config.yaml"

    run = train_and_save(
        data_path=data,
        config_path=config,
        target_column="target",
        task_type="classification",
        model_type="rf",
        seed=42,
    )
    assert run.run_path.exists()
    assert "accuracy" in run.metrics

    payload = run_test_from_run(
        run_id=run.run_id,
        test_data_path=data,
        target_column="target",
        save_artifacts=False,
    )
    assert "metrics" in payload
    assert "accuracy" in payload["metrics"]

    cv = cross_validate_and_report(
        data_path=data,
        target_column="target",
        task_type="classification",
        model_type="rf",
        seed=42,
        folds=3,
        model_params={},
    )
    assert "mean_metrics" in cv
    assert "accuracy" in cv["mean_metrics"]
