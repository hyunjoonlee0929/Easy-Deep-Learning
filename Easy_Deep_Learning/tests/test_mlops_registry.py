from __future__ import annotations

import json
from pathlib import Path

from Easy_Deep_Learning.core.model_registry_layer import ModelRegistry
from Easy_Deep_Learning.core.workflows import train_and_save


def test_mlops_metadata_registry_and_cards(project_root: Path) -> None:
    data = project_root / "data" / "example_dataset.csv"
    config = project_root / "config" / "model_config.yaml"

    run = train_and_save(
        data_path=data,
        config_path=config,
        target_column="target",
        task_type="classification",
        model_type="rf",
        seed=123,
        reuse_if_exists=False,
    )
    run_path = run.run_path

    standard_meta = run_path / "run_metadata.standard.json"
    run_card = run_path / "run_card.json"
    registry_tags = run_path / "registry_tags.json"

    assert standard_meta.exists()
    assert run_card.exists()
    assert registry_tags.exists()

    meta = json.loads(standard_meta.read_text(encoding="utf-8"))
    assert meta["run_type"] == "tabular"
    assert "dataset_hash" in meta
    assert "env" in meta
    assert "model_signature" in meta

    registry = ModelRegistry()
    latest_tag = f"latest:tabular:classification:{meta['model_type']}"
    assert registry.resolve_tag(latest_tag) is not None
