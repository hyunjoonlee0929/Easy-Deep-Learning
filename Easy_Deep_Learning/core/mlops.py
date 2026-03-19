"""MLOps helpers: standard metadata, registry updates, and run cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker
from Easy_Deep_Learning.core.model_cards import generate_run_card
from Easy_Deep_Learning.core.model_registry_layer import ModelRegistry


def finalize_run_tracking(
    run_path: Path,
    run_type: str,
    task_type: str,
    model_type: str,
    dataset_hash: str,
    metrics: dict[str, Any],
    model_params: dict[str, Any] | None = None,
    model_artifact: str | None = None,
    config_hash: str | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist standardized metadata, register run, and generate run cards."""
    tracker = ExperimentTracker(base_dir=Path("runs"))
    model_signature = tracker.build_model_signature(
        model_type=model_type,
        task_type=task_type,
        model_params=model_params or {},
        model_artifact=model_artifact,
    )
    standard_meta = tracker.save_standard_run_metadata(
        run_path=run_path,
        run_type=run_type,
        task_type=task_type,
        model_type=model_type,
        dataset_hash=dataset_hash,
        model_signature=model_signature,
        config_hash=config_hash,
        seed=seed,
        extra=extra or {},
    )

    registry = ModelRegistry(runs_dir=Path("runs"))
    tags = registry.register_run(
        run_id=run_path.name,
        run_type=run_type,
        model_type=model_type,
        task_type=task_type,
        metrics=metrics,
        dataset_hash=dataset_hash,
        model_signature=model_signature["signature_hash"],
    )
    tracker.save_json(run_path / "registry_tags.json", tags)

    card_json, card_md = generate_run_card(run_path)
    return {
        "run_metadata_standard_path": str((run_path / "run_metadata.standard.json").resolve()),
        "registry_tags_path": str((run_path / "registry_tags.json").resolve()),
        "run_card_json_path": str(card_json.resolve()),
        "run_card_md_path": str(card_md.resolve()),
        "registry_tags": tags,
        "model_signature": model_signature,
        "standard_metadata": standard_meta,
    }
