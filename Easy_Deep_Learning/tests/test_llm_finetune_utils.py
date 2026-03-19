from __future__ import annotations

from pathlib import Path

import pandas as pd

from Easy_Deep_Learning.core.llm_finetune import get_safe_generation_preset, validate_llm_dataset


def test_safe_generation_presets() -> None:
    balanced = get_safe_generation_preset("balanced")
    unknown = get_safe_generation_preset("unknown_name")
    assert "max_new_tokens" in balanced
    assert unknown == balanced


def test_validate_llm_dataset_csv(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "prompt": [f"Q{i}: " for i in range(10)],
            "completion": [f"A{i}" for i in range(10)],
        }
    )
    path = tmp_path / "llm_train.csv"
    df.to_csv(path, index=False)
    report = validate_llm_dataset(path, "prompt", "completion")
    assert report["row_count"] == 10
