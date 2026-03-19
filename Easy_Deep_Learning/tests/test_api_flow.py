from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from Easy_Deep_Learning.api.app import app
from Easy_Deep_Learning.core.workflows import train_and_save


def _prepare_run(project_root: Path) -> str:
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
    return run.run_id


def test_predict_endpoint_smoke(project_root: Path) -> None:
    run_id = _prepare_run(project_root)
    data = project_root / "data" / "example_dataset.csv"
    records = [
        {
            "age": 34,
            "bmi": 22.1,
            "glucose": 95,
            "smoker": "no",
            "exercise": "high",
        }
    ]

    client = TestClient(app)
    res = client.post("/predict", json={"run_id": run_id, "records": records, "target_column": "target"})
    assert res.status_code == 200
    payload = res.json()
    assert "predictions" in payload


def test_llm_endpoints_mockable(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr("Easy_Deep_Learning.api.app.generate_with_lora", lambda **_: "hello")
    res = client.post(
        "/llm/generate",
        json={"run_id": "dummy_llm_finetune", "prompt": "hi", "max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9},
    )
    assert res.status_code == 200
    assert res.json()["output"] == "hello"

    monkeypatch.setattr("Easy_Deep_Learning.api.app.generate_chat_with_lora", lambda **_: "chat")
    res = client.post(
        "/llm/chat",
        json={
            "run_id": "dummy_llm_finetune",
            "messages": [{"role": "user", "content": "hello"}],
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )
    assert res.status_code == 200
    assert res.json()["output"] == "chat"
