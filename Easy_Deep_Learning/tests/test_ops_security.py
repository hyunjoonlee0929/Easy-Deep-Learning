from __future__ import annotations

import json
from pathlib import Path

from Easy_Deep_Learning.core.observability import save_error_trace, track_tab_usage
from Easy_Deep_Learning.core.security import mask_api_key, validate_openai_key_format


def test_observability_usage_and_trace(tmp_path: Path) -> None:
    stats_path = tmp_path / "usage_stats.json"
    payload = track_tab_usage("Tabular", "session-a", base_path=stats_path)
    assert payload["tabs"]["Tabular"] == 1

    try:
        raise ValueError("demo")
    except Exception as exc:  # noqa: PERF203
        trace = save_error_trace("test", exc, {"k": "v"}, base_dir=tmp_path / "traces")
    trace_payload = json.loads(trace.read_text(encoding="utf-8"))
    assert trace_payload["scope"] == "test"
    assert trace_payload["error_type"] == "ValueError"


def test_key_mask_and_validation() -> None:
    assert mask_api_key("sk-1234567890ABCDEFGH").startswith("sk-1")
    assert validate_openai_key_format("sk-1234567890ABCDEFGH")
    assert not validate_openai_key_format("invalid-key")
