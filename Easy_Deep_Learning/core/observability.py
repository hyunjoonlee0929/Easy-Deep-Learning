"""Observability helpers: error traces and usage telemetry."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_error_trace(
    scope: str,
    exc: Exception,
    context: dict[str, Any] | None = None,
    base_dir: Path | str = Path("runs/error_traces"),
) -> Path:
    """Save exception traceback as a JSON artifact and return path."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{ts}_{scope}.json"
    payload = {
        "timestamp_utc": _utc_now(),
        "scope": scope,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "context": context or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def track_tab_usage(tab_name: str, session_id: str, base_path: Path | str = Path("runs/usage_stats.json")) -> dict[str, Any]:
    """Increment per-tab usage counter."""
    path = Path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"updated_at_utc": _utc_now(), "tabs": {}, "sessions": {}}

    tabs = payload.setdefault("tabs", {})
    sessions = payload.setdefault("sessions", {})
    tabs[tab_name] = int(tabs.get(tab_name, 0)) + 1
    sessions[session_id] = {
        "last_tab": tab_name,
        "updated_at_utc": _utc_now(),
    }
    payload["updated_at_utc"] = _utc_now()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
