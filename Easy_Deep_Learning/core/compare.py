"""Compare multiple runs and generate a summary report."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from Easy_Deep_Learning.core.automl import score_metrics


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_compare_report(run_ids: list[str]) -> dict[str, Any]:
    rows = []
    for rid in run_ids:
        run_path = Path("runs") / rid
        metrics = _read_json(run_path / "metrics.json") or {}
        info = _read_json(run_path / "model_info.json") or {}
        task_type = info.get("task_type", "")
        model_type = info.get("model_type", "")
        score = score_metrics(task_type, metrics) if task_type else None
        row = {
            "run_id": rid,
            "model_type": model_type,
            "task_type": task_type,
            "metrics": metrics,
            "score": score,
        }
        rows.append(row)

    best = None
    if rows:
        rows_sorted = sorted(rows, key=lambda r: r.get("score", -1e9), reverse=True)
        best = rows_sorted[0]
    else:
        rows_sorted = []

    reason = ""
    if best:
        metrics = best.get("metrics", {})
        task = best.get("task_type", "")
        if task == "classification":
            key = "f1_weighted" if "f1_weighted" in metrics else "accuracy"
        else:
            key = "r2"
        reason = f"Best score on {key}: {metrics.get(key)} with model {best.get('model_type')}"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": rows_sorted,
        "best_run": best,
        "reason": reason,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_rows = []
    for row in rows_sorted:
        html_rows.append(
            "<tr>"
            f"<td>{row['run_id']}</td>"
            f"<td>{row['model_type']}</td>"
            f"<td>{row['task_type']}</td>"
            f"<td>{row.get('score')}</td>"
            f"<td><pre>{json.dumps(row['metrics'], indent=2)}</pre></td>"
            "</tr>"
        )
    html = (
        "<html><body>"
        "<h1>Run Comparison</h1>"
        f"<div>Generated: {payload['generated_at']}</div>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>run_id</th><th>model_type</th><th>task_type</th><th>score</th><th>metrics</th></tr>"
        + "".join(html_rows)
        + "</table>"
        "</body></html>"
    )
    (out_dir / "compare_report.html").write_text(html, encoding="utf-8")

    payload["report_dir"] = str(out_dir.resolve())
    return payload
