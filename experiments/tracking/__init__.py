"""Append-only JSONL log of every experiment run.

Each line is one record with: timestamp, n_students, n_attempts, n_features,
seed, best_model, and a flat copy of the per-model metrics. Used for
post-hoc comparison across runs without re-executing anything.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACKING_DIR = Path(__file__).resolve().parent
TRACKING_FILE = TRACKING_DIR / "runs.jsonl"


def record(
    *,
    simulation_manifest: dict | None,
    dataset_statistics: dict,
    best_model: str,
    model_comparison: Any,
    feature_names: list[str] | None = None,
) -> Path:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        metrics = json.loads(model_comparison.to_json(orient="records"))
    except Exception:
        metrics = []
    record_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulation_manifest": simulation_manifest or {},
        "dataset_statistics": dataset_statistics,
        "best_model": best_model,
        "feature_names": feature_names or [],
        "model_comparison": metrics,
    }
    with TRACKING_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict, default=str) + "\n")
    return TRACKING_FILE


def all_runs() -> list[dict]:
    if not TRACKING_FILE.exists():
        return []
    with TRACKING_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
