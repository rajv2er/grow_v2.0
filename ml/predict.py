"""Inference and transparent weakness explanations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from config import DATABASE_PATH
from database.db import connection
from ml.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, current_topic_feature_rows


def status_for_probability(probability: float) -> str:
    if probability >= 0.75: return "Strong"
    if probability >= 0.55: return "Developing"
    return "Weak"


def explain_weakness(row: pd.Series) -> dict:
    reasons = []
    topic_attempts = int(row.get("topic_prior_attempts", row["prior_attempts"]))
    if topic_attempts == 0:
        reasons.append("no attempts yet on this topic — estimate uses the model's prior")
    else:
        if row["recent_accuracy_5"] < 0.55: reasons.append(f"recent accuracy is {row['recent_accuracy_5']:.0%}")
        if row["topic_avg_response_time"] > 120: reasons.append(f"average response time is {row['topic_avg_response_time']:.0f} seconds")
        if row["difficulty_accuracy"] < 0.45: reasons.append(f"{row['difficulty']}-difficulty accuracy is {row['difficulty_accuracy']:.0%}")
        if row["improvement_trend"] < -0.10: reasons.append("performance has declined across recent attempts")
    if not reasons: reasons.append("limited evidence is available; continue diagnostic practice")
    return {"reasons": reasons, "recommended_action": f"Practise {row['topic']} at {row['difficulty']} level, then reassess."}


def predict_student_mastery(model_path: Path, attempts: pd.DataFrame, questions: pd.DataFrame, student_id: str, db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    model = joblib.load(model_path)
    feature_rows = current_topic_feature_rows(attempts, student_id, questions)
    probabilities = model.predict_proba(feature_rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    result = feature_rows.copy()
    result["mastery_probability"] = probabilities
    result["status"] = result.mastery_probability.map(status_for_probability)
    result["explanation"] = result.apply(explain_weakness, axis=1)
    result["data_label"] = "Predictions based on SYNTHETIC / SIMULATED practice data"
    with connection(db_path) as conn:
        # Snapshot semantics: keep only the latest prediction set per student and model.
        conn.execute("DELETE FROM mastery_predictions WHERE student_id=? AND model_name=?", (student_id, model_path.stem))
        rows = [(
            student_id, r.subject, r.topic, float(r.mastery_probability), r.status,
            model_path.stem, datetime.now(timezone.utc).isoformat(), json.dumps(r.explanation),
        ) for r in result.itertuples(index=False)]
        conn.executemany(
            """INSERT INTO mastery_predictions(student_id, subject, topic, mastery_probability,
               status, model_name, predicted_at, explanation_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows,
        )
    return result.sort_values(["mastery_probability", "subject", "topic"]).reset_index(drop=True)
