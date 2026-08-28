"""Service layer: orchestrates predict / recommend / mastery updates.

The Streamlit UI and the FastAPI endpoints both call into this module
rather than reaching into db/ml/recommendation modules directly. This
separation gives us one place to test the feedback loop end-to-end, and
one place to add cross-cutting concerns (logging, metrics, validation)
without touching the UI or the API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.questions.question_bank import build_question_bank
from database.db import connection
from ml.online_mastery import (
    overlay_predictions as _overlay_mastery,
    read_mastery,
    update_after_attempt as _update_ema,
)
from ml.predict import predict_student_mastery
from recommendation.recommender import recommend_questions as _recommend_questions


def current_mastery(student_id: str, db_path: Path | None = None) -> pd.DataFrame:
    """Return the global model predictions overlaid with per-user EMA."""
    if db_path is None:
        from config import DATABASE_PATH
        db_path = DATABASE_PATH
    attempts = _user_attempts(student_id, db_path)
    if attempts.empty:
        return pd.DataFrame()
    from app.main import model_path

    m = model_path()
    if m is None:
        return pd.DataFrame()
    global_pred = predict_student_mastery(m, attempts, _questions(), student_id, db_path=db_path)
    return _overlay_mastery(global_pred, student_id, db_path=db_path)


def _questions() -> pd.DataFrame:
    return pd.DataFrame(build_question_bank())


def _user_attempts(student_id: str, db_path: Path | None = None) -> pd.DataFrame:
    if db_path is None:
        from config import DATABASE_PATH
        db_path = DATABASE_PATH
    with connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM attempts WHERE student_id=? ORDER BY timestamp",
            conn,
            params=(student_id,),
        )


def next_recommendations(student_id: str, limit: int = 5, db_path: Path | None = None) -> pd.DataFrame:
    """Compute mastery + recommendations in one call. Snapshot semantics."""
    if db_path is None:
        from config import DATABASE_PATH
        db_path = DATABASE_PATH
    p = current_mastery(student_id, db_path)
    if p.empty:
        return pd.DataFrame()
    return _recommend_questions(
        student_id, p, _user_attempts(student_id, db_path), _questions(), limit=limit, db_path=db_path,
    )


def record_answer(
    student_id: str,
    question: pd.Series,
    is_correct: bool,
    seconds: float,
    confidence: int,
    answer_text: str | None = None,
    score: float | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist an attempt and update the per-user EMA in one transaction."""
    if db_path is None:
        from config import DATABASE_PATH
        db_path = DATABASE_PATH
    with connection(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM attempts WHERE student_id=?", (student_id,)).fetchone()[0] + 1
        conn.execute(
            "INSERT INTO attempts(student_id, question_id, subject, topic, difficulty, is_correct, "
            "time_taken_seconds, attempt_number, timestamp, session_id, confidence_rating, "
            "is_synthetic, answer_text, score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                student_id, question["question_id"], question["subject"], question["topic"],
                question["difficulty"], int(bool(is_correct)), max(seconds, 1), n,
                datetime.now(timezone.utc).isoformat(), f"{student_id}_PRACTICE",
                confidence, 0, answer_text, score,
            ),
        )
    prior = _prior_for(student_id, question["subject"], question["topic"], db_path)
    _update_ema(
        student_id, question["subject"], question["topic"], int(bool(is_correct)),
        prior_mastery=prior, db_path=db_path,
    )


def _prior_for(student_id: str, subject: str, topic: str, db_path: Path | None = None) -> float | None:
    p = current_mastery(student_id, db_path)
    if p.empty:
        return None
    rows = p[(p.subject == subject) & (p.topic == topic)]
    return float(rows.mastery_probability.iloc[0]) if not rows.empty else None
