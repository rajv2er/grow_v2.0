"""End-to-end test of the recommend → answer → mastery update feedback loop.

This is the single most important test in the suite. It proves that:
- The recommender picks a topic based on the current mastery state.
- A recorded answer updates the per-user EMA mastery in `student_topic_mastery`.
- The next call to the recommender reflects the new mastery.

If this test ever fails, the core promise of the project ("the system
learns from the user") has been broken.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app import service
from data.questions.question_bank import build_question_bank
from ml.online_mastery import read_mastery
from recommendation.recommender import recommend_questions


@pytest.fixture
def student_id(tmp_path, monkeypatch):
    """Create a fresh student in a temp DB and patch the DB path for the test."""
    from config import DATABASE_PATH
    from database import db as db_mod
    from database.db import initialise_database

    test_db = tmp_path / "e2e.db"
    monkeypatch.setattr("ml.online_mastery.DATABASE_PATH", test_db)
    monkeypatch.setattr(db_mod, "DATABASE_PATH", test_db)
    monkeypatch.setattr("app.service.DATABASE_PATH", test_db) if hasattr(service, "DATABASE_PATH") else None
    initialise_database(test_db)
    with db_mod.connection(test_db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES (?,?,?,?)",
            ("U_E2E", "e2e", 0, "2026-01-01T00:00:00"),
        )
    return "U_E2E", test_db


@pytest.fixture(autouse=True)
def suppress_joblib_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*array.shape.*")
        yield


def _seed_attempts(student_id: str, db_path, n: int = 6):
    """Insert n historical attempts directly so the global model has signal."""
    from database.db import seed_question_bank

    qb = pd.DataFrame(build_question_bank())
    seed_question_bank(qb.to_dict("records"), db_path=db_path)
    arrays = qb[qb.topic == "Arrays"].head(n)
    if len(arrays) < n:
        arrays = qb.head(n)
    with __import__("database.db", fromlist=["connection"]).connection(db_path) as conn:
        for i, row in enumerate(arrays.itertuples(index=False), 1):
            conn.execute(
                "INSERT INTO attempts(student_id, question_id, subject, topic, difficulty, "
                "is_correct, time_taken_seconds, attempt_number, timestamp, session_id, "
                "confidence_rating, is_synthetic) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    student_id, row.question_id, row.subject, row.topic, row.difficulty,
                    int(i % 3 == 0), 60.0, i, f"2026-01-{i:02d}T12:00:00",
                    f"{student_id}_E2E", 3, 0,
                ),
            )


def test_feedback_loop_changes_recommendation(student_id):
    sid, db_path = student_id
    _seed_attempts(sid, db_path, n=6)
    qb = pd.DataFrame(build_question_bank())
    # Use the service so we exercise the real path.
    mastery_before = service.current_mastery(sid, db_path=db_path)
    assert not mastery_before.empty, "service.current_mastery must return rows for a user with attempts"
    recs_before = service.next_recommendations(sid, limit=5, db_path=db_path)
    assert not recs_before.empty, "recommender must return at least one item"

    # Pick the top recommended question and answer it correctly.
    top = recs_before.iloc[0]
    q = qb[qb.question_id == top.question_id].iloc[0]
    service.record_answer(sid, q, is_correct=True, seconds=42, confidence=4, db_path=db_path)

    # Mastery for that (subject, topic) should have a fresh EMA row.
    ema = read_mastery(sid, db_path=db_path)
    assert not ema.empty, "EMA must have at least one row after the first answer"
    row = ema[(ema.subject == q["subject"]) & (ema.topic == q["topic"])].iloc[0]
    assert row.n_attempts == 1
    assert row.n_correct == 1
    assert 0.0 < float(row.mastery_estimate) < 1.0

    # After 3 correct answers the EMA should be near 1.0 and the
    # recommender should drop this topic from the top of the list.
    for _ in range(2):
        service.record_answer(sid, q, is_correct=True, seconds=30, confidence=5, db_path=db_path)
    ema = read_mastery(sid, db_path=db_path)
    row = ema[(ema.subject == q["subject"]) & (ema.topic == q["topic"])].iloc[0]
    assert row.n_attempts == 3
    assert row.mastery_estimate > 0.85, f"after 3 correct answers mastery should be high, got {row.mastery_estimate}"

    recs_after = service.next_recommendations(sid, limit=5, db_path=db_path)
    # The topic we just aced should NOT be at the top of the new plan.
    top_after = recs_after.iloc[0]
    assert not (
        top_after.subject == q["subject"] and top_after.topic == q["topic"]
    ), "After 3 correct answers the topic should drop from the top recommendation"


def test_wrong_answers_keep_topic_in_recommendation(student_id):
    sid, db_path = student_id
    _seed_attempts(sid, db_path, n=6)
    qb = pd.DataFrame(build_question_bank())
    recs_before = service.next_recommendations(sid, limit=5, db_path=db_path)
    top = recs_before.iloc[0]
    q = qb[qb.question_id == top.question_id].iloc[0]

    # Three wrong answers — topic should stay (or move up) in the plan.
    for _ in range(3):
        service.record_answer(sid, q, is_correct=False, seconds=120, confidence=1, db_path=db_path)
    ema = read_mastery(sid, db_path=db_path)
    row = ema[(ema.subject == q["subject"]) & (ema.topic == q["topic"])].iloc[0]
    assert row.mastery_estimate < 0.20, f"3 wrong answers should drop mastery below 0.2, got {row.mastery_estimate}"

    recs_after = service.next_recommendations(sid, limit=5, db_path=db_path)
    topics_after = {(r.subject, r.topic) for r in recs_after.itertuples(index=False)}
    assert (q["subject"], q["topic"]) in topics_after, "Struggling topic should remain in the recommended plan"
