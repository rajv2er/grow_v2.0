"""Tests for the online per-(student, subject, topic) mastery EMA."""
from __future__ import annotations

import pytest

from ml.online_mastery import (
    MIN_ATTEMPTS_FOR_OVERLAY,
    _logit,
    _sigmoid_safe,
    overlay_predictions,
    read_mastery,
    status_for_probability,
    update_after_attempt,
)


def test_logit_sigmoid_roundtrip():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert abs(_sigmoid_safe(_logit(p)) - p) < 1e-6


def test_status_thresholds():
    assert status_for_probability(0.85) == "Strong"
    assert status_for_probability(0.65) == "Developing"
    assert status_for_probability(0.30) == "Weak"


def test_first_init_uses_prior(tmp_path, monkeypatch):
    """First call for a (student, subject, topic) row uses the given prior."""
    from config import DATABASE_PATH
    from database import db as db_mod

    test_db = tmp_path / "test.db"
    monkeypatch.setattr("ml.online_mastery.DATABASE_PATH", test_db)
    monkeypatch.setattr(db_mod, "DATABASE_PATH", test_db)
    db_mod.initialise_database(test_db)
    with db_mod.connection(test_db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES (?,?,?,?)",
            ("U_FAKE", "fake", 0, "2026-01-01T00:00:00"),
        )
    new = update_after_attempt("U_FAKE", "DSA", "Arrays", 1, prior_mastery=0.7)
    assert 0.0 < new < 1.0
    df = read_mastery("U_FAKE")
    assert len(df) == 1
    row = df.iloc[0]
    assert row.n_attempts == 1
    assert row.n_correct == 1
    assert abs(row.prior_mastery - 0.7) < 1e-6


def test_consecutive_corrects_raise_estimate(tmp_path, monkeypatch):
    from database import db as db_mod

    test_db = tmp_path / "test.db"
    monkeypatch.setattr("ml.online_mastery.DATABASE_PATH", test_db)
    monkeypatch.setattr(db_mod, "DATABASE_PATH", test_db)
    db_mod.initialise_database(test_db)
    with db_mod.connection(test_db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES (?,?,?,?)",
            ("U_FAKE2", "fake2", 0, "2026-01-01T00:00:00"),
        )

    estimates = []
    for _ in range(8):
        e = update_after_attempt("U_FAKE2", "DBMS", "SQL", 1, prior_mastery=0.5)
        estimates.append(e)
    assert estimates[-1] > 0.95
    assert estimates == sorted(estimates)


def test_consecutive_wrongs_lower_estimate(tmp_path, monkeypatch):
    from database import db as db_mod

    test_db = tmp_path / "test.db"
    monkeypatch.setattr("ml.online_mastery.DATABASE_PATH", test_db)
    monkeypatch.setattr(db_mod, "DATABASE_PATH", test_db)
    db_mod.initialise_database(test_db)
    with db_mod.connection(test_db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES (?,?,?,?)",
            ("U_FAKE3", "fake3", 0, "2026-01-01T00:00:00"),
        )

    estimates = []
    for _ in range(8):
        e = update_after_attempt("U_FAKE3", "OS", "Deadlocks", 0, prior_mastery=0.5)
        estimates.append(e)
    assert estimates[-1] < 0.05
    assert estimates == sorted(estimates, reverse=True)


def test_overlay_keeps_global_below_threshold(tmp_path, monkeypatch):
    import pandas as pd
    from database import db as db_mod

    test_db = tmp_path / "test.db"
    monkeypatch.setattr("ml.online_mastery.DATABASE_PATH", test_db)
    monkeypatch.setattr(db_mod, "DATABASE_PATH", test_db)
    db_mod.initialise_database(test_db)
    with db_mod.connection(test_db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES (?,?,?,?)",
            ("U_NO_DATA", "nodata", 0, "2026-01-01T00:00:00"),
        )

    global_pred = pd.DataFrame(
        {
            "subject": ["DSA", "DSA"],
            "topic": ["Arrays", "Hashing"],
            "mastery_probability": [0.4, 0.6],
            "status": ["Weak", "Developing"],
        }
    )
    overlaid = overlay_predictions(global_pred, "U_NO_DATA", min_attempts=MIN_ATTEMPTS_FOR_OVERLAY)
    assert (overlaid.mastery_probability == global_pred.mastery_probability).all()
    assert not overlaid.used_online_mastery.any()


def test_overlay_replaces_when_enough_data(tmp_path, monkeypatch):
    import pandas as pd
    from database import db as db_mod

    test_db = tmp_path / "test.db"
    monkeypatch.setattr("ml.online_mastery.DATABASE_PATH", test_db)
    monkeypatch.setattr(db_mod, "DATABASE_PATH", test_db)
    db_mod.initialise_database(test_db)
    with db_mod.connection(test_db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES (?,?,?,?)",
            ("U_OVERLAY", "overlay", 0, "2026-01-01T00:00:00"),
        )

    for _ in range(5):
        update_after_attempt("U_OVERLAY", "DSA", "Arrays", 1, prior_mastery=0.4)
    global_pred = pd.DataFrame(
        {
            "subject": ["DSA", "DSA"],
            "topic": ["Arrays", "Hashing"],
            "mastery_probability": [0.4, 0.6],
            "status": ["Weak", "Developing"],
        }
    )
    overlaid = overlay_predictions(global_pred, "U_OVERLAY", min_attempts=3)
    arrays_row = overlaid[overlaid.topic == "Arrays"].iloc[0]
    hashing_row = overlaid[overlaid.topic == "Hashing"].iloc[0]
    assert arrays_row.mastery_probability > 0.85
    assert bool(arrays_row.used_online_mastery) is True
    assert abs(hashing_row.mastery_probability - 0.6) < 1e-6
    assert bool(hashing_row.used_online_mastery) is False
