from datetime import datetime, timedelta, timezone

from database.db import (
    complete_queue_item, connection, due_queue_items, enqueue_questions,
    initialise_database, pending_queue_items, seed_question_bank,
)
from data.questions.question_bank import build_question_bank


def _db(tmp_path):
    db = tmp_path / "queue_test.db"
    initialise_database(db)
    seed_question_bank(build_question_bank(), db)
    with connection(db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) VALUES ('S1', 'Test', 1, '2026-01-01')"
        )
    return db


def test_enqueue_skips_already_pending_items(tmp_path):
    db = _db(tmp_path)
    item = {"question_id": "Q0101A", "subject": "DSA", "topic": "Arrays"}
    now = datetime.now(timezone.utc).isoformat()
    assert enqueue_questions("S1", [item], now, db_path=db) == 1
    assert enqueue_questions("S1", [item], now, db_path=db) == 0
    assert len(pending_queue_items("S1", db_path=db)) == 1


def test_only_due_items_are_served(tmp_path):
    db = _db(tmp_path)
    now = datetime.now(timezone.utc)
    due = {"question_id": "Q0101A", "subject": "DSA", "topic": "Arrays"}
    later = {"question_id": "Q0102A", "subject": "DSA", "topic": "Strings"}
    enqueue_questions("S1", [due], now.isoformat(), db_path=db)
    enqueue_questions("S1", [later], (now + timedelta(days=1)).isoformat(), db_path=db)
    served = due_queue_items("S1", now.isoformat(), limit=5, db_path=db)
    assert [row["question_id"] for row in served] == ["Q0101A"]


def test_subject_filter_restricts_served_items(tmp_path):
    db = _db(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    enqueue_questions("S1", [
        {"question_id": "Q0101A", "subject": "DSA", "topic": "Arrays"},
        {"question_id": "Q0201A", "subject": "DBMS", "topic": "DBMS Basics"},
    ], now, db_path=db)
    served = due_queue_items("S1", now, subject="DBMS", limit=5, db_path=db)
    assert [row["question_id"] for row in served] == ["Q0201A"]


def test_completed_items_leave_the_pending_queue(tmp_path):
    db = _db(tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    enqueue_questions("S1", [{"question_id": "Q0101A", "subject": "DSA", "topic": "Arrays"}], now, db_path=db)
    complete_queue_item("S1", "Q0101A", db_path=db)
    assert pending_queue_items("S1", db_path=db) == []
    assert due_queue_items("S1", now, db_path=db) == []


def test_migration_rebuilds_legacy_questions_table(tmp_path):
    """A pre-subjective database (NOT NULL options, A-D CHECK) must migrate cleanly."""
    import sqlite3
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE questions (
            question_id TEXT PRIMARY KEY, subject TEXT NOT NULL, topic TEXT NOT NULL,
            question TEXT NOT NULL, difficulty TEXT NOT NULL,
            option_a TEXT NOT NULL, option_b TEXT NOT NULL, option_c TEXT NOT NULL, option_d TEXT NOT NULL,
            correct_answer TEXT NOT NULL CHECK(correct_answer IN ('A','B','C','D')), explanation TEXT NOT NULL
        );
        CREATE TABLE students (student_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, is_synthetic INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, question_id TEXT NOT NULL,
            subject TEXT NOT NULL, topic TEXT NOT NULL, difficulty TEXT NOT NULL, is_correct INTEGER NOT NULL,
            time_taken_seconds REAL NOT NULL, attempt_number INTEGER NOT NULL, timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL, is_synthetic INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO questions VALUES ('QLEG01','DSA','Arrays','q','Easy','a','b','c','d','A','Because.');
    """)
    conn.commit(); conn.close()
    seed_question_bank(build_question_bank(), db)
    check = sqlite3.connect(db)
    cols = {row[1] for row in check.execute("PRAGMA table_info(questions)")}
    assert {"question_type", "difficulty_rating", "model_answer"} <= cols
    legacy = check.execute("SELECT difficulty_rating, question_type FROM questions WHERE question_id='QLEG01'").fetchone()
    assert legacy == (0.25, "MCQ")
    subjective = check.execute("SELECT COUNT(*) FROM questions WHERE question_type='Subjective'").fetchone()[0]
    assert subjective == 60
    check.close()
