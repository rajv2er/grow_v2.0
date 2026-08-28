"""Database creation, seeding and export helpers."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import DATABASE_PATH
from database.schema import SCHEMA


def connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _rebuild_questions_table(conn: sqlite3.Connection) -> None:
    """Rebuild questions so MCQ-only NOT NULL/CHECK constraints stop blocking subjective rows."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript("""
        CREATE TABLE questions_new (
            question_id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            question_type TEXT NOT NULL DEFAULT 'MCQ' CHECK(question_type IN ('MCQ', 'Subjective')),
            difficulty TEXT NOT NULL CHECK(difficulty IN ('Easy', 'Medium', 'Hard')),
            difficulty_rating REAL NOT NULL CHECK(difficulty_rating BETWEEN 0.1 AND 1.0),
            option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
            correct_answer TEXT, model_answer TEXT,
            explanation TEXT NOT NULL
        );
        INSERT INTO questions_new
            (question_id, subject, topic, question, question_type, difficulty, difficulty_rating,
             option_a, option_b, option_c, option_d, correct_answer, model_answer, explanation)
        SELECT question_id, subject, topic, question, 'MCQ', difficulty,
               CASE difficulty WHEN 'Easy' THEN 0.25 WHEN 'Medium' THEN 0.55 ELSE 0.85 END,
               option_a, option_b, option_c, option_d, correct_answer, NULL, explanation
        FROM questions;
        DROP TABLE questions;
        ALTER TABLE questions_new RENAME TO questions;
    """)
    conn.execute("PRAGMA foreign_keys = ON")


def _relax_questions_check(conn: sqlite3.Connection) -> None:
    """Rebuild the questions table with the expanded question_type CHECK and
    extra columns (blanks_json, correct_answers_json, expected_value, tolerance).
    Preserves all existing rows. Idempotent: drops any leftover questions_new
    from a prior failed attempt.
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DROP TABLE IF EXISTS questions_new")
    conn.executescript("""
        CREATE TABLE questions_new (
            question_id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            question_type TEXT NOT NULL DEFAULT 'MCQ'
                CHECK(question_type IN ('MCQ','Subjective','TrueFalse','MultipleSelect','FillInBlank','Numerical')),
            difficulty TEXT NOT NULL CHECK(difficulty IN ('Easy','Medium','Hard')),
            difficulty_rating REAL NOT NULL CHECK(difficulty_rating BETWEEN 0.1 AND 1.0),
            option_a TEXT, option_b TEXT, option_c TEXT, option_d TEXT,
            correct_answer TEXT, model_answer TEXT, explanation TEXT NOT NULL,
            blanks_json TEXT, correct_answers_json TEXT,
            expected_value REAL, tolerance REAL DEFAULT 0.01
        );
        INSERT INTO questions_new
            (question_id, subject, topic, question, question_type, difficulty, difficulty_rating,
             option_a, option_b, option_c, option_d, correct_answer, model_answer, explanation,
             blanks_json, correct_answers_json, expected_value, tolerance)
        SELECT question_id, subject, topic, question, question_type, difficulty, difficulty_rating,
               option_a, option_b, option_c, option_d, correct_answer, model_answer, explanation,
               blanks_json, correct_answers_json, expected_value, tolerance
        FROM questions;
        DROP TABLE questions;
        ALTER TABLE questions_new RENAME TO questions;
    """)
    conn.execute("PRAGMA foreign_keys = ON")


def initialise_database(db_path: Path = DATABASE_PATH) -> None:
    with connection(db_path) as conn:
        conn.executescript(SCHEMA)
        # Safe migrations keep early research databases usable as the prototype evolves.
        student_columns = {row[1] for row in conn.execute("PRAGMA table_info(students)")}
        attempt_columns = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
        question_columns = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
        if "password_hash" not in student_columns:
            conn.execute("ALTER TABLE students ADD COLUMN password_hash TEXT")
        if "email" not in student_columns:
            conn.execute("ALTER TABLE students ADD COLUMN email TEXT")
        if "confidence_rating" not in attempt_columns:
            conn.execute("ALTER TABLE attempts ADD COLUMN confidence_rating INTEGER CHECK(confidence_rating BETWEEN 1 AND 5)")
        if "answer_text" not in attempt_columns:
            conn.execute("ALTER TABLE attempts ADD COLUMN answer_text TEXT")
        if "score" not in attempt_columns:
            conn.execute("ALTER TABLE attempts ADD COLUMN score REAL CHECK(score BETWEEN 0.0 AND 1.0)")
        if "question_type" not in question_columns:
            _rebuild_questions_table(conn)
        if "difficulty_rating" not in {row[1] for row in conn.execute("PRAGMA table_info(questions)")}:
            conn.execute("ALTER TABLE questions ADD COLUMN difficulty_rating REAL NOT NULL DEFAULT 0.55")
        # Schema version bookkeeping. Every record the new table sees has
        # a description; the test suite can inspect this to know which
        # migration set is active.
        current = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
        if current < 1:
            conn.execute(
                "INSERT INTO schema_version(version, applied_at, description) VALUES (?,?,?)",
                (1, datetime.now(timezone.utc).isoformat(), "initial schema + early migrations"),
            )
        if current < 2:
            conn.execute(
                "INSERT INTO schema_version(version, applied_at, description) VALUES (?,?,?)",
                (2, datetime.now(timezone.utc).isoformat(), "student_topic_mastery + schema_version table"),
            )
        if current < 3:
            conn.execute(
                "INSERT INTO schema_version(version, applied_at, description) VALUES (?,?,?)",
                (3, datetime.now(timezone.utc).isoformat(), "attempt_feedback table for 👍/👎"),
            )
        if current < 4:
            conn.execute(
                "INSERT INTO schema_version(version, applied_at, description) VALUES (?,?,?)",
                (4, datetime.now(timezone.utc).isoformat(), "expanded question_type CHECK + blanks/answers/value/tolerance columns"),
            )
        # Add the new question columns and relax the CHECK for older DBs.
        question_columns_now = {row[1] for row in conn.execute("PRAGMA table_info(questions)")}
        if "blanks_json" not in question_columns_now:
            conn.execute("ALTER TABLE questions ADD COLUMN blanks_json TEXT")
        if "correct_answers_json" not in question_columns_now:
            conn.execute("ALTER TABLE questions ADD COLUMN correct_answers_json TEXT")
        if "expected_value" not in question_columns_now:
            conn.execute("ALTER TABLE questions ADD COLUMN expected_value REAL")
        if "tolerance" not in question_columns_now:
            conn.execute("ALTER TABLE questions ADD COLUMN tolerance REAL DEFAULT 0.01")
        # Old DBs have the strict MCQ/Subjective CHECK. Rebuild the table
        # to relax it so the new types can be inserted.
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='questions'"
        ):
            sql = row[0] or ""
            if "TrueFalse" not in sql:
                _relax_questions_check(conn)


def seed_question_bank(questions: Iterable[dict], db_path: Path = DATABASE_PATH) -> None:
    """Upsert modular question records and their subject/topic lookup records."""
    initialise_database(db_path)
    rows = []
    for q in questions:
        row = dict(q)
        # New question-type columns: default to None / 0.01 so older MCQ-only
        # records still bind cleanly after the schema migration.
        row.setdefault("blanks_json", None)
        row.setdefault("correct_answers_json", None)
        row.setdefault("expected_value", None)
        row.setdefault("tolerance", 0.01)
        rows.append(row)
    with connection(db_path) as conn:
        conn.executemany("INSERT OR IGNORE INTO subjects(name) VALUES (?)", [(q["subject"],) for q in rows])
        conn.executemany(
            "INSERT OR IGNORE INTO topics(subject, name) VALUES (?, ?)",
            [(q["subject"], q["topic"]) for q in rows],
        )
        conn.executemany(
            """INSERT OR REPLACE INTO questions
               (question_id, subject, topic, question, question_type, difficulty, difficulty_rating,
                option_a, option_b, option_c, option_d, correct_answer, model_answer, explanation,
                blanks_json, correct_answers_json, expected_value, tolerance)
               VALUES (:question_id, :subject, :topic, :question, :question_type, :difficulty,
                :difficulty_rating, :option_a, :option_b, :option_c, :option_d, :correct_answer,
                :model_answer, :explanation, :blanks_json, :correct_answers_json, :expected_value,
                :tolerance)""",
            rows,
        )


def enqueue_questions(
    student_id: str,
    items: Iterable[dict],
    due_at: str,
    reason: str = "",
    db_path: Path = DATABASE_PATH,
) -> int:
    """Queue (subject, topic, question_id) records for future practice; skip already-pending items."""
    initialise_database(db_path)
    queued = 0
    with connection(db_path) as conn:
        pending = {
            row["question_id"]
            for row in conn.execute(
                "SELECT question_id FROM practice_queue WHERE student_id=? AND status='pending'", (student_id,)
            )
        }
        for item in items:
            if item["question_id"] in pending:
                continue
            conn.execute(
                """INSERT INTO practice_queue(student_id, question_id, subject, topic, status, reason, queued_at, due_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (student_id, item["question_id"], item["subject"], item["topic"], reason,
                 datetime.now(timezone.utc).isoformat(), due_at),
            )
            pending.add(item["question_id"])
            queued += 1
    return queued


def due_queue_items(
    student_id: str,
    now_iso: str,
    subject: str | None = None,
    limit: int = 1,
    db_path: Path = DATABASE_PATH,
) -> list[sqlite3.Row]:
    query = """SELECT * FROM practice_queue
               WHERE student_id=? AND status='pending' AND due_at<=?
               ORDER BY due_at, queue_id LIMIT ?"""
    params: tuple = (student_id, now_iso, limit)
    if subject is not None:
        query = query.replace("AND due_at<=?", "AND due_at<=? AND subject=?")
        params = (student_id, now_iso, subject, limit)
    with connection(db_path) as conn:
        return list(conn.execute(query, params).fetchall())


def pending_queue_items(student_id: str, db_path: Path = DATABASE_PATH) -> list[sqlite3.Row]:
    with connection(db_path) as conn:
        return list(
            conn.execute(
                "SELECT * FROM practice_queue WHERE student_id=? AND status='pending' ORDER BY due_at, queue_id",
                (student_id,),
            ).fetchall()
        )


def complete_queue_item(student_id: str, question_id: str, db_path: Path = DATABASE_PATH) -> None:
    with connection(db_path) as conn:
        conn.execute(
            "UPDATE practice_queue SET status='completed' WHERE student_id=? AND question_id=? AND status='pending'",
            (student_id, question_id),
        )


def record_feedback(
    attempt_id: int, student_id: str, useful: bool, db_path: Path = DATABASE_PATH
) -> int:
    """Record 👍/👎 feedback for a specific attempt. Idempotent per attempt."""
    with connection(db_path) as conn:
        existing = conn.execute(
            "SELECT feedback_id FROM attempt_feedback WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE attempt_feedback SET useful=? WHERE attempt_id=?",
                (1 if useful else 0, attempt_id),
            )
            return int(existing["feedback_id"])
        cur = conn.execute(
            "INSERT INTO attempt_feedback(attempt_id, student_id, useful, created_at) VALUES (?,?,?,?)",
            (attempt_id, student_id, 1 if useful else 0, datetime.now(timezone.utc).isoformat()),
        )
        return int(cur.lastrowid)


def feedback_summary(student_id: str, db_path: Path = DATABASE_PATH) -> dict:
    with connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, SUM(useful) AS useful FROM attempt_feedback WHERE student_id=?",
            (student_id,),
        ).fetchone()
    return {
        "total_feedback": int(row["total"] or 0),
        "useful": int(row["useful"] or 0),
        "rate": (float(row["useful"]) / float(row["total"])) if row["total"] else None,
    }


def write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_run(run_id: str, created_at: str, config: dict, db_path: Path = DATABASE_PATH) -> None:
    with connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO simulation_runs(run_id, created_at, config_json) VALUES (?, ?, ?)",
            (run_id, created_at, json.dumps(config, sort_keys=True)),
        )
