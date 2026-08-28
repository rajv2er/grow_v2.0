"""Database creation, seeding and export helpers."""
from __future__ import annotations

import json
import sqlite3
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


def initialise_database(db_path: Path = DATABASE_PATH) -> None:
    with connection(db_path) as conn:
        conn.executescript(SCHEMA)
        # Safe migrations keep early research databases usable as the prototype evolves.
        student_columns = {row[1] for row in conn.execute("PRAGMA table_info(students)")}
        attempt_columns = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
        if "password_hash" not in student_columns:
            conn.execute("ALTER TABLE students ADD COLUMN password_hash TEXT")
        if "confidence_rating" not in attempt_columns:
            conn.execute("ALTER TABLE attempts ADD COLUMN confidence_rating INTEGER CHECK(confidence_rating BETWEEN 1 AND 5)")


def seed_question_bank(questions: Iterable[dict], db_path: Path = DATABASE_PATH) -> None:
    """Upsert modular question records and their subject/topic lookup records."""
    initialise_database(db_path)
    rows = list(questions)
    with connection(db_path) as conn:
        conn.executemany("INSERT OR IGNORE INTO subjects(name) VALUES (?)", [(q["subject"],) for q in rows])
        conn.executemany(
            "INSERT OR IGNORE INTO topics(subject, name) VALUES (?, ?)",
            [(q["subject"], q["topic"]) for q in rows],
        )
        conn.executemany(
            """INSERT OR REPLACE INTO questions
               (question_id, subject, topic, question, difficulty, option_a, option_b,
                option_c, option_d, correct_answer, explanation)
               VALUES (:question_id, :subject, :topic, :question, :difficulty, :option_a,
                :option_b, :option_c, :option_d, :correct_answer, :explanation)""",
            rows,
        )


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
