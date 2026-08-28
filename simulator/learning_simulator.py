"""High-level reproducible simulation run and CSV/SQLite persistence."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import GENERATED_DIR, DATABASE_PATH, SimulationConfig
from data.questions.question_bank import build_question_bank, export_question_bank
from database.db import connection, initialise_database, save_run, seed_question_bank, write_dataframe
from simulator.attempt_generator import generate_attempts
from simulator.student_generator import generate_students


def run_simulation(config: SimulationConfig, output_dir: Path = GENERATED_DIR, db_path: Path = DATABASE_PATH) -> dict:
    """Create a labelled synthetic run. The supplied seed makes it reproducible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = pd.DataFrame(build_question_bank())
    students, profiles = generate_students(config.number_of_students, config.random_seed, config.ability_distribution)
    attempts, truth = generate_attempts(
        students, profiles, questions, config.attempts_per_student, config.learning_rate,
        config.noise_level, config.difficulty_distribution, config.random_seed, config.start_date,
    )
    run_id = f"synthetic_{uuid.uuid4().hex[:10]}"
    manifest = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "label": "SYNTHETIC / SIMULATED DATA — NOT REAL STUDENT RESEARCH DATA",
        "config": config.to_dict(),
        "row_counts": {"students": len(students), "questions": len(questions), "attempts": len(attempts)},
    }
    paths = {
        "students": write_dataframe(students, output_dir / "students.csv"),
        "questions": write_dataframe(questions, output_dir / "questions.csv"),
        "attempts": write_dataframe(attempts, output_dir / "attempts.csv"),
        "synthetic_truth": write_dataframe(truth, output_dir / "synthetic_truth.csv"),
    }
    (output_dir / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    export_question_bank(output_dir.parent / "questions" / "question_bank.json")
    initialise_database(db_path)
    seed_question_bank(questions.to_dict("records"), db_path)
    with connection(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO students(student_id, display_name, is_synthetic, created_at, latent_profile_json)
               VALUES (?, ?, ?, ?, ?)""",
            [tuple(r) for r in students[["student_id", "display_name", "is_synthetic", "created_at", "latent_profile_json"]].itertuples(index=False, name=None)],
        )
        conn.executemany(
            """INSERT INTO attempts(student_id, question_id, subject, topic, difficulty, is_correct,
               time_taken_seconds, attempt_number, timestamp, session_id, is_synthetic)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [tuple(r) for r in attempts[["student_id", "question_id", "subject", "topic", "difficulty", "is_correct", "time_taken_seconds", "attempt_number", "timestamp", "session_id", "is_synthetic"]].itertuples(index=False, name=None)],
        )
    save_run(run_id, manifest["created_at"], config.to_dict(), db_path)
    return {"run_id": run_id, "students": students, "questions": questions, "attempts": attempts, "truth": truth, "paths": paths, "manifest": manifest}


if __name__ == "__main__":
    result = run_simulation(SimulationConfig())
    print(json.dumps(result["manifest"], indent=2))
