"""SQLite schema. Foreign keys provide a clean, inspectable research dataset."""

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    email TEXT,
    password_hash TEXT,
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK(is_synthetic IN (0, 1)),
    created_at TEXT NOT NULL,
    latent_profile_json TEXT
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE(subject, name)
);

CREATE TABLE IF NOT EXISTS questions (
    question_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    topic TEXT NOT NULL,
    question TEXT NOT NULL,
    question_type TEXT NOT NULL DEFAULT 'MCQ' CHECK(question_type IN ('MCQ', 'Subjective')),
    difficulty TEXT NOT NULL CHECK(difficulty IN ('Easy', 'Medium', 'Hard')),
    difficulty_rating REAL NOT NULL CHECK(difficulty_rating BETWEEN 0.1 AND 1.0),
    option_a TEXT,
    option_b TEXT,
    option_c TEXT,
    option_d TEXT,
    correct_answer TEXT,
    model_answer TEXT,
    explanation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    question_id TEXT NOT NULL REFERENCES questions(question_id),
    subject TEXT NOT NULL,
    topic TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
    time_taken_seconds REAL NOT NULL,
    attempt_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    confidence_rating INTEGER CHECK(confidence_rating BETWEEN 1 AND 5),
    is_synthetic INTEGER NOT NULL DEFAULT 0 CHECK(is_synthetic IN (0, 1)),
    answer_text TEXT,
    score REAL CHECK(score BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS idx_attempts_student_time ON attempts(student_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_attempts_student_topic ON attempts(student_id, subject, topic);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT 'SYNTHETIC / SIMULATED DATA'
);

CREATE TABLE IF NOT EXISTS mastery_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    subject TEXT NOT NULL,
    topic TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    status TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_at TEXT NOT NULL,
    explanation_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    question_id TEXT NOT NULL REFERENCES questions(question_id),
    subject TEXT NOT NULL,
    topic TEXT NOT NULL,
    recommended_difficulty TEXT NOT NULL,
    reason TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS practice_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(student_id),
    question_id TEXT NOT NULL REFERENCES questions(question_id),
    subject TEXT NOT NULL,
    topic TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'completed')),
    reason TEXT NOT NULL DEFAULT '',
    queued_at TEXT NOT NULL,
    due_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_queue_student_due ON practice_queue(student_id, status, due_at);
"""
