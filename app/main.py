"""MasteryLab — adaptive multi-subject learning research platform.

Streamlit front-end wired to the project's real SQLite database, simulator,
ML pipeline, recommender, and experiment runner. Every statistic rendered in
the UI is derived from the underlying dataset, predictions, or trained
model artefacts; no values are fabricated.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.reports import (
    dataset_statistics,
    learning_curve,
    learning_curve_frame,
    performance_by_subject_topic,
)
from config import GENERATED_DIR, MODELS_DIR, RESULTS_DIR, SimulationConfig
from data.questions.question_bank import build_question_bank
from data.topics.references import get_links as _get_topic_links
from database.db import (
    complete_queue_item,
    connection,
    due_queue_items,
    enqueue_questions,
    initialise_database,
    pending_queue_items,
    seed_question_bank,
)
from experiments.experiment_1 import run_experiment_from_data
from ml.predict import predict_student_mastery
from ml.online_mastery import (
    MIN_ATTEMPTS_FOR_OVERLAY as _EMA_MIN_ATTEMPTS,
    overlay_predictions as _overlay_mastery,
    update_after_attempt as _update_ema,
)
from ml.subjective_grading import grade_subjective
from recommendation.recommender import recommend_questions
from simulator.learning_simulator import run_simulation

NOTICE = "Synthetic records are simulated research data only — never real student-study evidence."
SUBJECTS = ["DSA", "DBMS", "Operating Systems", "Computer Networks", "Software Engineering"]
SUBJECT_SHORT = {
    "DSA": "DSA",
    "DBMS": "DBMS",
    "Operating Systems": "OS",
    "Computer Networks": "CN",
    "Software Engineering": "SE",
}
SUBJECT_MASTERY_THRESHOLD = 0.75
MIN_QUESTIONS_PER_SUBJECT = 5
TOPIC_HOLD_QUESTIONS = 3
SESSION_LENGTH = 5
TIME_LIMITS = {"Easy": 90, "Medium": 120, "Hard": 150}
SUBJECTIVE_TIME_LIMIT = 240
REVIEW_DELAY_HOURS = 24

STATUSES_FOR_PRIORITY = [("HIGH", 0.45), ("MEDIUM", 0.65), ("LOW", 1.01)]


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

PALETTE = {
    "bg": "#0b1020",
    "surface": "#111a2e",
    "surface_alt": "#16213b",
    "border": "#1f2a44",
    "border_strong": "#2b3a5e",
    "text": "#e6ecf7",
    "text_muted": "#9aa6c0",
    "accent": "#6366f1",
    "accent_soft": "#818cf8",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --ml-bg: {PALETTE['bg']};
            --ml-surface: {PALETTE['surface']};
            --ml-surface-alt: {PALETTE['surface_alt']};
            --ml-border: {PALETTE['border']};
            --ml-border-strong: {PALETTE['border_strong']};
            --ml-text: {PALETTE['text']};
            --ml-muted: {PALETTE['text_muted']};
            --ml-accent: {PALETTE['accent']};
            --ml-accent-soft: {PALETTE['accent_soft']};
            --ml-success: {PALETTE['success']};
            --ml-warning: {PALETTE['warning']};
            --ml-danger: {PALETTE['danger']};
        }}
        html, body, [class*="css"]  {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        .stApp {{
            background: linear-gradient(180deg, #0a0f1f 0%, #0b1020 100%);
            color: var(--ml-text);
        }}
        [data-testid="stSidebar"] {{
            background: #0a1024;
            border-right: 1px solid var(--ml-border);
        }}
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {{ color: var(--ml-text); }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        h1, h2, h3, h4 {{ color: var(--ml-text); letter-spacing: -0.01em; font-weight: 650; }}
        .ml-card {{
            background: var(--ml-surface);
            border: 1px solid var(--ml-border);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 12px;
        }}
        .ml-card-elev {{
            background: linear-gradient(180deg, #131d36 0%, #0f182d 100%);
            border: 1px solid var(--ml-border-strong);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 1px 0 rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.25);
        }}
        .ml-section-label {{
            color: var(--ml-muted);
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            font-weight: 600;
            text-transform: uppercase;
            margin: 18px 0 8px 0;
        }}
        .ml-kpi {{
            background: var(--ml-surface);
            border: 1px solid var(--ml-border);
            border-radius: 14px;
            padding: 16px 18px;
        }}
        .ml-kpi .label {{ color: var(--ml-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }}
        .ml-kpi .value {{ color: var(--ml-text); font-size: 1.7rem; font-weight: 700; margin-top: 4px; }}
        .ml-kpi .delta {{ font-size: 0.8rem; margin-top: 2px; }}
        .ml-subject-card {{
            background: var(--ml-surface);
            border: 1px solid var(--ml-border);
            border-radius: 14px;
            padding: 16px;
            transition: border-color 0.15s ease, transform 0.15s ease;
        }}
        .ml-subject-card:hover {{ border-color: var(--ml-accent-soft); transform: translateY(-1px); }}
        .ml-subject-card .name {{ color: var(--ml-muted); font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; }}
        .ml-subject-card .pct {{ color: var(--ml-text); font-size: 1.6rem; font-weight: 700; }}
        .ml-progress {{ height: 6px; background: #1a2540; border-radius: 4px; overflow: hidden; margin-top: 8px; }}
        .ml-progress > div {{ height: 100%; background: linear-gradient(90deg, var(--ml-accent), var(--ml-accent-soft)); border-radius: 4px; }}
        .ml-pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }}
        .ml-pill.up {{ color: #4ade80; background: rgba(34,197,94,0.12); }}
        .ml-pill.down {{ color: #f87171; background: rgba(239,68,68,0.12); }}
        .ml-pill.flat {{ color: var(--ml-muted); background: rgba(154,166,192,0.12); }}
        .ml-pill.synthetic {{ color: #fcd34d; background: rgba(245,158,11,0.15); }}
        .ml-pill.weak {{ color: #f87171; background: rgba(239,68,68,0.15); }}
        .ml-pill.dev {{ color: #fbbf24; background: rgba(245,158,11,0.15); }}
        .ml-pill.strong {{ color: #4ade80; background: rgba(34,197,94,0.15); }}
        .ml-empty {{
            text-align: center; color: var(--ml-muted); padding: 40px 20px;
            background: var(--ml-surface); border: 1px dashed var(--ml-border-strong);
            border-radius: 14px;
        }}
        .ml-banner {{
            padding: 10px 14px; border-radius: 10px; font-size: 0.85rem;
            background: rgba(245,158,11,0.10); color: #fcd34d; border: 1px solid rgba(245,158,11,0.25);
            margin-bottom: 12px;
        }}
        .ml-sidebar-brand {{ font-size: 1.15rem; font-weight: 700; color: var(--ml-text); margin: 0; }}
        .ml-sidebar-tag {{ font-size: 0.7rem; color: var(--ml-muted); letter-spacing: 0.12em; text-transform: uppercase; }}
        .ml-profile-card {{
            background: var(--ml-surface-alt); border: 1px solid var(--ml-border);
            border-radius: 12px; padding: 12px 14px; margin-top: auto;
        }}
        .ml-profile-card .name {{ color: var(--ml-text); font-weight: 600; font-size: 0.95rem; }}
        .ml-profile-card .meta {{ color: var(--ml-muted); font-size: 0.75rem; }}
        .stButton>button, .stDownloadButton>button {{
            border-radius: 10px; font-weight: 600; border: 1px solid var(--ml-border-strong);
            background: var(--ml-surface-alt); color: var(--ml-text);
        }}
        .stButton>button:hover {{ border-color: var(--ml-accent-soft); color: var(--ml-text); }}
        .stButton>button[kind="primary"] {{
            background: linear-gradient(135deg, #6366f1, #4f46e5); border: 1px solid #4f46e5; color: white;
        }}
        .stButton>button[kind="primary"]:hover {{ filter: brightness(1.08); color: white; }}
        [data-testid="stMetricValue"] {{ color: var(--ml-text); }}
        .stProgress > div > div > div > div {{ background: var(--ml-accent); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f"<div class='ml-section-label'>{text}</div>", unsafe_allow_html=True)


def card(content: str, elevated: bool = False) -> None:
    cls = "ml-card-elev" if elevated else "ml-card"
    st.markdown(f"<div class='{cls}'>{content}</div>", unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"<div class='ml-empty'><div style='font-size:1.05rem;color:#e6ecf7;margin-bottom:4px'>{title}</div>"
        f"<div>{body}</div></div>",
        unsafe_allow_html=True,
    )


def info_banner(text: str) -> None:
    st.markdown(f"<div class='ml-banner'>{text}</div>", unsafe_allow_html=True)


def pill(label: str, kind: str = "flat") -> str:
    return f"<span class='ml-pill {kind}'>{label}</span>"


def kpi(label: str, value: str, delta: str | None = None, delta_kind: str = "flat") -> str:
    delta_html = f"<div class='delta'>{pill(delta, delta_kind)}</div>" if delta else ""
    return (
        f"<div class='ml-kpi'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>{delta_html}</div>"
    )


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def password_hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 180000).hex()
    return f"{salt}${digest}"


def password_ok(password: str, stored: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    return hmac.compare_digest(password_hash(password, salt).split("$", 1)[1], digest)


def create_user(name: str, email: str, password: str) -> str:
    with connection() as conn:
        no = conn.execute("SELECT COUNT(*) FROM students WHERE is_synthetic=0").fetchone()[0] + 1
        sid = f"U{no:05d}"
        conn.execute(
            "INSERT INTO students(student_id, display_name, email, password_hash, is_synthetic, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                sid,
                name.strip(),
                email.strip() or None,
                password_hash(password),
                0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return sid


def log_in(sid: str, password: str) -> bool:
    with connection() as conn:
        row = conn.execute(
            "SELECT display_name, password_hash, is_synthetic FROM students WHERE student_id=?",
            (sid,),
        ).fetchone()
    if not row or row["is_synthetic"] or not password_ok(password, row["password_hash"]):
        return False
    st.session_state.user_id = sid
    st.session_state.user_name = row["display_name"]
    st.session_state.login_at = datetime.now(timezone.utc).isoformat()
    return True


def session_valid(ttl_hours: int = 24) -> bool:
    """True if the current session is still within the TTL window."""
    if not st.session_state.get("user_id"):
        return False
    login_at = st.session_state.get("login_at")
    if not login_at:
        return True
    try:
        stamp = datetime.fromisoformat(login_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    age = datetime.now(timezone.utc) - stamp
    if age > timedelta(hours=ttl_hours):
        st.session_state.pop("user_id", None)
        st.session_state.pop("user_name", None)
        st.session_state.pop("login_at", None)
        return False
    return True


# ---------------------------------------------------------------------------
# Data accessors
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def questions_df() -> pd.DataFrame:
    return pd.DataFrame(build_question_bank())


@st.cache_data(show_spinner=False)
def generated_attempts_df() -> pd.DataFrame:
    p = GENERATED_DIR / "attempts.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def generated_truth_df() -> pd.DataFrame:
    p = GENERATED_DIR / "synthetic_truth.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def experiment_results() -> dict[str, pd.DataFrame | bool]:
    files = {
        "test": RESULTS_DIR / "test_model_comparison.csv",
        "validation": RESULTS_DIR / "validation_model_comparison.csv",
        "recommendation": RESULTS_DIR / "recommendation_policy_comparison.csv",
        "subject_perf": RESULTS_DIR / "subject_performance.csv",
        "topic_perf": RESULTS_DIR / "topic_performance.csv",
    }
    out: dict[str, pd.DataFrame | bool] = {}
    for key, path in files.items():
        out[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    out["trained"] = (MODELS_DIR / "xgboost.joblib").exists() and (
        RESULTS_DIR / "test_model_comparison.csv"
    ).exists()
    return out


def user_attempts(student_id: str) -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM attempts WHERE student_id=? ORDER BY timestamp",
            conn,
            params=(student_id,),
        )


def user_predictions(student_id: str) -> pd.DataFrame:
    with connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM mastery_predictions WHERE student_id=? ORDER BY mastery_probability",
            conn,
            params=(student_id,),
        )


def model_path() -> Path | None:
    metric = RESULTS_DIR / "validation_model_comparison.csv"
    if metric.exists():
        best = pd.read_csv(metric).iloc[0]["model"]
        p = MODELS_DIR / f"{best}.joblib"
        if p.exists():
            return p
    paths = sorted(MODELS_DIR.glob("*.joblib"))
    return paths[0] if paths else None


def predictions_for(student_id: str, attempts: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    m = model_path()
    if m is None or attempts.empty:
        return pd.DataFrame()
    global_pred = predict_student_mastery(m, attempts, q, student_id)
    return _overlay_mastery(global_pred, student_id)


# ---------------------------------------------------------------------------
# Quiz state
# ---------------------------------------------------------------------------

def initialise_quiz_subject(sid: str) -> str:
    if st.session_state.get("quiz_student_id") != sid:
        for key in (
            "active_question",
            "why",
            "started",
            "answered",
            "quiz_topic",
            "quiz_topic_served",
            "mastery_before",
            "session",
        ):
            st.session_state.pop(key, None)
        st.session_state.quiz_student_id = sid
        st.session_state.current_subject = SUBJECTS[0]
        st.session_state.subject_manually_selected = False
        st.session_state.quiz_subject_picker_subject = SUBJECTS[0]
        st.session_state.pop("quiz_subject_picker", None)
    return st.session_state.current_subject


def _clear_question_state() -> None:
    for key in (
        "active_question",
        "why",
        "started",
        "answered",
        "result",
        "mastery_before",
        "quiz_topic",
        "quiz_topic_served",
    ):
        st.session_state.pop(key, None)


def _apply_topic_prefill(q: pd.DataFrame, attempts: pd.DataFrame) -> None:
    """If the user clicked 'Start session on this topic' elsewhere, honour it.

    Picks the first unseen Easy MCQ in the requested (subject, topic) and
    starts a session there. Falls back to the recommender's adaptive pick if
    the prefill target is already exhausted.
    """
    target = st.session_state.pop("_prefill_topic", None)
    if target is None:
        return
    subject, topic = target
    if subject not in SUBJECTS:
        return
    _start_session(subject)
    st.session_state.subject_manually_selected = True
    st.session_state.current_subject = subject
    st.session_state.quiz_subject_picker_subject = subject
    st.session_state.pop("quiz_subject_picker", None)
    st.session_state.quiz_topic = topic
    st.session_state.quiz_topic_served = 0
    seen = set(attempts.question_id) if not attempts.empty else set()
    candidates = q[
        (q.subject == subject) & (q.topic == topic) & (q.difficulty == "Easy")
    ].sort_values("question_id")
    if not candidates.empty:
        unseen = candidates[~candidates.question_id.isin(seen)]
        pick = unseen.iloc[0] if not unseen.empty else candidates.iloc[0]
        st.session_state.active_question = pick.question_id
        st.session_state.last_served_id = pick.question_id
        st.session_state.why = (
            f"Pre-selected from your Recommendations reference view: focus on {topic} "
            f"({subject}) so the difficulty ladder receives consecutive same-topic evidence."
        )
        st.session_state.started = time.monotonic()
        st.session_state.answered = False
        st.session_state.mastery_before = None
        st.session_state.pop("result", None)


def _start_session(subject: str) -> None:
    st.session_state.session = {"subject": subject, "answered": 0, "correct": 0, "items": []}
    st.session_state.current_subject = subject
    st.session_state.subject_manually_selected = True
    st.session_state.quiz_subject_picker_subject = subject
    _clear_question_state()


def release_topic_hold() -> None:
    st.session_state.quiz_topic = None
    st.session_state.quiz_topic_served = 0


def _topic_mastery(p: pd.DataFrame, subject: str, topic: str) -> float | None:
    if p is None or p.empty:
        return None
    rows = p[(p.subject == subject) & (p.topic == topic)]
    return float(rows.mastery_probability.iloc[0]) if not rows.empty else None


def initial_diagnostic(q: pd.DataFrame, attempts: pd.DataFrame, subject: str) -> pd.Series:
    seen = set(attempts.question_id) if not attempts.empty else set()
    candidates = q[
        (q.subject == subject) & (q.difficulty == "Easy") & (q.question_type == "MCQ")
    ].sort_values("question_id")
    unseen = candidates[~candidates.question_id.isin(seen)]
    return (unseen if not unseen.empty else candidates).iloc[0]


def subject_ready_to_advance(attempts: pd.DataFrame, subject: str, subject_predictions: pd.DataFrame) -> bool:
    subject_attempts = attempts[attempts.subject == subject]
    mastered = (
        not subject_predictions.empty
        and subject_predictions.mastery_probability.ge(SUBJECT_MASTERY_THRESHOLD).all()
    )
    return len(subject_attempts) >= MIN_QUESTIONS_PER_SUBJECT or mastered


def save_answer(
    sid: str,
    question: pd.Series,
    correct: bool,
    seconds: float,
    confidence: int,
    answer_text: str | None = None,
    score: float | None = None,
) -> int:
    with connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM attempts WHERE student_id=?", (sid,)).fetchone()[0] + 1
        cur = conn.execute(
            "INSERT INTO attempts(student_id, question_id, subject, topic, difficulty, is_correct, "
            "time_taken_seconds, attempt_number, timestamp, session_id, confidence_rating, is_synthetic, "
            "answer_text, score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                question.question_id,
                question.subject,
                question.topic,
                question.difficulty,
                int(correct),
                max(seconds, 1),
                n,
                datetime.now(timezone.utc).isoformat(),
                f"{sid}_PRACTICE",
                confidence,
                0,
                answer_text,
                score,
            ),
        )
        attempt_id = int(cur.lastrowid)
    complete_queue_item(sid, question.question_id)
    if not correct:
        review_at = (datetime.now(timezone.utc) + timedelta(hours=REVIEW_DELAY_HOURS)).isoformat()
        enqueue_questions(
            sid,
            [{"question_id": question.question_id, "subject": question.subject, "topic": question.topic}],
            review_at,
            reason="Scheduled review — previously answered incorrectly",
        )
    # Online mastery update: shift the per-(student, subject, topic) EMA
    # toward the new answer. First-time rows are initialised from the global
    # model's prediction so the EMA starts at a learned prior, not 0.5.
    prior = _prior_mastery(sid, question.subject, question.topic)
    _update_ema(
        sid, question.subject, question.topic, int(bool(correct)), prior_mastery=prior
    )
    return attempt_id


def _prior_mastery(sid: str, subject: str, topic: str) -> float | None:
    """Best available mastery prior from the global model for a single topic."""
    model = model_path()
    if model is None:
        return None
    attempts = user_attempts(sid)
    if attempts.empty:
        return None
    p = predict_student_mastery(model, attempts, questions_df(), sid)
    if p.empty:
        return None
    rows = p[(p.subject == subject) & (p.topic == topic)]
    if rows.empty:
        return None
    return float(rows.mastery_probability.iloc[0])


def next_question(
    sid: str, attempts: pd.DataFrame, q: pd.DataFrame
) -> tuple[pd.Series, str, float | None]:
    """Pick the next item: due practice-queue entries first, then adaptive selection."""
    p = predictions_for(sid, attempts, q)
    subject = initialise_quiz_subject(sid)
    now_iso = datetime.now(timezone.utc).isoformat()
    queued = due_queue_items(
        sid, now_iso, subject=subject if st.session_state.get("subject_manually_selected") else None
    )
    if queued:
        row = queued[0]
        item = q[q.question_id == row["question_id"]].iloc[0]
        if item.subject != subject:
            st.session_state.current_subject = item.subject
            subject = item.subject
        release_topic_hold()
        before = _topic_mastery(p, item.subject, item.topic)
        note = f" ({row['reason']})" if row["reason"] else ""
        return item, f"Scheduled future practice{note}.", before

    if int(st.session_state.get("quiz_topic_served", 0)) >= TOPIC_HOLD_QUESTIONS:
        release_topic_hold()

    subject_predictions = p[p.subject == subject] if not p.empty else pd.DataFrame()
    subject_index = SUBJECTS.index(subject)
    if (
        not st.session_state.get("subject_manually_selected")
        and subject_index < len(SUBJECTS) - 1
        and subject_ready_to_advance(attempts, subject, subject_predictions)
    ):
        subject = SUBJECTS[subject_index + 1]
        st.session_state.current_subject = subject
        subject_predictions = p[p.subject == subject] if not p.empty else pd.DataFrame()
        release_topic_hold()

    held = st.session_state.get("quiz_topic")
    if held:
        held_rows = subject_predictions[subject_predictions.topic == held]
        if held_rows.empty:
            release_topic_hold()
            held = None

    if subject_predictions.empty:
        item = initial_diagnostic(q, attempts, subject)
        why = f"{subject} diagnostic: establish initial evidence before personalised recommendations."
        return item, why, _topic_mastery(p, subject, item.topic)

    exclude = (
        {st.session_state["last_served_id"]} if st.session_state.get("last_served_id") else None
    )
    recs = recommend_questions(
        sid, held_rows if held else subject_predictions, attempts, q, limit=1, exclude=exclude
    )
    if recs.empty:
        item = initial_diagnostic(q, attempts, subject)
        return item, f"{subject} diagnostic revision question.", _topic_mastery(p, subject, item.topic)

    rec = recs.iloc[0]
    item = q[q.question_id == rec.question_id].iloc[0]
    served = int(st.session_state.get("quiz_topic_served", 0))
    if item.topic != st.session_state.get("quiz_topic"):
        st.session_state.quiz_topic = item.topic
        st.session_state.quiz_topic_served = 1
        why = (
            f"{rec.reason} Focusing on {item.topic} for the next {TOPIC_HOLD_QUESTIONS} questions "
            "so difficulty can adapt."
        )
    else:
        st.session_state.quiz_topic_served = served + 1
        why = (
            f"{rec.reason} Continuing {item.topic} (question {served + 1} of {TOPIC_HOLD_QUESTIONS}) "
            "to let difficulty adapt."
        )
    return item, why, float(rec.mastery_probability)


# ---------------------------------------------------------------------------
# Demo student seeding
# ---------------------------------------------------------------------------

DEMO_STUDENT_ID = "U00003"
DEMO_WARMUP_TARGET = 30


def ensure_demo_warmup(student_id: str) -> int:
    """Seed a real user's history with up to DEMO_WARMUP_TARGET simulator attempts.

    Synthetic attempts are written only when the user has fewer warm-up records
    than the target; idempotent across re-runs.
    """
    with connection() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM attempts WHERE student_id=? AND is_synthetic=1",
            (student_id,),
        ).fetchone()[0]
    if existing >= DEMO_WARMUP_TARGET:
        return 0
    needed = DEMO_WARMUP_TARGET - existing
    cfg = SimulationConfig(
        number_of_students=1,
        attempts_per_student=needed,
        random_seed=20250828,
        learning_rate=0.04,
        noise_level=0.08,
    )
    result = run_simulation(cfg)
    student_df = result["students"].iloc[0:1]
    sim_id = str(student_df["student_id"].iloc[0])
    attempts_df = result["attempts"][result["attempts"]["student_id"] == sim_id].copy()
    with connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO students(student_id, display_name, is_synthetic, created_at) "
            "VALUES(?,?,?,?)",
            (
                f"__demo_seed_{student_id}",
                f"Demo warmup for {student_id}",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.execute(
            "UPDATE students SET latent_profile_json=? WHERE student_id=?",
            (json.dumps(student_df.iloc[0].to_dict()), sim_id),
        )
        rows = []
        for r in attempts_df.itertuples(index=False):
            # Strip the timezone suffix so seeded timestamps parse the same way
            # as the existing synthetic history (no `+00:00`).
            ts = str(r.timestamp)
            if ts.endswith("+00:00"):
                ts = ts[:-6]
            rows.append(
                (
                    student_id,
                    r.question_id,
                    r.subject,
                    r.topic,
                    r.difficulty,
                    int(r.is_correct),
                    float(r.time_taken_seconds),
                    int(r.attempt_number),
                    ts,
                    f"{student_id}_DEMO",
                    3,
                    1,
                    None,
                    None,
                )
            )
        conn.executemany(
            "INSERT INTO attempts(student_id, question_id, subject, topic, difficulty, is_correct, "
            "time_taken_seconds, attempt_number, timestamp, session_id, confidence_rating, "
            "is_synthetic, answer_text, score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    user_attempts.clear() if hasattr(user_attempts, "clear") else None
    return len(rows)


# Late import to keep the section above readable
import json  # noqa: E402


# ---------------------------------------------------------------------------
# Page: Login / Profile
# ---------------------------------------------------------------------------

def login_page() -> None:
    st.markdown(
        """
        <div style='padding: 32px 0 16px 0;'>
            <div class='ml-sidebar-tag' style='color:#818cf8;'>MasteryLab</div>
            <h1 style='margin: 6px 0 0 0; font-size: 2.4rem;'>Personalized learning, calibrated by machine learning.</h1>
            <p style='color:#9aa6c0; max-width: 560px; margin-top: 8px;'>
                MasteryLab reads your timed performance history, predicts topic-level mastery, and serves the next
                question that will move the needle — instead of a random quiz.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("user_id"):
        with connection() as conn:
            row = conn.execute(
                "SELECT display_name, created_at, email FROM students WHERE student_id=?",
                (st.session_state.user_id,),
            ).fetchone()
        st.success(
            f"Signed in as **{row['display_name']}** · {st.session_state.user_id}"
            + (f" · {row['email']}" if row["email"] else "")
        )
        c1, c2 = st.columns(2)
        if c1.button("Open dashboard", type="primary", width="stretch"):
            st.session_state.page = "Dashboard"
            st.rerun()
        if c2.button("Sign out", width="stretch"):
            st.session_state.pop("user_id", None)
            st.session_state.pop("user_name", None)
            st.rerun()
        return

    left, right = st.columns(2, gap="large")
    with left:
        section_label("Create learner profile")
        with st.form("register", clear_on_submit=True):
            name = st.text_input("Full name", placeholder="e.g. Rahul Sharma")
            email = st.text_input("Email (optional)", placeholder="you@example.com")
            pw = st.text_input("Password", type="password", help="Minimum 6 characters.")
            pw2 = st.text_input("Confirm password", type="password")
            submit = st.form_submit_button("Create account", type="primary", width="stretch")
        if submit:
            if len(name.strip()) < 2:
                st.error("Please enter a name of at least 2 characters.")
            elif len(pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif pw != pw2:
                st.error("Passwords do not match.")
            else:
                sid = create_user(name, email, pw)
                st.session_state.user_id = sid
                st.session_state.user_name = name.strip()
                st.success(f"Account created — Student ID {sid}.")
                st.rerun()

    with right:
        section_label("Sign in")
        with st.form("login"):
            sid_in = st.text_input("Student ID", placeholder="U00001")
            pw_in = st.text_input("Password", type="password")
            submit_l = st.form_submit_button("Continue", type="primary", width="stretch")
        if submit_l:
            if log_in(sid_in.strip().upper(), pw_in):
                st.rerun()
            st.error("Invalid student ID or password.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        section_label("Try a populated demo")
        st.caption(
            "Loads a real learner account and seeds ~30 simulator-generated warm-up attempts so the "
            "dashboard, recommendations, and ML predictions are immediately meaningful."
        )
        if st.button("Use Demo Student", width="stretch"):
            with st.spinner("Seeding demo history from the simulator…"):
                n = ensure_demo_warmup(DEMO_STUDENT_ID)
            with connection() as conn:
                row = conn.execute(
                    "SELECT display_name FROM students WHERE student_id=?", (DEMO_STUDENT_ID,)
                ).fetchone()
            if row is None:
                st.error("Demo account is not provisioned. Run `python run_demo.py` once to create U00003.")
            else:
                st.session_state.user_id = DEMO_STUDENT_ID
                st.session_state.user_name = row["display_name"]
                if n:
                    st.toast(f"Seeded {n} warm-up attempts from the simulator.", icon="✨")
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def streak_days(attempts: pd.DataFrame) -> int:
    if attempts.empty:
        return 0
    days = pd.to_datetime(attempts.timestamp, format="ISO8601", utc=True).dt.date
    unique = sorted(set(days), reverse=True)
    today = datetime.now(timezone.utc).date()
    streak = 0
    cursor = today
    for d in unique:
        if d == cursor:
            streak += 1
            cursor = cursor - timedelta(days=1)
        elif d < cursor:
            break
    return streak


def _mastery_trend(attempts: pd.DataFrame, subject: str) -> float:
    sub = attempts[attempts.subject == subject].sort_values("timestamp")
    if len(sub) < 4:
        return 0.0
    mid = len(sub) // 2
    return float(sub.is_correct.iloc[mid:].mean() - sub.is_correct.iloc[:mid].mean())


def dashboard_page(q: pd.DataFrame) -> None:
    if not st.session_state.get("user_id"):
        st.info("Sign in from the Profile page to access your dashboard.")
        return
    sid = st.session_state.user_id
    name = st.session_state.get("user_name", "Learner")
    attempts = user_attempts(sid)

    st.markdown(
        f"<h2 style='margin-bottom:0'>Good day, {name.split()[0]}.</h2>"
        "<p style='color:#9aa6c0; margin-top:4px'>Here is your learning intelligence, "
        "calculated from your actual attempt history and the trained mastery model.</p>",
        unsafe_allow_html=True,
    )

    if attempts.empty:
        empty_state(
            "No attempts yet",
            "Open the Practice page and complete a session — your dashboard populates from real answers.",
        )
        return

    p = predictions_for(sid, attempts, q)
    if p.empty:
        info_banner("Mastery probabilities are not available yet. Train the baseline in the ML Experiments page.")

    # KPI row
    accuracy = float(attempts.is_correct.mean())
    overall_mastery = float(p.mastery_probability.mean()) if not p.empty else accuracy
    topics_mastered = (
        int((p.mastery_probability >= SUBJECT_MASTERY_THRESHOLD).sum()) if not p.empty else 0
    )
    total_topics = int(p.topic.nunique()) if not p.empty else 0
    streak = streak_days(attempts)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        kpi("Overall mastery", f"{overall_mastery:.0%}", "From ML predictions" if not p.empty else None),
        unsafe_allow_html=True,
    )
    c2.markdown(kpi("Questions attempted", f"{len(attempts):,}"), unsafe_allow_html=True)
    c3.markdown(kpi("Current streak", f"{streak} day{'s' if streak != 1 else ''}"), unsafe_allow_html=True)
    c4.markdown(
        kpi("Topics mastered", f"{topics_mastered}/{total_topics}", "≥ 75% mastery", "strong"),
        unsafe_allow_html=True,
    )

    # Subject mastery
    section_label("Subject mastery")
    if p.empty:
        st.caption("Train the ML baseline to see per-subject mastery.")
    subj_cols = st.columns(5)
    for col, subject in zip(subj_cols, SUBJECTS):
        subj_attempts = attempts[attempts.subject == subject]
        if p.empty:
            mastery = float(subj_attempts.is_correct.mean()) if not subj_attempts.empty else 0.0
            source = "Observed accuracy"
        else:
            mastery = float(p[p.subject == subject].mastery_probability.mean())
            source = "ML estimate"
        trend = _mastery_trend(attempts, subject)
        delta_text = (
            f"{'▲' if trend >= 0 else '▼'} {abs(trend):.0%} vs earlier" if abs(trend) > 0.01 else "stable"
        )
        delta_kind = "up" if trend > 0.01 else "down" if trend < -0.01 else "flat"
        with col:
            st.markdown(
                f"<div class='ml-subject-card'>"
                f"<div class='name'>{subject}</div>"
                f"<div class='pct'>{mastery:.0%}</div>"
                f"<div class='ml-progress'><div style='width:{mastery*100:.1f}%'></div></div>"
                f"<div style='margin-top:8px'>{pill(delta_text, delta_kind)} {pill(source, 'flat')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Weakest / Strongest
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_label("Learning intelligence")
    if p.empty:
        empty_state("Predictions not available", "Train the ML baseline in ML Experiments to see weakness analysis.")
    else:
        weakest = p.iloc[0]
        strongest = p.iloc[-1]
        col_w, col_s = st.columns(2)
        reasons = (
            weakest.explanation.get("reasons", [])
            if isinstance(weakest.explanation, dict)
            else []
        )
        with col_w:
            st.markdown(
                f"<div class='ml-card-elev'>"
                f"<div class='ml-section-label' style='margin-top:0'>Your weakest area</div>"
                f"<div style='font-size:1.15rem;font-weight:600'>{weakest.subject} → {weakest.topic}</div>"
                f"<div style='display:flex;align-items:center;gap:12px;margin:8px 0'>"
                f"<div style='font-size:1.6rem;font-weight:700'>{weakest.mastery_probability:.0%}</div>"
                f"{pill(weakest.status, 'weak')}</div>"
                f"<ul style='color:#9aa6c0;margin:0;padding-left:18px'>"
                + "".join(f"<li>{r}</li>" for r in reasons)
                + f"</ul></div>",
                unsafe_allow_html=True,
            )
            if st.button("Practise this topic", key="dash_practise_weak", type="primary"):
                st.session_state.page = "Practice"
                st.session_state._prefill_topic = (weakest.subject, weakest.topic)
                st.rerun()
        with col_s:
            st.markdown(
                f"<div class='ml-card-elev'>"
                f"<div class='ml-section-label' style='margin-top:0'>Your strongest area</div>"
                f"<div style='font-size:1.15rem;font-weight:600'>{strongest.subject} → {strongest.topic}</div>"
                f"<div style='display:flex;align-items:center;gap:12px;margin:8px 0'>"
                f"<div style='font-size:1.6rem;font-weight:700'>{strongest.mastery_probability:.0%}</div>"
                f"{pill(strongest.status, 'strong')}</div>"
                f"<div style='color:#9aa6c0'>Maintain with periodic revision questions.</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Mastery over time
    section_label("Recent performance")
    curve = learning_curve_frame(attempts) if not attempts.empty else pd.DataFrame()
    if curve.empty or curve.shape[0] < 2:
        st.caption("Mastery over time will appear after a few attempts.")
    else:
        st.line_chart(curve, height=240, width="stretch")

    # Today's plan
    section_label("Today's recommended plan")
    if p.empty:
        st.caption("Generate recommendations by completing a few attempts and ensuring the ML model is trained.")
    else:
        recs = recommend_questions(sid, p, attempts, q, limit=4)
        if recs.empty:
            st.caption("No recommendations available — try answering more questions.")
        else:
            for i, row in enumerate(recs.itertuples(index=False), 1):
                st.markdown(
                    f"<div class='ml-card'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px'>"
                    f"<div><div style='font-weight:600'>{i}. {row.topic}</div>"
                    f"<div style='color:#9aa6c0;font-size:0.85rem'>{row.subject} · {row.recommended_difficulty} · "
                    f"est. mastery {row.mastery_probability:.0%}</div></div>"
                    f"<div style='min-width:200px;text-align:right'>"
                    f"{pill(row.recommended_difficulty, 'flat')}</div></div>"
                    f"<div style='color:#9aa6c0;font-size:0.85rem;margin-top:6px'>{row.reason}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Page: Subjects
# ---------------------------------------------------------------------------

def subjects_page(q: pd.DataFrame) -> None:
    if not st.session_state.get("user_id"):
        st.info("Sign in to view subject mastery.")
        return
    sid = st.session_state.user_id
    attempts = user_attempts(sid)
    p = predictions_for(sid, attempts, q)

    st.markdown("<h2 style='margin-bottom:0'>Subjects</h2>"
                "<p style='color:#9aa6c0;margin-top:4px'>Per-subject mastery and topic grid, derived from your predictions and attempts.</p>",
                unsafe_allow_html=True)

    if p.empty:
        info_banner("No predictions yet. Train the ML baseline and complete a few attempts to populate this view.")
        return

    subject = st.selectbox("Subject", SUBJECTS, key="subjects_select")
    subj_pred = p[p.subject == subject].sort_values("mastery_probability")
    overall = float(subj_pred.mastery_probability.mean()) if not subj_pred.empty else 0.0
    topic_count = int(subj_pred.topic.nunique())
    st.markdown(
        f"<div class='ml-card-elev'><div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div><div style='font-size:1.4rem;font-weight:700'>{subject}</div>"
        f"<div style='color:#9aa6c0'>{topic_count} topics in mastery model</div></div>"
        f"<div style='text-align:right'><div style='font-size:2rem;font-weight:700'>{overall:.0%}</div>"
        f"<div style='color:#9aa6c0'>Overall mastery</div></div></div>"
        f"<div class='ml-progress' style='margin-top:10px'><div style='width:{overall*100:.1f}%'></div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if subj_pred.empty:
        empty_state("No topic data yet", f"Complete some {subject} attempts to populate the topic grid.")
        return

    section_label("Topic mastery grid")
    grid_cols = st.columns(2)
    for i, row in enumerate(subj_pred.itertuples(index=False)):
        mastery = float(row.mastery_probability)
        bucket = "Strong" if mastery >= 0.75 else "Developing" if mastery >= 0.55 else "Weak"
        kind = "strong" if bucket == "Strong" else "dev" if bucket == "Developing" else "weak"
        with grid_cols[i % 2]:
            st.markdown(
                f"<div class='ml-card'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div style='font-weight:600'>{row.topic}</div>"
                f"{pill(bucket, kind)}</div>"
                f"<div style='display:flex;align-items:center;gap:12px;margin-top:6px'>"
                f"<div style='font-size:1.4rem;font-weight:700;min-width:54px'>{mastery:.0%}</div>"
                f"<div class='ml-progress' style='flex:1'><div style='width:{mastery*100:.1f}%'></div></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page: Practice
# ---------------------------------------------------------------------------

def _timer_fragment(limit: int):
    @st.fragment(run_every="1s")
    def tick():
        remaining = max(int(limit - (time.monotonic() - st.session_state.started)), 0)
        st.metric("Time remaining", f"{remaining}s")
        if remaining <= 0 and not st.session_state.get("answered"):
            try:
                st.rerun(scope="app")
            except Exception:
                st.rerun()

    tick()


def _finalize(
    q: pd.DataFrame,
    sid: str,
    question: pd.Series,
    answer,
    elapsed: float,
    confidence: int,
    timed_out: bool,
) -> None:
    if question.question_type == "Subjective":
        graded = grade_subjective(question, answer or "")
        correct = bool(graded["is_correct"])
        score = graded["score"]
        feedback = graded["feedback"]
    else:
        correct = bool(answer == question.correct_answer)
        score = None
        feedback = None
    attempt_id = save_answer(
        sid,
        question,
        correct,
        elapsed,
        confidence,
        answer_text=(answer if question.question_type == "Subjective" else None),
        score=score,
    )
    after = _topic_mastery(
        predictions_for(sid, user_attempts(sid), q), question.subject, question.topic
    )
    st.session_state.answered = True
    session = st.session_state.get("session")
    if session is not None:
        session["answered"] += 1
        session["correct"] += 1 if correct else 0
        session.setdefault("items", []).append(
            {
                "question_id": question.question_id,
                "topic": question.topic,
                "difficulty": question.difficulty,
                "subject": question.subject,
                "correct": correct,
                "elapsed": elapsed,
                "before": st.session_state.get("mastery_before"),
                "after": after,
            }
        )
    st.session_state.result = {
        "correct": correct,
        "score": score,
        "feedback": feedback,
        "timed_out": timed_out,
        "before": st.session_state.get("mastery_before"),
        "after": after,
        "attempt_id": attempt_id,
    }


def _question_panel(question: pd.Series, why: str, mastery_before: float | None) -> None:
    mastery_pct = f"{mastery_before:.0%}" if mastery_before is not None else "—"
    section_label("Personalised practice")
    st.markdown(
        f"<div class='ml-card-elev'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>"
        f"<div><div style='color:#9aa6c0;font-size:0.85rem'>Target</div>"
        f"<div style='font-weight:600'>{question.subject} → {question.topic}</div></div>"
        f"<div><div style='color:#9aa6c0;font-size:0.85rem'>Difficulty</div>"
        f"<div style='font-weight:600'>{question.difficulty} · rating {question.difficulty_rating:.2f}</div></div>"
        f"<div><div style='color:#9aa6c0;font-size:0.85rem'>Current mastery</div>"
        f"<div style='font-weight:600'>{mastery_pct}</div></div></div>"
        f"<div style='color:#9aa6c0;margin-top:10px;font-size:0.88rem'><b>Why this question?</b> {why}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='ml-card' style='margin-top:10px'><div style='font-size:1.05rem'>{question.question}</div></div>",
        unsafe_allow_html=True,
    )


def _answer_block(question: pd.Series, key: str):
    if question.question_type == "Subjective":
        return st.text_area("Your answer", height=180, key=key, placeholder="Type your reasoning here…")
    choices = {
        "A": question.option_a,
        "B": question.option_b,
        "C": question.option_c,
        "D": question.option_d,
    }
    return st.radio(
        "Choose an answer",
        list(choices),
        key=key,
        format_func=lambda k: f"{k}. {choices[k]}",
    )


def _result_panel(question: pd.Series) -> None:
    res = st.session_state.get("result")
    if res is None:
        return
    header = ("⏰ Time expired — " if res["timed_out"] else "") + (
        "Correct" if res["correct"] else "Review"
    )
    if res["correct"]:
        st.success(f"{header}\n\n{question.explanation}")
    else:
        st.error(f"{header}\n\n{question.explanation}")
    if res.get("feedback"):
        st.info(res["feedback"])
    if res.get("score") is not None:
        st.caption(f"Rubric score: {res['score']:.0%}")
    if res["after"] is not None:
        before, after = res["before"], res["after"]
        if before is None:
            st.markdown(
                f"<div class='ml-card'>📈 Topic mastery is now <b>{after:.0%}</b>.</div>",
                unsafe_allow_html=True,
            )
        else:
            delta = after - before
            arrow = "▲" if delta >= 0 else "▼"
            st.markdown(
                f"<div class='ml-card'>📈 Topic mastery: <b>{before:.0%} → {after:.0%}</b> "
                f"({arrow} {abs(delta):.0%}). Difficulty adapts from your next item.</div>",
                unsafe_allow_html=True,
            )
    if st.session_state.get("user_id") and res.get("attempt_id"):
        from database.db import record_feedback
        fb = st.session_state.get(f"feedback_{res['attempt_id']}")
        if fb is None:
            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button("👍 Useful", key=f"fb_up_{res['attempt_id']}"):
                record_feedback(res["attempt_id"], st.session_state.user_id, useful=True)
                st.session_state[f"feedback_{res['attempt_id']}"] = "up"
                st.rerun()
            if c2.button("👎 Not useful", key=f"fb_dn_{res['attempt_id']}"):
                record_feedback(res["attempt_id"], st.session_state.user_id, useful=False)
                st.session_state[f"feedback_{res['attempt_id']}"] = "down"
                st.rerun()
        else:
            st.caption(f"Feedback recorded: {'👍' if fb == 'up' else '👎'}")


def _session_summary(sid: str, q: pd.DataFrame, session: dict) -> None:
    items = session.get("items", [])
    total = session["answered"]
    correct = session["correct"]
    accuracy = correct / total if total else 0.0
    avg_time = sum(i["elapsed"] for i in items) / len(items) if items else 0.0

    st.markdown("<h2>Practice complete</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{correct} / {total}")
    c2.metric("Accuracy", f"{accuracy:.0%}")
    c3.metric("Avg response", f"{avg_time:.0f}s")

    st.markdown("<div class='ml-section-label'>Mastery update</div>", unsafe_allow_html=True)
    topic_deltas: dict[tuple[str, str], dict] = {}
    for it in items:
        key = (it["subject"], it["topic"])
        if key not in topic_deltas:
            topic_deltas[key] = {"before": it["before"], "after": it["after"]}
    if not topic_deltas:
        st.caption("No mastery changes recorded in this session.")
    for (subject, topic), d in topic_deltas.items():
        before = d["before"]
        after = d["after"]
        if before is None and after is None:
            continue
        delta_txt = ""
        if before is not None and after is not None:
            delta = after - before
            arrow = "▲" if delta >= 0 else "▼"
            delta_txt = f" ({arrow} {abs(delta):.0%})"
        st.markdown(
            f"<div class='ml-card'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<div><b>{subject} → {topic}</b></div>"
            f"<div>{'—' if before is None else f'{before:.0%}'} → "
            f"{'—' if after is None else f'{after:.0%}'}{delta_txt}</div></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='ml-section-label'>What we learned about you</div>", unsafe_allow_html=True)
    insights = []
    if items:
        by_diff = pd.DataFrame(items)
        if not by_diff.empty:
            for diff in ("Easy", "Medium", "Hard"):
                subset = by_diff[by_diff.difficulty == diff]
                if not subset.empty:
                    acc = float(subset.correct.mean())
                    insights.append(
                        f"{diff} accuracy this session: {acc:.0%} on {len(subset)} question(s)."
                    )
        slow = [i for i in items if i["elapsed"] > 90]
        if slow:
            insights.append(
                f"{len(slow)} answer(s) exceeded 90s — consider focusing on medium-difficulty pacing."
            )
        missed_topics = sorted({i["topic"] for i in items if not i["correct"]})
        if missed_topics:
            insights.append("Struggled on: " + ", ".join(missed_topics[:4]) + ".")
    if not insights:
        insights = ["No items recorded yet."]
    for line in insights:
        st.markdown(f"<div class='ml-card'>• {line}</div>", unsafe_allow_html=True)

    st.markdown("<div class='ml-section-label'>Next recommendation</div>", unsafe_allow_html=True)
    p = predictions_for(sid, user_attempts(sid), q)
    if not p.empty:
        recs = recommend_questions(sid, p, user_attempts(sid), q, limit=1)
        if not recs.empty:
            r = recs.iloc[0]
            st.markdown(
                f"<div class='ml-card-elev'><b>{r.topic}</b> ({r.subject}) — {r.recommended_difficulty}<br>"
                f"<span style='color:#9aa6c0'>{r.reason}</span></div>",
                unsafe_allow_html=True,
            )

    c_a, c_b = st.columns(2)
    if c_a.button("Another session", width="stretch", key="summary_again"):
        _start_session(session["subject"])
        st.rerun()
    next_subject_idx = SUBJECTS.index(session["subject"]) + 1
    if next_subject_idx < len(SUBJECTS):
        nxt = SUBJECTS[next_subject_idx]
        if c_b.button(f"Continue → {nxt}", type="primary", width="stretch", key="summary_next"):
            _start_session(nxt)
            st.rerun()


def practice_page(q: pd.DataFrame) -> None:
    if not st.session_state.get("user_id"):
        st.info("Sign in from the Profile page to start practising.")
        return
    sid = st.session_state.user_id
    attempts = user_attempts(sid)
    _apply_topic_prefill(q, attempts)
    current_subject = initialise_quiz_subject(sid)
    if st.session_state.get("quiz_subject_picker_subject") != current_subject:
        st.session_state.pop("quiz_subject_picker", None)
        st.session_state.quiz_subject_picker_subject = current_subject
    selected_subject = st.selectbox(
        "Practice subject", SUBJECTS, key="quiz_subject_picker",
        index=SUBJECTS.index(current_subject),
    )
    if selected_subject != current_subject:
        st.session_state.current_subject = selected_subject
        st.session_state.subject_manually_selected = True
        st.session_state.quiz_subject_picker_subject = selected_subject
        _clear_question_state()
        st.session_state.pop("session", None)
        st.rerun()

    queue = pending_queue_items(sid)
    if queue:
        st.caption(
            f"Scheduled future practice: {len(queue)} item(s) — next due "
            f"{pd.to_datetime(queue[0]['due_at'], format='ISO8601', utc=True):%d %b %H:%M} UTC."
        )

    session = st.session_state.get("session")
    if session is None:
        st.markdown(
            f"<div class='ml-card-elev'>"
            f"<div style='font-size:1.1rem;font-weight:600'>Ready to practise {selected_subject}.</div>"
            f"<div style='color:#9aa6c0;margin-top:6px'>A session asks {SESSION_LENGTH} questions "
            f"focused on your weakest topic in this subject. Incorrect answers are automatically "
            f"scheduled for review in {REVIEW_DELAY_HOURS} hours.</div></div>",
            unsafe_allow_html=True,
        )
        if st.button(f"Start session · {selected_subject}", type="primary", width="stretch"):
            _start_session(selected_subject)
            st.rerun()
        return

    if session["answered"] >= SESSION_LENGTH:
        _session_summary(sid, q, session)
        if st.button("Finish and return to dashboard", width="stretch", key="summary_finish"):
            st.session_state.pop("session", None)
            _clear_question_state()
            st.session_state.page = "Dashboard"
            st.rerun()
        return

    st.progress(
        session["answered"] / SESSION_LENGTH,
        text=f"Question {session['answered'] + 1} of {SESSION_LENGTH} · {session['subject']}",
    )
    if "active_question" not in st.session_state:
        item, why, before = next_question(sid, attempts, q)
        st.session_state.active_question = item.question_id
        st.session_state.last_served_id = item.question_id
        st.session_state.why = why
        st.session_state.started = time.monotonic()
        st.session_state.answered = False
        st.session_state.mastery_before = before
        st.session_state.pop("result", None)

    question = q[q.question_id == st.session_state.active_question].iloc[0]
    limit = SUBJECTIVE_TIME_LIMIT if question.question_type == "Subjective" else TIME_LIMITS[question.difficulty]
    _timer_fragment(limit)
    _question_panel(question, st.session_state.why, st.session_state.get("mastery_before"))
    answer = _answer_block(question, key=f"a_{question.question_id}")
    confidence = st.select_slider(
        "How confident are you?",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: ["Guessing", "Low", "Moderate", "High", "Very confident"][x - 1],
    )
    elapsed = time.monotonic() - st.session_state.started
    st.caption(f"Time on this question: {int(elapsed)}s")
    if not st.session_state.answered and elapsed > limit:
        _finalize(q, sid, question, answer, elapsed, confidence, timed_out=True)
    if not st.session_state.answered and st.button("Submit answer", type="primary", width="stretch"):
        _finalize(q, sid, question, answer, elapsed, confidence, timed_out=False)
    if st.session_state.answered:
        _result_panel(question)
        label = "See session summary" if session["answered"] >= SESSION_LENGTH else "Next question"
        if st.button(label, width="stretch", key="next_q_btn"):
            _clear_question_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Progress
# ---------------------------------------------------------------------------

def progress_page(q: pd.DataFrame) -> None:
    if not st.session_state.get("user_id"):
        st.info("Sign in to view your progress.")
        return
    sid = st.session_state.user_id
    attempts = user_attempts(sid)
    p = predictions_for(sid, attempts, q)

    st.markdown("<h2 style='margin-bottom:0'>My progress</h2>"
                "<p style='color:#9aa6c0;margin-top:4px'>All values come from your attempt history and trained model predictions.</p>",
                unsafe_allow_html=True)
    if attempts.empty:
        empty_state("No attempts yet", "Practise a few questions to populate progress analytics.")
        return

    section_label("Overall learning curve")
    curve = learning_curve_frame(attempts)
    if not curve.empty and curve.shape[0] >= 2:
        st.line_chart(curve, height=240, width="stretch")
    else:
        st.caption("Need a few attempts before the learning curve becomes informative.")

    section_label("Subject comparison")
    if not p.empty:
        subj = p.groupby("subject").mastery_probability.mean().reindex(SUBJECTS)
        st.bar_chart(subj, height=240, width="stretch")
    else:
        st.caption("Train the ML baseline to see subject comparison.")

    section_label("Topic mastery heatmap")
    if not p.empty:
        heat = p.pivot_table(
            index="subject", columns="topic", values="mastery_probability", aggfunc="mean"
        ).reindex(SUBJECTS)
        st.dataframe(
            heat.style.format("{:.0%}", na_rep="—").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            width="stretch",
        )
    else:
        st.caption("No predictions available.")

    section_label("Difficulty performance")
    diff_acc = attempts.groupby("difficulty").is_correct.mean().reindex(["Easy", "Medium", "Hard"])
    diff_time = attempts.groupby("difficulty").time_taken_seconds.mean().reindex(["Easy", "Medium", "Hard"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ml-section-label'>Accuracy by difficulty</div>", unsafe_allow_html=True)
        st.bar_chart(diff_acc, height=220, width="stretch")
    with c2:
        st.markdown("<div class='ml-section-label'>Avg response time (s)</div>", unsafe_allow_html=True)
        st.bar_chart(diff_time, height=220, width="stretch")

    section_label("Recent attempts")
    recent = attempts.sort_values("timestamp", ascending=False).head(15)[
        ["timestamp", "subject", "topic", "difficulty", "is_correct", "time_taken_seconds"]
    ].copy()
    recent["result"] = recent["is_correct"].map({1: "✓", 0: "✗"})
    recent["time_taken_seconds"] = recent["time_taken_seconds"].round(1)
    recent = recent.rename(columns={"time_taken_seconds": "time_sec"})
    st.dataframe(
        recent[["timestamp", "subject", "topic", "difficulty", "result", "time_sec"]],
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Page: AI Recommendations
# ---------------------------------------------------------------------------

def recommendations_page(q: pd.DataFrame) -> None:
    if not st.session_state.get("user_id"):
        st.info("Sign in to see personalised recommendations.")
        return
    sid = st.session_state.user_id
    attempts = user_attempts(sid)
    p = predictions_for(sid, attempts, q)

    st.markdown("<h2 style='margin-bottom:0'>Your personalised learning plan</h2>"
                "<p style='color:#9aa6c0;margin-top:4px'>Generated from your performance history and ML mastery predictions — never motivational filler.</p>",
                unsafe_allow_html=True)
    if p.empty:
        info_banner("Train the ML baseline in ML Experiments to enable personalised recommendations.")
        return

    sorted_pred = p.sort_values("mastery_probability")
    buckets = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for r in sorted_pred.itertuples(index=False):
        if r.mastery_probability < STATUSES_FOR_PRIORITY[0][1]:
            buckets["HIGH"].append(r)
        elif r.mastery_probability < STATUSES_FOR_PRIORITY[1][1]:
            buckets["MEDIUM"].append(r)
        else:
            buckets["LOW"].append(r)

    for label, kind, items in [
        ("HIGH priority", "weak", buckets["HIGH"][:5]),
        ("MEDIUM priority", "dev", buckets["MEDIUM"][:5]),
        ("LOW priority", "strong", buckets["LOW"][:5]),
    ]:
        if not items:
            continue
        section_label(label)
        for r in items:
            st.markdown(
                f"<div class='ml-card'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div><div style='font-weight:600'>{r.subject} → {r.topic}</div>"
                f"<div style='color:#9aa6c0;font-size:0.85rem'>Mastery {r.mastery_probability:.0%} · "
                f"recommended at {r.difficulty}</div></div>"
                f"{pill(r.status, kind)}</div></div>",
                unsafe_allow_html=True,
            )
            doc_key = (r.subject, r.topic)
            show_doc = st.session_state.get("_doc_topic") == doc_key
            if show_doc:
                _render_inline_doc(q, p, r.subject, r.topic)
            if st.button(
                "View reference + practise" if not show_doc else "Close reference",
                key=f"ref_{r.subject}_{r.topic}",
                width="stretch",
            ):
                st.session_state._doc_topic = None if show_doc else doc_key
                st.rerun()

    section_label("Today")
    recs = recommend_questions(sid, p, attempts, q, limit=5)
    if recs.empty:
        st.caption("No actionable recommendations yet — answer a few more questions.")
    else:
        for r in recs.itertuples(index=False):
            st.markdown(
                f"<div class='ml-card-elev'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div><div style='font-weight:600'>Practise {r.topic} — {r.recommended_difficulty}</div>"
                f"<div style='color:#9aa6c0;font-size:0.85rem'>{r.reason}</div></div>"
                f"<div>{pill(f'est. mastery {r.mastery_probability:.0%}', 'flat')}</div></div></div>",
                unsafe_allow_html=True,
            )
        if st.button("Queue this plan", type="primary", width="stretch", key="rec_queue"):
            now = datetime.now(timezone.utc)
            queued = 0
            for i, row in enumerate(recs.itertuples(index=False)):
                queued += enqueue_questions(
                    sid,
                    [{"question_id": row.question_id, "subject": row.subject, "topic": row.topic}],
                    (now + timedelta(days=i)).isoformat(),
                    reason="From your personalised study plan",
                )
            st.success(f"{queued} question(s) scheduled.")


def _render_inline_doc(q: pd.DataFrame, p: pd.DataFrame, subject: str, name: str) -> None:
    mastery = None
    if not p.empty:
        rows = p[(p.subject == subject) & (p.topic == name)]
        if not rows.empty:
            mastery = float(rows.mastery_probability.iloc[0])
    mastery_html = (
        f"<div style='text-align:right'><div style='font-size:1.4rem;font-weight:700'>{mastery:.0%}</div>"
        f"<div style='color:#9aa6c0;font-size:0.75rem'>predicted mastery</div></div>"
        if mastery is not None
        else "<div style='color:#9aa6c0'>mastery not yet estimated</div>"
    )
    links = _get_topic_links(subject, name)
    links_html = (
        "".join(
            f"<a href='{url}' target='_blank' rel='noopener noreferrer' "
            f"style='display:inline-block;margin:4px 6px 4px 0;padding:6px 10px;border-radius:8px;"
            f"background:#16213b;border:1px solid #2b3a5e;color:#c7d2fe;text-decoration:none;"
            f"font-size:0.8rem'>{label} ↗</a>"
            for label, url in links
        )
        if links
        else "<div style='color:#9aa6c0;font-size:0.85rem'>No external references mapped for this topic yet.</div>"
    )
    st.markdown(
        f"<div class='ml-card-elev' style='border-left:3px solid #6366f1'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px'>"
        f"<div><div style='font-size:1.05rem;font-weight:600'>Read about {name}</div>"
        f"<div style='color:#9aa6c0;margin-top:2px'>{subject} · external references</div></div>"
        f"{mastery_html}</div>"
        f"<div style='margin-top:10px'>{links_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    topic_qs = q[(q.subject == subject) & (q.topic == name)].sort_values(["difficulty", "question_id"])
    type_label = {"MCQ": "Choice", "Subjective": "Written"}
    if not topic_qs.empty:
        section_label("Questions in this topic")
        for row in topic_qs.itertuples(index=False):
            st.markdown(
                f"<div class='ml-card'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:12px'>"
                f"<div><div style='font-weight:600'>{row.difficulty} · {type_label.get(row.question_type, row.question_type)}</div>"
                f"<div style='color:#9aa6c0;font-size:0.85rem'>rating {row.difficulty_rating:.2f} · {row.question_id}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    if st.button("Start practice session on this topic", type="primary", width="stretch", key=f"doc_start_{subject}_{name}"):
        st.session_state.page = "Practice"
        st.session_state._prefill_topic = (subject, name)
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Question Bank
# ---------------------------------------------------------------------------

def question_bank_page(q: pd.DataFrame) -> None:
    st.markdown("<h2>Question bank</h2>", unsafe_allow_html=True)
    st.caption(f"{len(q)} questions across {q.subject.nunique()} subjects, "
               f"{q.topic.nunique()} topics, {q.difficulty.nunique()} difficulties.")
    f1, f2, f3 = st.columns(3)
    subject = f1.selectbox("Subject", ["All", *SUBJECTS], key="qb_subject")
    difficulty = f2.selectbox("Difficulty", ["All", "Easy", "Medium", "Hard"], key="qb_diff")
    qtype = f3.selectbox("Type", ["All", "MCQ", "Subjective"], key="qb_type")
    filtered = q.copy()
    if subject != "All":
        filtered = filtered[filtered.subject == subject]
    if difficulty != "All":
        filtered = filtered[filtered.difficulty == difficulty]
    if qtype != "All":
        filtered = filtered[filtered.question_type == qtype]
    st.caption(f"Showing {len(filtered)} of {len(q)} questions.")
    st.dataframe(
        filtered[["question_id", "subject", "topic", "difficulty", "difficulty_rating", "question_type"]],
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Page: Simulation
# ---------------------------------------------------------------------------

def simulation_page() -> None:
    st.markdown("<h2>Synthetic student laboratory</h2>", unsafe_allow_html=True)
    info_banner(NOTICE)
    with st.form("sim"):
        a, b, c = st.columns(3)
        students = a.select_slider("Number of students", [10, 30, 100, 1000, 10000], value=30)
        per = b.number_input("Attempts per student", 10, 500, 60, 10)
        seed = c.number_input("Random seed", 0, value=42)
        d, e, f = st.columns(3)
        learning = d.slider("Learning rate", 0.005, 0.10, 0.035, 0.005)
        ability = e.selectbox("Ability mix", ["Typical", "Emerging-heavy", "Advanced-heavy"])
        difficulty = f.selectbox("Difficulty mix", ["Balanced", "Foundation", "Challenge"])
        go = st.form_submit_button("Generate synthetic cohort", type="primary", width="stretch")

    if go:
        abilities = {
            "Typical": {"emerging": 0.18, "developing": 0.38, "proficient": 0.30, "advanced": 0.14},
            "Emerging-heavy": {"emerging": 0.42, "developing": 0.34, "proficient": 0.18, "advanced": 0.06},
            "Advanced-heavy": {"emerging": 0.06, "developing": 0.18, "proficient": 0.36, "advanced": 0.40},
        }
        diffs = {
            "Balanced": {"Easy": 0.4, "Medium": 0.4, "Hard": 0.2},
            "Foundation": {"Easy": 0.6, "Medium": 0.3, "Hard": 0.1},
            "Challenge": {"Easy": 0.2, "Medium": 0.45, "Hard": 0.35},
        }
        cfg = SimulationConfig(
            number_of_students=int(students),
            attempts_per_student=int(per),
            random_seed=int(seed),
            learning_rate=float(learning),
            ability_distribution=abilities[ability],
            difficulty_distribution=diffs[difficulty],
        )
        with st.spinner("Generating synthetic histories…"):
            result = run_simulation(cfg)
        generated_attempts_df.clear()
        generated_truth_df.clear()
        st.success(f"Generated {len(result['attempts']):,} synthetic attempts across {int(students):,} students.")
        a, b, c, d = st.columns(4)
        a.metric("Students", f"{int(students):,}")
        b.metric("Attempts", f"{len(result['attempts']):,}")
        c.metric("Subjects", result["attempts"].subject.nunique())
        d.metric("Avg accuracy", f"{result['attempts'].is_correct.mean():.1%}")
        for name, frame in [
            ("students", result["students"]),
            ("attempts", result["attempts"]),
            ("synthetic_truth", result["truth"]),
        ]:
            st.download_button(
                f"Download {name}.csv",
                frame.to_csv(index=False).encode(),
                f"{name}.csv",
                "text/csv",
                key=f"dl_{name}",
            )

    info_banner(NOTICE)
    st.markdown(
        "<div style='color:#9aa6c0;font-size:0.85rem'>All records above are SYNTHETIC. They must never be presented as real "
        "student-study data; any model trained on them must be re-validated on ethically approved real data before "
        "deployment to learners.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: ML Experiments
# ---------------------------------------------------------------------------

def _show_confusion(name: str, results: dict) -> None:
    path = RESULTS_DIR / f"{name}_confusion_matrix.png"
    if path.exists():
        st.image(str(path), caption=name.replace("_", " "))


def ml_experiments_page() -> None:
    st.markdown("<h2>Mastery prediction experiments</h2>", unsafe_allow_html=True)
    info_banner(NOTICE)
    a = generated_attempts_df()
    truth = generated_truth_df()
    results = experiment_results()
    trained = bool(results.get("trained"))

    if a.empty:
        empty_state("No synthetic cohort", "Generate a synthetic cohort on the Simulation page before training.")
        return

    n_students = a.student_id.nunique()
    n_attempts = len(a)
    n_features = 15  # matches training_metadata.json feature_columns
    info_banner(
        f"Current dataset: {n_students:,} students, {n_attempts:,} attempts, {n_features} features. "
        "Split protocol: 70% train / 15% validation / 15% test by student (GroupShuffleSplit)."
    )

    if not trained and st.button("Train Logistic Regression, Random Forest & XGBoost", type="primary", width="stretch"):
        if truth.empty:
            st.error("Synthetic truth labels missing — regenerate the cohort.")
            return
        with st.spinner("Fitting models on the current synthetic cohort…"):
            _, test, best = run_experiment_from_data(a, truth)
        st.success(f"Training complete. Validation-selected model: {best}.")
        st.cache_data.clear()
        st.rerun()

    if not trained:
        empty_state("Experiment not run yet", "Click the train button above to fit the models and populate metrics.")
        return

    st.cache_data.clear()
    results = experiment_results()
    test_df = results.get("test")
    val_df = results.get("validation")
    if isinstance(val_df, pd.DataFrame) and not val_df.empty:
        section_label("Validation model selection")
        st.dataframe(val_df, hide_index=True, width="stretch")
        best_model = val_df.iloc[0]["model"]
    else:
        best_model = None
    if isinstance(test_df, pd.DataFrame) and not test_df.empty:
        section_label("Held-out test metrics")
        st.dataframe(test_df, hide_index=True, width="stretch")

    section_label("Confusion matrices")
    cols = st.columns(3)
    for col, name in zip(cols, ["logistic_regression", "random_forest", "xgboost"]):
        with col:
            _show_confusion(name, results)

    roc = RESULTS_DIR / "test_roc_curves.png"
    if roc.exists():
        section_label("ROC curves (held-out test students)")
        st.image(str(roc), width="stretch")

    if best_model:
        imp_path = RESULTS_DIR / f"{best_model}_feature_importance.csv"
        if imp_path.exists():
            section_label(f"Feature importance · {best_model}")
            imp = pd.read_csv(imp_path).head(15)
            st.bar_chart(imp.set_index("feature")["importance"], height=320, width="stretch")

    section_label("Best model")
    if best_model:
        best_row = test_df[test_df["model"] == best_model].iloc[0] if isinstance(test_df, pd.DataFrame) else None
        if best_row is not None:
            st.markdown(
                f"<div class='ml-card-elev'>"
                f"<div style='font-size:1.2rem;font-weight:600'>{best_model.replace('_', ' ')}</div>"
                f"<div style='color:#9aa6c0'>Selected on validation F1, evaluated on held-out test students.</div>"
                f"<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px'>"
                f"<div><div class='ml-section-label' style='margin:0'>Accuracy</div><div style='font-size:1.2rem;font-weight:700'>{best_row.get('accuracy', 0):.1%}</div></div>"
                f"<div><div class='ml-section-label' style='margin:0'>Precision</div><div style='font-size:1.2rem;font-weight:700'>{best_row.get('precision', 0):.1%}</div></div>"
                f"<div><div class='ml-section-label' style='margin:0'>Recall</div><div style='font-size:1.2rem;font-weight:700'>{best_row.get('recall', 0):.1%}</div></div>"
                f"<div><div class='ml-section-label' style='margin:0'>F1</div><div style='font-size:1.2rem;font-weight:700'>{best_row.get('f1', 0):.1%}</div></div>"
                f"<div><div class='ml-section-label' style='margin:0'>ROC-AUC</div><div style='font-size:1.2rem;font-weight:700'>{best_row.get('roc_auc', 0):.1%}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page: Research Analytics
# ---------------------------------------------------------------------------

def analytics_page() -> None:
    st.markdown("<h2>Research analytics</h2>", unsafe_allow_html=True)
    info_banner(NOTICE)
    a = generated_attempts_df()
    results = experiment_results()

    if a.empty:
        empty_state("No synthetic dataset", "Generate a synthetic cohort first.")
        return

    stats = dataset_statistics(a)
    section_label("Dataset overview")
    c = st.columns(5)
    c[0].metric("Students", f"{stats['students']:,}")
    c[1].metric("Attempts", f"{stats['attempts']:,}")
    c[2].metric("Subjects", a.subject.nunique())
    c[3].metric("Topics", a.topic.nunique())
    c[4].metric("Questions", questions_df().shape[0])

    section_label("Learning behaviour")
    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown("<div class='ml-section-label'>Accuracy distribution</div>", unsafe_allow_html=True)
        st.bar_chart(a.groupby("difficulty").is_correct.mean().reindex(["Easy", "Medium", "Hard"]), height=240)
    with cb2:
        st.markdown("<div class='ml-section-label'>Response time distribution</div>", unsafe_allow_html=True)
        st.bar_chart(
            a.groupby("difficulty").time_taken_seconds.mean().reindex(["Easy", "Medium", "Hard"]),
            height=240,
        )

    section_label("Mastery analysis")
    subject, topic = performance_by_subject_topic(a)
    cs1, cs2 = st.columns(2)
    with cs1:
        st.markdown("<div class='ml-section-label'>Subject mastery</div>", unsafe_allow_html=True)
        st.bar_chart(subject.set_index("subject")["accuracy"], height=260)
    with cs2:
        st.markdown("<div class='ml-section-label'>Weakest topics</div>", unsafe_allow_html=True)
        st.dataframe(topic.sort_values("accuracy").head(10), hide_index=True, width="stretch")

    section_label("Model performance")
    if results.get("trained") and isinstance(results.get("test"), pd.DataFrame) and not results["test"].empty:
        st.dataframe(results["test"], hide_index=True, width="stretch")
        roc = RESULTS_DIR / "test_roc_curves.png"
        if roc.exists():
            st.image(str(roc), caption="ROC curves from held-out test students", width="stretch")
    else:
        empty_state("Experiment not run yet", "Train the models in ML Experiments to populate this section.")

    section_label("Adaptive learning results")
    rec_df = results.get("recommendation")
    if isinstance(rec_df, pd.DataFrame) and not rec_df.empty:
        st.dataframe(rec_df, hide_index=True, width="stretch")
    else:
        empty_state("Policy comparison not run", "Run a recommendation-policy experiment to compare random / rule-based / ML-based selection.")


# ---------------------------------------------------------------------------
# Sidebar + routing
# ---------------------------------------------------------------------------

PAGES = [
    ("Dashboard", "🎯"),
    ("Practice", "📝"),
    ("My Progress", "📈"),
    ("Recommendations", "🗺️"),
    ("Subjects", "📚"),
    ("Question Bank", "🗃️"),
    ("Simulation", "⚗️"),
    ("ML Experiments", "🧠"),
    ("Research Analytics", "📊"),
    ("Profile", "👤"),
]


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 16px 0;'>"
            "<div style='display:flex;align-items:center;gap:10px'>"
            "<div style='width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#22d3ee);'></div>"
            "<div><div class='ml-sidebar-brand'>MasteryLab</div>"
            "<div class='ml-sidebar-tag'>Adaptive learning</div></div></div></div>",
            unsafe_allow_html=True,
        )
        section_label("Main")
        for label, icon in PAGES[:4]:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", width="stretch"):
                st.session_state.page = label
                st.rerun()
        section_label("Learning")
        for label, icon in PAGES[4:6]:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", width="stretch"):
                st.session_state.page = label
                st.rerun()
        section_label("Research")
        for label, icon in PAGES[6:9]:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", width="stretch"):
                st.session_state.page = label
                st.rerun()
        section_label("System")
        for label, icon in PAGES[9:]:
            if st.button(f"{icon}  {label}", key=f"nav_{label}", width="stretch"):
                st.session_state.page = label
                st.rerun()

        if st.session_state.get("user_id"):
            with connection() as conn:
                row = conn.execute(
                    "SELECT display_name FROM students WHERE student_id=?",
                    (st.session_state.user_id,),
                ).fetchone()
            name = row["display_name"] if row else "Learner"
            streak = streak_days(user_attempts(st.session_state.user_id))
            st.markdown(
                f"<div class='ml-profile-card'>"
                f"<div class='name'>{name}</div>"
                f"<div class='meta'>{st.session_state.user_id} · 🔥 {streak} day streak</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    return st.session_state.get("page", "Dashboard")


def setup() -> pd.DataFrame:
    initialise_database()
    q = questions_df()
    seed_question_bank(q.to_dict("records"))
    return q


def main() -> None:
    st.set_page_config(
        page_title="MasteryLab",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    q = setup()
    if not session_valid():
        st.toast("Your session expired. Please sign in again.", icon="🔒")
    page = _sidebar()

    if page == "Profile":
        login_page()
    elif page == "Practice":
        practice_page(q)
    elif page == "My Progress":
        progress_page(q)
    elif page == "Recommendations":
        recommendations_page(q)
    elif page == "Subjects":
        subjects_page(q)
    elif page == "Question Bank":
        question_bank_page(q)
    elif page == "Simulation":
        simulation_page()
    elif page == "ML Experiments":
        ml_experiments_page()
    elif page == "Research Analytics":
        analytics_page()
    else:
        dashboard_page(q)


if __name__ == "__main__":
    main()
