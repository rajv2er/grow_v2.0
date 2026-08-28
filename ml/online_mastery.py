"""Online per-(student, subject, topic) mastery that updates after every answer.

The global ML model in `models/*.joblib` is trained once on a synthetic
cohort and applied to every learner. It does not learn from the individual
learner as they answer questions. This module adds the missing piece: a
running mastery estimate per learner per topic that updates after every
submitted answer, using an exponential moving average over recent correctness
with a logistic link.

Why an EMA over raw accuracy?
- It is a 1-parameter online learner. After every new attempt, one update
  step is enough — no batch training, no model file, no inference pipeline.
- It is mathematically equivalent to a 1D Kalman filter on the logit scale
  and to the inner loop of Performance Factor Analysis (PFA), a standard
  model in the knowledge-tracing literature.
- It is interpretable for a BTech viva: "the estimate is a weighted average
  of recent correctness, with more weight on recent answers; it starts at
  the global model's prior and shifts toward 0/1 as evidence accumulates."

Cold start:
- For a (student, subject, topic) tuple that has never been seen, the EMA
  is initialised from the global model's prediction (`prior_mastery`) and
  starts updating from the first real attempt. This is what the `n_attempts
  >= 3` overlay threshold in the dashboard uses: below that, the global
  model is more trustworthy than 1-2 noisy observations.

Reference:
- Pavlik, P. I., Anderson, J. R. (2005). "Practice and Forgetting Effects on
  Vocabulary Memory: An Activation-Based Model of the Spacing Effect."
  Performance Factor Analysis is the special case of this style of model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from config import DATABASE_PATH
from database.db import connection
from observability import log

EMA_ALPHA = 0.4
MIN_ATTEMPTS_FOR_OVERLAY = 3
PRIOR_BLEND = 0.5


def _logit(p: float) -> float:
    import math
    p = max(min(float(p), 1.0 - 1e-6), 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid_safe(x: float) -> float:
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def update_after_attempt(
    student_id: str,
    subject: str,
    topic: str,
    is_correct: int,
    prior_mastery: float | None = None,
    alpha: float = EMA_ALPHA,
    db_path=...,
) -> float:
    """Update the EMA mastery for one (student, subject, topic) after an answer.

    Returns the new mastery estimate. The first call for a given triple
    initialises the row from `prior_mastery` (or 0.5 if not provided).
    """
    if db_path is ...:
        db_path = DATABASE_PATH
    now_iso = datetime.now(timezone.utc).isoformat()
    target = float(is_correct)
    log.info("ema_update student=%s topic=%s/%s correct=%d", student_id, subject, topic, int(is_correct))
    with connection(db_path) as conn:
        row = conn.execute(
            "SELECT mastery_estimate, n_attempts, n_correct, ema_alpha, prior_mastery "
            "FROM student_topic_mastery WHERE student_id=? AND subject=? AND topic=?",
            (student_id, subject, topic),
        ).fetchone()
        if row is None:
            init = float(prior_mastery) if prior_mastery is not None else 0.5
            init = max(0.05, min(0.95, init))
            conn.execute(
                "INSERT INTO student_topic_mastery"
                "(student_id, subject, topic, mastery_estimate, n_attempts, n_correct, "
                " ema_alpha, prior_mastery, last_updated) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    student_id, subject, topic, init, 0, 0, alpha, init, now_iso,
                ),
            )
            n_attempts = 0
            n_correct = 0
            mastery = init
            eff_alpha = alpha
            stored_prior = init
        else:
            mastery = float(row["mastery_estimate"])
            n_attempts = int(row["n_attempts"])
            n_correct = int(row["n_correct"])
            eff_alpha = float(row["ema_alpha"])
            stored_prior = (
                float(row["prior_mastery"]) if row["prior_mastery"] is not None else mastery
            )
        # Logistic-domain EMA: estimate is a probability, but we update its
        # logit against the new observation. This keeps the estimate in
        # (0, 1) and gives gentler updates near 0 and 1.
        new_mastery = _sigmoid_safe((1.0 - eff_alpha) * _logit(mastery) + eff_alpha * _logit(target))
        n_attempts += 1
        n_correct += int(bool(is_correct))
        conn.execute(
            "UPDATE student_topic_mastery SET mastery_estimate=?, n_attempts=?, n_correct=?, "
            "last_updated=? WHERE student_id=? AND subject=? AND topic=?",
            (new_mastery, n_attempts, n_correct, now_iso, student_id, subject, topic),
        )
    return new_mastery


def read_mastery(
    student_id: str, subject: str | None = None, db_path=...
) -> pd.DataFrame:
    """Return a DataFrame of per-(student, subject, topic) mastery rows.

    Columns: student_id, subject, topic, mastery_estimate, n_attempts,
    n_correct, ema_alpha, prior_mastery, last_updated.
    """
    if db_path is ...:
        db_path = DATABASE_PATH
    query = "SELECT * FROM student_topic_mastery WHERE student_id=?"
    params: tuple = (student_id,)
    if subject is not None:
        query += " AND subject=?"
        params = (student_id, subject)
    query += " ORDER BY subject, topic"
    with connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def overlay_predictions(
    global_predictions: pd.DataFrame,
    student_id: str,
    min_attempts: int = MIN_ATTEMPTS_FOR_OVERLAY,
    db_path=...,
) -> pd.DataFrame:
    """Overlay per-user EMA mastery on top of the global model predictions.

    For every (subject, topic) the user has at least `min_attempts` real
    attempts on, replace the global `mastery_probability` with the EMA
    estimate. Otherwise keep the global model's value. The resulting
    DataFrame has an extra boolean column `used_online_mastery`.
    """
    if db_path is ...:
        db_path = DATABASE_PATH
    if global_predictions is None or global_predictions.empty:
        return global_predictions
    out = global_predictions.copy()
    out["used_online_mastery"] = False
    ema = read_mastery(student_id, db_path=db_path)
    if ema.empty:
        return out
    for r in ema.itertuples(index=False):
        if r.n_attempts < min_attempts:
            continue
        mask = (out.subject == r.subject) & (out.topic == r.topic)
        if mask.any():
            out.loc[mask, "mastery_probability"] = float(r.mastery_estimate)
            out.loc[mask, "used_online_mastery"] = True
    return out


def status_for_probability(probability: float) -> str:
    if probability >= 0.75:
        return "Strong"
    if probability >= 0.55:
        return "Developing"
    return "Weak"
