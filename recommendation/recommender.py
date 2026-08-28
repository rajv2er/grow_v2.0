"""Novelty-aware recommendations derived from mastery predictions."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from config import DATABASE_PATH
from database.db import connection
from ml.online_mastery import read_mastery
from ml.recommendation_explainer import build_explanation
from observability import log, timed
from recommendation.adaptive_difficulty import TARGET_BAND, target_difficulty_rating


def recommend_questions(
    student_id: str,
    predictions: pd.DataFrame,
    attempts: pd.DataFrame,
    questions: pd.DataFrame,
    limit: int = 5,
    exclude: set[str] | None = None,
    db_path=DATABASE_PATH,
) -> pd.DataFrame:
    """Prioritize weak concepts, then unseen items whose numeric rating sits in the mastery band."""
    student_attempts = attempts[attempts.student_id == student_id]
    seen_ids = set(student_attempts.question_id)
    skip = exclude or set()
    ema_rows = read_mastery(student_id, db_path=db_path)
    ema_by_topic = {
        (r.subject, r.topic): float(r.mastery_estimate)
        for r in ema_rows.itertuples(index=False)
    } if not ema_rows.empty else {}
    records = []
    for prediction in predictions.itertuples(index=False):
        topic_history = student_attempts[(student_attempts.subject == prediction.subject) & (student_attempts.topic == prediction.topic)]
        mastery = float(prediction.mastery_probability)
        target = target_difficulty_rating(topic_history, mastery)
        candidates = questions[(questions.subject == prediction.subject) & (questions.topic == prediction.topic)].copy()
        if candidates.empty or "difficulty_rating" not in candidates.columns:
            continue
        candidates["seen"] = candidates.question_id.isin(seen_ids)
        candidates["rating_distance"] = (candidates.difficulty_rating.astype(float) - target).abs()
        candidates["in_band"] = candidates.rating_distance <= TARGET_BAND
        if skip:
            candidates = candidates[~candidates.question_id.isin(skip)]
            if candidates.empty:
                continue
        candidates = candidates.sort_values(["in_band", "seen", "rating_distance", "question_id"], ascending=[False, True, True, True])
        q = candidates.iloc[0]
        novelty_bonus = 0.15 if not bool(q.seen) else 0.0
        score = (1.0 - mastery) + novelty_bonus - 0.5 * float(q.rating_distance) + (0.10 if bool(q.in_band) else 0.0)
        ema_estimate = ema_by_topic.get((prediction.subject, prediction.topic))
        explanation = build_explanation(
            subject=prediction.subject,
            topic=prediction.topic,
            mastery_probability=mastery,
            topic_history=topic_history,
            ema_estimate=ema_estimate,
        )
        records.append({
            "student_id": student_id, "question_id": q.question_id, "subject": q.subject,
            "topic": q.topic, "recommended_difficulty": q.difficulty,
            "question_type": q.get("question_type", "MCQ"), "target_rating": target,
            "difficulty_rating": float(q.difficulty_rating), "in_band": bool(q.in_band),
            "mastery_probability": mastery, "score": round(score, 4),
            "reason": (
                f"{prediction.status} mastery ({mastery:.0%}); target rating {target:.2f} (±{TARGET_BAND}); "
                f"serving {float(q.difficulty_rating):.2f} ({q.difficulty}"
                f"{', ' + str(q.question_type) if q.get('question_type') == 'Subjective' else ''}); "
                f"{'new question' if not q.seen else 'spaced revision'}."
            ),
            "explanation_json": json.dumps(explanation),
            "confidence": float(explanation["confidence"]),
        })
    result = pd.DataFrame(records).sort_values("score", ascending=False).drop_duplicates("question_id").head(limit).reset_index(drop=True)
    log.info("recommend student=%s limit=%d n_predictions=%d n_attempts=%d", student_id, limit, len(predictions), len(student_attempts))
    with timed("recommend_questions", student=student_id):
        if not result.empty:
            with connection(db_path) as conn:
                # Snapshot semantics: the table always reflects the latest plan per student.
                conn.execute("DELETE FROM recommendations WHERE student_id=?", (student_id,))
                conn.executemany(
                    """INSERT INTO recommendations(student_id, question_id, subject, topic,
                       recommended_difficulty, reason, score, explanation_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(r.student_id, r.question_id, r.subject, r.topic, r.recommended_difficulty, r.reason, r.score, r.explanation_json, datetime.now(timezone.utc).isoformat()) for r in result.itertuples(index=False)],
                )
        log.info("recommend done student=%s n_returned=%d", student_id, len(result))
    return result
