"""Novelty-aware recommendations derived from mastery predictions."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import DATABASE_PATH
from database.db import connection
from recommendation.adaptive_difficulty import ORDER, next_difficulty


def recommend_questions(
    student_id: str,
    predictions: pd.DataFrame,
    attempts: pd.DataFrame,
    questions: pd.DataFrame,
    limit: int = 5,
    db_path=DATABASE_PATH,
) -> pd.DataFrame:
    """Prioritize weak concepts, matching difficulty, coverage and unseen items."""
    student_attempts = attempts[attempts.student_id == student_id]
    seen_ids = set(student_attempts.question_id)
    records = []
    for prediction in predictions.itertuples(index=False):
        topic_history = student_attempts[(student_attempts.subject == prediction.subject) & (student_attempts.topic == prediction.topic)]
        target = next_difficulty(topic_history, float(prediction.mastery_probability))
        candidates = questions[(questions.subject == prediction.subject) & (questions.topic == prediction.topic)].copy()
        candidates["seen"] = candidates.question_id.isin(seen_ids)
        # Match requested difficulty where possible; otherwise use nearest level.
        candidates["difficulty_distance"] = candidates.difficulty.map(lambda d: abs(ORDER.index(d) - ORDER.index(target)))
        candidates = candidates.sort_values(["seen", "difficulty_distance", "question_id"])
        if candidates.empty:
            continue
        q = candidates.iloc[0]
        novelty_bonus = 0.15 if not bool(q.seen) else 0.0
        score = (1.0 - float(prediction.mastery_probability)) + novelty_bonus - 0.08 * float(q.difficulty_distance)
        records.append({
            "student_id": student_id, "question_id": q.question_id, "subject": q.subject,
            "topic": q.topic, "recommended_difficulty": q.difficulty,
            "mastery_probability": float(prediction.mastery_probability), "score": round(score, 4),
            "reason": f"{prediction.status} mastery ({prediction.mastery_probability:.0%}); target difficulty is {target}; {'new question' if not q.seen else 'spaced revision'}.",
        })
    result = pd.DataFrame(records).sort_values("score", ascending=False).drop_duplicates("question_id").head(limit).reset_index(drop=True)
    if not result.empty:
        with connection(db_path) as conn:
            conn.executemany(
                """INSERT INTO recommendations(student_id, question_id, subject, topic,
                   recommended_difficulty, reason, score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(r.student_id, r.question_id, r.subject, r.topic, r.recommended_difficulty, r.reason, r.score, datetime.now(timezone.utc).isoformat()) for r in result.itertuples(index=False)],
            )
    return result
