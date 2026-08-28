"""Generate realistic responses from hidden mastery, difficulty, noise and time."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


DIFFICULTY_SKILL = {"Easy": 0.28, "Medium": 0.53, "Hard": 0.78}
DIFFICULTY_TIME = {"Easy": 50.0, "Medium": 88.0, "Hard": 142.0}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _nearest_question(candidates: pd.DataFrame, requested: str) -> pd.Series:
    """Select a question at/near the requested difficulty within a topic."""
    order = {"Easy": 0, "Medium": 1, "Hard": 2}
    distances = candidates["difficulty"].map(lambda d: abs(order[d] - order[requested]))
    return candidates.loc[distances[distances == distances.min()].index].sample(n=1).iloc[0]


def generate_attempts(
    students: pd.DataFrame,
    profiles: dict[str, dict[str, dict[str, float]]],
    questions: pd.DataFrame,
    attempts_per_student: int,
    learning_rate: float,
    noise_level: float,
    difficulty_distribution: dict[str, float],
    seed: int,
    start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate repeated practice and return attempts plus latent-skill snapshots.

    A response follows a logistic item-response-like mechanism. Mastery changes
    slightly after practice, so progression is created causally rather than by
    post-hoc random labels.
    """
    if attempts_per_student < 1:
        raise ValueError("attempts_per_student must be at least 1")
    rng = np.random.default_rng(seed + 1)
    subject_topics = questions[["subject", "topic"]].drop_duplicates().values.tolist()
    difficulty_names = list(difficulty_distribution)
    difficulty_weights = np.array([difficulty_distribution[d] for d in difficulty_names], dtype=float)
    difficulty_weights /= difficulty_weights.sum()
    start = datetime.fromisoformat(start_date)
    attempts, truth_rows = [], []

    for student_record in students.itertuples(index=False):
        student_id = student_record.student_id
        skill = {subject: dict(topics) for subject, topics in profiles[student_id].items()}
        seen_by_topic: dict[tuple[str, str], int] = defaultdict(int)
        recent_by_topic: dict[tuple[str, str], deque[int]] = defaultdict(lambda: deque(maxlen=3))
        timestamp = start + timedelta(days=float(rng.uniform(0, 2)))
        for n in range(1, attempts_per_student + 1):
            # Students revisit weaker / less-practised topics more frequently.
            topic_weights = np.array([
                (1.15 - skill[s][t]) * (1.25 if seen_by_topic[(s, t)] < 2 else 1.0)
                for s, t in subject_topics
            ])
            subject, topic = subject_topics[int(rng.choice(len(subject_topics), p=topic_weights / topic_weights.sum()))]
            recent = recent_by_topic[(subject, topic)]
            requested = str(rng.choice(difficulty_names, p=difficulty_weights))
            if len(recent) >= 2 and sum(recent) == 2:
                requested = "Hard" if requested != "Hard" else "Medium"
            elif len(recent) >= 2 and sum(recent) == 0:
                requested = "Easy"
            q = _nearest_question(questions[(questions.subject == subject) & (questions.topic == topic)], requested)
            question_difficulty = q.difficulty
            current_skill = skill[subject][topic]
            # Difficulty lowers probability, while Gaussian noise permits exceptions.
            probability = float(np.clip(sigmoid(4.2 * (current_skill - DIFFICULTY_SKILL[question_difficulty])) + rng.normal(0, noise_level), 0.03, 0.97))
            correct = bool(rng.random() < probability)
            time_mean = DIFFICULTY_TIME[question_difficulty] * (1.55 - current_skill) * (0.90 if correct else 1.13)
            response_time = float(np.clip(rng.lognormal(np.log(max(time_mean, 8)), 0.28), 5, 900))
            prior_seen = seen_by_topic[(subject, topic)]
            # Correct practice improves more; incorrect practice still has a small learning effect.
            gain = learning_rate * (1.0 - current_skill) * (1.0 if correct else 0.22) * (1.05 if question_difficulty != "Easy" else 0.8)
            skill[subject][topic] = float(np.clip(current_skill + gain + rng.normal(0, noise_level * 0.025), 0.03, 0.99))
            timestamp += timedelta(hours=float(rng.uniform(3, 38)))
            session_id = f"{student_id}_SESSION_{((n - 1) // 10) + 1:03d}"
            attempts.append({
                "student_id": student_id, "question_id": q.question_id, "subject": subject,
                "topic": topic, "difficulty": question_difficulty, "is_correct": int(correct),
                "time_taken_seconds": round(response_time, 2), "attempt_number": n,
                "timestamp": timestamp.isoformat(), "session_id": session_id,
                "is_synthetic": True, "data_label": "SYNTHETIC / SIMULATED — NOT REAL STUDENT DATA",
            })
            truth_rows.append({
                "student_id": student_id, "subject": subject, "topic": topic,
                "attempt_number": n, "timestamp": timestamp.isoformat(),
                "synthetic_true_mastery": round(skill[subject][topic], 6),
                "synthetic_mastered_label": int(skill[subject][topic] >= 0.70),
            })
            seen_by_topic[(subject, topic)] = prior_seen + 1
            recent.append(int(correct))
    return pd.DataFrame(attempts), pd.DataFrame(truth_rows)
