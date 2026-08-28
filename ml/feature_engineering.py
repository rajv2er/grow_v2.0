"""Temporal, history-only features for synthetic mastery modelling."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "prior_attempts", "overall_accuracy", "subject_accuracy", "topic_accuracy",
    "recent_accuracy_5", "recent_accuracy_10", "topic_avg_response_time",
    "topic_response_time_std", "difficulty_accuracy", "previous_correct",
    "improvement_trend", "hours_since_topic_attempt",
]
CATEGORICAL_FEATURES = ["subject", "topic", "difficulty"]
TARGET_COLUMN = "mastered_label"


def _summary(stats: list[float] | None, default_accuracy: float = 0.5) -> tuple[float, float, float]:
    """Return count, accuracy and response-time standard deviation from prior data."""
    if not stats or stats[0] == 0:
        return 0.0, default_accuracy, 0.0
    count, correct, time_sum, time_square_sum = stats
    mean = time_sum / count
    variance = max(time_square_sum / count - mean * mean, 0.0)
    return count, correct / count, float(np.sqrt(variance))


def build_feature_dataset(attempts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Build records using only events strictly before each prediction time.

    This avoids the common leakage error of computing a student's topic accuracy
    over their complete history before splitting the dataset.
    """
    required = {"student_id", "subject", "topic", "difficulty", "is_correct", "time_taken_seconds", "timestamp", "attempt_number"}
    missing = required - set(attempts.columns)
    if missing:
        raise ValueError(f"attempts missing columns: {sorted(missing)}")
    events = attempts.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events = events.sort_values(["student_id", "timestamp", "attempt_number"], kind="stable").reset_index(drop=True)
    target = truth[["student_id", "attempt_number", "timestamp", "synthetic_mastered_label", "synthetic_true_mastery"]].copy()
    target["timestamp"] = pd.to_datetime(target["timestamp"], utc=True)
    events = events.merge(target, on=["student_id", "attempt_number", "timestamp"], how="left", validate="one_to_one")
    if events["synthetic_mastered_label"].isna().any():
        raise ValueError("Synthetic truth must align one-to-one with attempts.")

    rows: list[dict] = []
    for student_id, student_events in events.groupby("student_id", sort=False):
        overall: list[float] = [0.0, 0.0, 0.0, 0.0]
        subject_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        topic_stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        difficulty_stats: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        topic_recent: dict[tuple[str, str], deque[int]] = defaultdict(lambda: deque(maxlen=10))
        topic_last_time: dict[tuple[str, str], pd.Timestamp] = {}
        for event in student_events.itertuples(index=False):
            topic_key = (event.subject, event.topic)
            diff_key = (event.subject, event.topic, event.difficulty)
            prior_count, overall_acc, _ = _summary(overall)
            _, subject_acc, _ = _summary(subject_stats[event.subject])
            topic_count, topic_acc, topic_time_std = _summary(topic_stats[topic_key])
            _, difficulty_acc, _ = _summary(difficulty_stats[diff_key])
            history = topic_recent[topic_key]
            recent5 = float(np.mean(list(history)[-5:])) if history else 0.5
            recent10 = float(np.mean(history)) if history else 0.5
            older5 = list(history)[:-5]
            trend = recent5 - float(np.mean(older5)) if older5 else 0.0
            topic_avg_time = topic_stats[topic_key][2] / topic_count if topic_count else 0.0
            previous = float(history[-1]) if history else 0.5
            last_time = topic_last_time.get(topic_key)
            elapsed = float((event.timestamp - last_time).total_seconds() / 3600) if last_time is not None else 168.0
            rows.append({
                "student_id": student_id, "timestamp": event.timestamp.isoformat(), "attempt_number": event.attempt_number,
                "subject": event.subject, "topic": event.topic, "difficulty": event.difficulty,
                "prior_attempts": prior_count, "overall_accuracy": overall_acc,
                "subject_accuracy": subject_acc, "topic_accuracy": topic_acc,
                "recent_accuracy_5": recent5, "recent_accuracy_10": recent10,
                "topic_avg_response_time": topic_avg_time, "topic_response_time_std": topic_time_std,
                "difficulty_accuracy": difficulty_acc, "previous_correct": previous,
                "improvement_trend": trend, "hours_since_topic_attempt": min(elapsed, 720.0),
                "mastered_label": int(event.synthetic_mastered_label),
                "synthetic_true_mastery": float(event.synthetic_true_mastery),
                "data_label": "SYNTHETIC / SIMULATED — NOT REAL STUDENT DATA",
            })
            correct, response_time = float(event.is_correct), float(event.time_taken_seconds)
            for stats in (overall, subject_stats[event.subject], topic_stats[topic_key], difficulty_stats[diff_key]):
                stats[0] += 1
                stats[1] += correct
                stats[2] += response_time
                stats[3] += response_time * response_time
            history.append(int(correct))
            topic_last_time[topic_key] = event.timestamp
    return pd.DataFrame(rows)


def current_topic_feature_rows(attempts: pd.DataFrame, student_id: str, topics: pd.DataFrame) -> pd.DataFrame:
    """Build inference records from all existing history, without any hidden truth."""
    events = attempts[attempts.student_id == student_id].copy()
    if events.empty:
        raise ValueError(f"No attempts available for {student_id}")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    now = events.timestamp.max()
    rows = []
    for item in topics[["subject", "topic"]].drop_duplicates().itertuples(index=False):
        history = events[(events.subject == item.subject) & (events.topic == item.topic)].sort_values("timestamp")
        subject_history = events[events.subject == item.subject]
        def accuracy(frame: pd.DataFrame, default: float = 0.5) -> float:
            return float(frame.is_correct.mean()) if len(frame) else default
        time_mean = float(history.time_taken_seconds.mean()) if len(history) else 0.0
        time_std = float(history.time_taken_seconds.std(ddof=0)) if len(history) else 0.0
        recent = history.is_correct.tail(10).tolist()
        last = pd.to_datetime(history.timestamp.iloc[-1], utc=True) if len(history) else now - pd.Timedelta(hours=168)
        desired = "Medium"
        if len(recent) >= 2 and sum(recent[-2:]) == 2:
            desired = "Hard"
        elif len(recent) >= 2 and sum(recent[-2:]) == 0:
            desired = "Easy"
        difficulty_history = history[history.difficulty == desired]
        older5 = recent[:-5]
        rows.append({
            "student_id": student_id, "subject": item.subject, "topic": item.topic, "difficulty": desired,
            "prior_attempts": float(len(events)), "overall_accuracy": accuracy(events),
            "subject_accuracy": accuracy(subject_history), "topic_accuracy": accuracy(history),
            "recent_accuracy_5": float(np.mean(recent[-5:])) if recent else 0.5,
            "recent_accuracy_10": float(np.mean(recent)) if recent else 0.5,
            "topic_avg_response_time": time_mean, "topic_response_time_std": time_std,
            "difficulty_accuracy": accuracy(difficulty_history), "previous_correct": float(recent[-1]) if recent else 0.5,
            "improvement_trend": (float(np.mean(recent[-5:])) - float(np.mean(older5))) if older5 else 0.0,
            "hours_since_topic_attempt": min(float((now - last).total_seconds() / 3600), 720.0),
        })
    return pd.DataFrame(rows)
