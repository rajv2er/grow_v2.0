"""Explicit, testable difficulty-transition policy (not UI-only logic)."""
from __future__ import annotations

import pandas as pd


ORDER = ["Easy", "Medium", "Hard"]
LEVEL_RATING = {"Easy": 0.25, "Medium": 0.55, "Hard": 0.85}
TARGET_BAND = 0.15


def next_difficulty(topic_attempts: pd.DataFrame, mastery_probability: float) -> str:
    """Choose a target difficulty from recent topic performance and mastery.

    Two consecutive correct answers permit a single-step increase. Two
    consecutive incorrect answers lower it. Sparse evidence begins at Easy.
    """
    if topic_attempts.empty:
        return "Easy"
    recent = topic_attempts.sort_values("timestamp").tail(3)
    current = str(recent.iloc[-1].difficulty)
    current_index = ORDER.index(current)
    last_two = recent.is_correct.tail(2).tolist()
    if len(last_two) == 2 and sum(last_two) == 2 and mastery_probability >= 0.55:
        return ORDER[min(current_index + 1, len(ORDER) - 1)]
    if len(last_two) == 2 and sum(last_two) == 0:
        return ORDER[max(current_index - 1, 0)]
    if mastery_probability < 0.40:
        return "Easy"
    if mastery_probability > 0.80:
        return "Hard"
    return current


def target_difficulty_rating(topic_attempts: pd.DataFrame, mastery_probability: float) -> float:
    """Numeric target in [0.1, 1.0]: the ladder's level blended with estimated mastery.

    Selection then looks for items within target ± TARGET_BAND, so adaptation is
    continuous instead of jumping across three coarse levels.
    """
    level = next_difficulty(topic_attempts, mastery_probability)
    blended = 0.5 * LEVEL_RATING[level] + 0.5 * mastery_probability
    return round(min(1.0, max(0.1, blended)), 2)
