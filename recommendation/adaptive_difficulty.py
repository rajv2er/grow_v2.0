"""Explicit, testable difficulty-transition policy (not UI-only logic)."""
from __future__ import annotations

import pandas as pd


ORDER = ["Easy", "Medium", "Hard"]


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
