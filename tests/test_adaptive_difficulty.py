import pandas as pd

from recommendation.adaptive_difficulty import next_difficulty


def test_adaptive_difficulty_moves_up_after_two_successes():
    history = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "difficulty": ["Easy", "Easy"], "is_correct": [1, 1]})
    assert next_difficulty(history, 0.70) == "Medium"


def test_adaptive_difficulty_moves_down_after_two_failures():
    history = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "difficulty": ["Hard", "Hard"], "is_correct": [0, 0]})
    assert next_difficulty(history, 0.30) == "Medium"
