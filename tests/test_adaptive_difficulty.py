import pandas as pd

from recommendation.adaptive_difficulty import TARGET_BAND, next_difficulty, target_difficulty_rating


def test_adaptive_difficulty_moves_up_after_two_successes():
    history = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "difficulty": ["Easy", "Easy"], "is_correct": [1, 1]})
    assert next_difficulty(history, 0.70) == "Medium"


def test_adaptive_difficulty_moves_down_after_two_failures():
    history = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "difficulty": ["Hard", "Hard"], "is_correct": [0, 0]})
    assert next_difficulty(history, 0.30) == "Medium"


def test_target_rating_blends_level_with_mastery_and_stays_in_range():
    history = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "difficulty": ["Medium", "Medium"], "is_correct": [1, 1]})
    target = target_difficulty_rating(history, 0.70)
    assert 0.1 <= target <= 1.0
    # Two consecutive successes promote the ladder to Hard (0.85); the blend keeps
    # the target between the promoted level and the 0.70 mastery estimate.
    assert 0.70 <= target <= 0.85


def test_target_rating_rises_with_mastery_on_sparse_history():
    empty = pd.DataFrame(columns=["timestamp", "difficulty", "is_correct"])
    low = target_difficulty_rating(empty, 0.30)
    high = target_difficulty_rating(empty, 0.90)
    assert high - low >= TARGET_BAND

