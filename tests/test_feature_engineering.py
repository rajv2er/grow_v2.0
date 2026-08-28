import pandas as pd

from ml.feature_engineering import build_feature_dataset


def test_features_use_only_prior_history_not_future_attempts():
    attempts = pd.DataFrame([
        {"student_id": "S1", "question_id": "Q1", "subject": "DSA", "topic": "Graphs", "difficulty": "Easy", "is_correct": 1, "time_taken_seconds": 40, "attempt_number": 1, "timestamp": "2025-01-01T00:00:00+00:00"},
        {"student_id": "S1", "question_id": "Q2", "subject": "DSA", "topic": "Graphs", "difficulty": "Easy", "is_correct": 0, "time_taken_seconds": 90, "attempt_number": 2, "timestamp": "2025-01-02T00:00:00+00:00"},
    ])
    truth = pd.DataFrame([
        {"student_id": "S1", "attempt_number": 1, "timestamp": "2025-01-01T00:00:00+00:00", "synthetic_mastered_label": 0, "synthetic_true_mastery": 0.4},
        {"student_id": "S1", "attempt_number": 2, "timestamp": "2025-01-02T00:00:00+00:00", "synthetic_mastered_label": 0, "synthetic_true_mastery": 0.4},
    ])
    features = build_feature_dataset(attempts, truth)
    assert features.loc[0, "topic_accuracy"] == 0.5  # no prior evidence
    assert features.loc[1, "topic_accuracy"] == 1.0  # first response only; never future response
