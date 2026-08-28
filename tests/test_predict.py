import pandas as pd

from ml.predict import explain_weakness


def test_unattempted_topic_explanation_does_not_describe_default_accuracy_as_observed():
    row = pd.Series({
        "prior_attempts": 7,
        "topic_prior_attempts": 0,
        "recent_accuracy_5": 0.5,
        "topic_avg_response_time": 0.0,
        "difficulty_accuracy": 0.5,
        "improvement_trend": 0.0,
        "topic": "Graphs",
        "difficulty": "Medium",
    })

    explanation = explain_weakness(row)

    assert explanation["reasons"] == ["no attempts yet on this topic — estimate uses the model's prior"]


def test_attempted_topic_can_report_observed_recent_accuracy():
    row = pd.Series({
        "prior_attempts": 7,
        "topic_prior_attempts": 1,
        "recent_accuracy_5": 0.5,
        "topic_avg_response_time": 0.0,
        "difficulty_accuracy": 0.5,
        "improvement_trend": 0.0,
        "topic": "Graphs",
        "difficulty": "Medium",
    })

    assert "recent accuracy is 50%" in explain_weakness(row)["reasons"]
