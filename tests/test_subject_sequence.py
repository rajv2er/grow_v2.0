import pandas as pd

from app.main import MIN_QUESTIONS_PER_SUBJECT, SUBJECT_MASTERY_THRESHOLD, subject_ready_to_advance


def test_subject_advances_after_minimum_questions():
    attempts = pd.DataFrame({"subject": ["DSA"] * MIN_QUESTIONS_PER_SUBJECT})

    assert subject_ready_to_advance(attempts, "DSA", pd.DataFrame())


def test_subject_advances_when_all_subject_topics_are_mastered():
    attempts = pd.DataFrame({"subject": ["DSA"]})
    predictions = pd.DataFrame({"mastery_probability": [SUBJECT_MASTERY_THRESHOLD, 0.88]})

    assert subject_ready_to_advance(attempts, "DSA", predictions)
