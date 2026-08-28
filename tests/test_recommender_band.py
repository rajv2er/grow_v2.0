import pandas as pd

from data.questions.question_bank import build_question_bank
from database.db import connection, initialise_database, seed_question_bank
from recommendation.adaptive_difficulty import TARGET_BAND
from recommendation.recommender import recommend_questions


def _db_with_student(tmp_path, student_id="S1"):
    db = tmp_path / "rec_test.db"
    initialise_database(db)
    seed_question_bank(build_question_bank(), db)
    with connection(db) as conn:
        conn.execute(
            "INSERT INTO students(student_id, display_name, is_synthetic, created_at) VALUES (?, 'Test', 1, '2026-01-01')",
            (student_id,),
        )
    return db


def _questions():
    return pd.DataFrame(build_question_bank())


def test_selected_rating_lands_inside_target_band_when_possible(tmp_path):
    db = _db_with_student(tmp_path)
    q = _questions()
    preds = pd.DataFrame([
        {"subject": "DSA", "topic": topic, "mastery_probability": mastery, "status": "Developing"}
        for topic, mastery in [("Arrays", 0.30), ("Trees", 0.85)]
    ])
    attempts = pd.DataFrame([
        {"student_id": "S1", "question_id": "Q0101A", "subject": "DSA", "topic": "Arrays", "difficulty": "Medium", "is_correct": 0, "timestamp": "2026-01-01T10:00:00"},
        {"student_id": "S1", "question_id": "Q0101B", "subject": "DSA", "topic": "Arrays", "difficulty": "Medium", "is_correct": 0, "timestamp": "2026-01-01T10:05:00"},
        {"student_id": "S1", "question_id": "Q0106C", "subject": "DSA", "topic": "Trees", "difficulty": "Hard", "is_correct": 1, "timestamp": "2026-01-01T10:10:00"},
        {"student_id": "S1", "question_id": "Q0106B", "subject": "DSA", "topic": "Trees", "difficulty": "Hard", "is_correct": 1, "timestamp": "2026-01-01T10:15:00"},
    ])
    result = recommend_questions("S1", preds, attempts, q, limit=2, db_path=db)
    assert len(result) == 2
    for row in result.itertuples(index=False):
        assert abs(row.difficulty_rating - row.target_rating) <= TARGET_BAND + 1e-9, row.question_id
        assert row.in_band
    by_topic = result.set_index("topic")
    # Two consecutive failures pull the weak topic's target down; two successes keep Trees hard.
    assert by_topic.loc["Arrays", "target_rating"] < by_topic.loc["Trees", "target_rating"]
    assert by_topic.loc["Arrays", "difficulty_rating"] < by_topic.loc["Trees", "difficulty_rating"]


def test_weakest_topic_ranks_first(tmp_path):
    db = _db_with_student(tmp_path)
    q = _questions()
    preds = pd.DataFrame([
        {"subject": "DSA", "topic": "Graphs", "mastery_probability": 0.20, "status": "Weak"},
        {"subject": "DSA", "topic": "Hashing", "mastery_probability": 0.90, "status": "Strong"},
    ])
    attempts = pd.DataFrame(columns=["student_id", "question_id", "subject", "topic", "difficulty", "is_correct", "timestamp"])
    result = recommend_questions("S1", preds, attempts, q, limit=2, db_path=db)
    assert result.iloc[0].topic == "Graphs"


def test_recommendations_table_uses_snapshot_semantics(tmp_path):
    db = _db_with_student(tmp_path)
    q = _questions()
    preds = pd.DataFrame([{"subject": "DSA", "topic": "Graphs", "mastery_probability": 0.20, "status": "Weak"}])
    attempts = pd.DataFrame(columns=["student_id", "question_id", "subject", "topic", "difficulty", "is_correct", "timestamp"])
    recommend_questions("S1", preds, attempts, q, limit=1, db_path=db)
    recommend_questions("S1", preds, attempts, q, limit=1, db_path=db)
    with connection(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM recommendations WHERE student_id='S1'").fetchone()[0]
    assert count == 1


def test_spaced_revision_prefers_in_band_seen_item_over_off_band_unseen(tmp_path):
    db = _db_with_student(tmp_path)
    q = _questions()
    preds = pd.DataFrame([{"subject": "DSA", "topic": "Trees", "mastery_probability": 0.85, "status": "Strong"}])
    # The only Hard Trees item was already answered — it should still be served
    # (spaced revision) because it is the sole in-band option.
    attempts = pd.DataFrame([
        {"student_id": "S1", "question_id": "Q0106C", "subject": "DSA", "topic": "Trees", "difficulty": "Hard", "is_correct": 1, "timestamp": "2026-01-01T10:00:00"},
    ])
    result = recommend_questions("S1", preds, attempts, q, limit=1, db_path=db)
    assert result.iloc[0].question_id == "Q0106C"
    assert result.iloc[0].in_band


def test_exclude_forces_a_different_item_even_off_band(tmp_path):
    db = _db_with_student(tmp_path)
    q = _questions()
    preds = pd.DataFrame([{"subject": "DSA", "topic": "Trees", "mastery_probability": 0.85, "status": "Strong"}])
    attempts = pd.DataFrame([
        {"student_id": "S1", "question_id": "Q0106C", "subject": "DSA", "topic": "Trees", "difficulty": "Hard", "is_correct": 1, "timestamp": "2026-01-01T10:00:00"},
    ])
    result = recommend_questions("S1", preds, attempts, q, limit=1, exclude={"Q0106C"}, db_path=db)
    assert result.iloc[0].question_id != "Q0106C"
    assert not result.iloc[0].in_band


def test_plan_merge_does_not_duplicate_question_type(tmp_path):
    """The plan already carries question_type; re-merging it from the bank must not suffix _x/_y."""
    db = _db_with_student(tmp_path)
    q = _questions()
    preds = pd.DataFrame([{"subject": "DSA", "topic": "Graphs", "mastery_probability": 0.20, "status": "Weak"}])
    attempts = pd.DataFrame(columns=["student_id", "question_id", "subject", "topic", "difficulty", "is_correct", "timestamp"])
    plan = recommend_questions("S1", preds, attempts, q, limit=3, db_path=db)
    question_columns = ["question_id", "question", "option_a", "option_b", "option_c", "option_d"]
    merged = plan.merge(q[question_columns], on="question_id", how="left")
    assert "question_type" in merged.columns
    assert "question_type_x" not in merged.columns and "question_type_y" not in merged.columns
