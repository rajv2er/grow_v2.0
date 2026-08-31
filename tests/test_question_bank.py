from data.questions.question_bank import build_question_bank


def test_question_bank_has_three_difficulties_plus_subjective_for_every_topic():
    questions = build_question_bank()
    mcqs = [q for q in questions if q["question_type"] == "MCQ"]
    subjective = [q for q in questions if q["question_type"] == "Subjective"]
    assert len(questions) == 300, f"Expected 300 (200 base + 100 extended), got {len(questions)}"
    assert len(mcqs) == 160 and len(subjective) == 60
    assert len({q["question_type"] for q in questions}) == 6
    assert len({q["subject"] for q in questions}) == 5
    assert {q["difficulty"] for q in questions} == {"Easy", "Medium", "Hard"}
    assert all(q["correct_answer"] in {"A", "B", "C", "D"} for q in mcqs)
    by_topic = {}
    for question in mcqs:
        by_topic.setdefault((question["subject"], question["topic"]), set()).add(question["difficulty"])
    assert all(levels == {"Easy", "Medium", "Hard"} for levels in by_topic.values())
    for difficulty in ("Easy", "Medium", "Hard"):
        assert len({q["correct_answer"] for q in mcqs if q["difficulty"] == difficulty}) == 4
    extended_types = {"TrueFalse", "MultipleSelect", "FillInBlank", "Numerical"}
    extended = [q for q in questions if q["question_type"] in extended_types]
    assert len(extended) == 80
    for t in extended_types:
        assert sum(1 for q in extended if q["question_type"] == t) == 20


def test_every_question_carries_a_numeric_rating_in_range():
    for q in build_question_bank():
        assert 0.1 <= q["difficulty_rating"] <= 1.0, q["question_id"]


def test_band_has_multiple_distinct_ratings_per_topic():
    questions = build_question_bank()
    by_topic = {}
    for q in questions:
        by_topic.setdefault((q["subject"], q["topic"]), set()).add(q["difficulty_rating"])
    assert all(len(ratings) >= 3 for ratings in by_topic.values())


def test_subjective_items_have_model_answers_and_no_options():
    for q in build_question_bank():
        if q["question_type"] != "Subjective":
            continue
        assert q["model_answer"] and len(q["model_answer"]) > 10
        assert q["correct_answer"] is None
        assert q["option_a"] is None and q["option_d"] is None
        assert q["question_id"].endswith("S")
