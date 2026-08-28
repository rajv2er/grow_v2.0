from data.questions.question_bank import build_question_bank


def test_question_bank_has_three_difficulties_for_every_topic():
    questions = build_question_bank()
    assert len(questions) == 150
    assert len({q["subject"] for q in questions}) == 5
    assert {q["difficulty"] for q in questions} == {"Easy", "Medium", "Hard"}
    assert all(q["correct_answer"] in {"A", "B", "C", "D"} for q in questions)
    by_topic = {}
    for question in questions:
        by_topic.setdefault((question["subject"], question["topic"]), set()).add(question["difficulty"])
    assert all(levels == {"Easy", "Medium", "Hard"} for levels in by_topic.values())
