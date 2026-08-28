from ml.subjective_grading import grade_subjective, rubric_score


QUESTION = {
    "question_id": "Q0101S", "subject": "DSA", "topic": "Arrays",
    "question_type": "Subjective",
    "model_answer": "Array elements are stored at contiguous memory locations.",
}


def test_answer_covering_key_terms_passes():
    result = grade_subjective(QUESTION, "Arrays keep their elements in contiguous memory locations, so indexing is direct.")
    assert result["is_correct"] and result["score"] >= 0.6
    assert "contiguous" in result["matched"] and "memory" in result["matched"]


def test_unrelated_answer_fails_and_lists_missing_terms():
    result = grade_subjective(QUESTION, "A linked list stores nodes anywhere in memory with pointers.")
    assert not result["is_correct"]
    assert "contiguous" in result["missing"]


def test_very_short_answer_is_capped_below_pass_threshold():
    # "arrays" prefix-matches the model term "array", but the short-answer cap
    # keeps the score well below the pass threshold.
    score, matched, missing = rubric_score("arrays", QUESTION["model_answer"])
    assert matched == ["array"]
    assert score <= 0.5


def test_plurals_and_inflections_count_as_matches():
    _, matched, _ = rubric_score("elements stored across contiguous locations", QUESTION["model_answer"])
    assert "elements" in matched and "stored" in matched


def test_llm_hook_overrides_rubric_score():
    result = grade_subjective(QUESTION, "Arrays.", llm=lambda answer, model: 0.9)
    assert result["score"] == 0.9 and result["source"] == "llm"


def test_llm_hook_failure_falls_back_to_rubric():
    def broken(answer, model):
        raise RuntimeError("provider unavailable")
    full_answer = "Array elements are stored at contiguous memory locations."
    result = grade_subjective(QUESTION, full_answer, llm=broken)
    assert result["source"] == "rubric" and result["is_correct"]
