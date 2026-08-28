import pandas as pd

from config import SimulationConfig
from data.questions.question_bank import build_question_bank
from simulator.attempt_generator import generate_attempts
from simulator.student_generator import generate_students


def test_simulator_is_seeded_and_emits_required_fields():
    students, profiles = generate_students(10, seed=7)
    questions = pd.DataFrame(build_question_bank())
    config = SimulationConfig(number_of_students=10, attempts_per_student=12, random_seed=7)
    attempts, truth = generate_attempts(students, profiles, questions, config.attempts_per_student, config.learning_rate, config.noise_level, config.difficulty_distribution, config.random_seed, config.start_date)
    assert len(attempts) == 120
    assert len(truth) == len(attempts)
    assert {"student_id", "question_id", "subject", "topic", "difficulty", "is_correct", "time_taken_seconds", "attempt_number", "timestamp", "session_id"} <= set(attempts)
    assert attempts.is_synthetic.all()
    assert attempts.time_taken_seconds.gt(0).all()
