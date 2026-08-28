"""Generate latent learner profiles with subject/topic correlations."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data.questions.question_bank import TOPIC_FACTS


ABILITY_BANDS = {
    "emerging": (0.27, 0.075),
    "developing": (0.47, 0.080),
    "proficient": (0.67, 0.075),
    "advanced": (0.84, 0.055),
}


def generate_students(
    number_of_students: int,
    seed: int = 42,
    ability_distribution: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    """Return labelled synthetic students and their hidden topic-level skills.

    Profiles are intentionally correlated: student-level ability shapes subjects,
    then subjects shape their topics. This is more realistic than independent
    random correct/incorrect labels.
    """
    if number_of_students < 1:
        raise ValueError("number_of_students must be at least 1")
    rng = np.random.default_rng(seed)
    bands = list(ABILITY_BANDS)
    distribution = ability_distribution or {"emerging": 0.18, "developing": 0.38, "proficient": 0.30, "advanced": 0.14}
    if set(distribution) != set(bands) or any(value < 0 for value in distribution.values()):
        raise ValueError(f"ability_distribution must contain non-negative weights for {bands}")
    weights = np.array([distribution[band] for band in bands], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("ability_distribution weights must sum to a positive value")
    weights /= weights.sum()
    students, profiles = [], {}
    created_at = datetime.now(timezone.utc).isoformat()
    for i in range(1, number_of_students + 1):
        band = str(rng.choice(bands, p=weights))
        mean, std = ABILITY_BANDS[band]
        general = float(np.clip(rng.normal(mean, std), 0.10, 0.95))
        profile: dict[str, dict[str, float]] = {}
        for subject, concepts in TOPIC_FACTS.items():
            # Subject factors create clear strengths and weaknesses.
            subject_skill = float(np.clip(general + rng.normal(0, 0.16), 0.08, 0.96))
            profile[subject] = {
                topic: round(float(np.clip(subject_skill + rng.normal(0, 0.12), 0.05, 0.98)), 4)
                for topic, *_ in concepts
            }
        student_id = f"S{i:05d}"
        profiles[student_id] = profile
        students.append({
            "student_id": student_id,
            "display_name": f"Synthetic Student {i:05d}",
            "is_synthetic": True,
            "ability_band": band,
            "initial_general_ability": round(general, 4),
            "created_at": created_at,
            "latent_profile_json": json.dumps(profile, sort_keys=True),
            "data_label": "SYNTHETIC / SIMULATED — NOT REAL STUDENT DATA",
        })
    return pd.DataFrame(students), profiles
