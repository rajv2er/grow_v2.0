"""Central, reproducible configuration for simulation and experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "experiments" / "results"
DATABASE_PATH = ROOT / "database" / "learning_system.db"


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters are saved with every simulation for reproducibility.

    All generated records are *synthetic* and must not be represented as real
    student data or evidence of student outcomes.
    """

    number_of_students: int = 30
    attempts_per_student: int = 60
    random_seed: int = 42
    learning_rate: float = 0.035
    noise_level: float = 0.08
    difficulty_distribution: dict[str, float] = field(
        default_factory=lambda: {"Easy": 0.40, "Medium": 0.40, "Hard": 0.20}
    )
    ability_distribution: dict[str, float] = field(
        default_factory=lambda: {"emerging": 0.18, "developing": 0.38, "proficient": 0.30, "advanced": 0.14}
    )
    start_date: str = "2025-01-01"

    def to_dict(self) -> dict:
        return asdict(self)
