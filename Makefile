.PHONY: help install test app demo train experiment clean lock

api:
	. .venv/bin/activate && uvicorn api.main:app --reload --port 8766

help:
	@echo "MasteryLab — common workflows"
	@echo "  make install     create venv and install pinned deps"
	@echo "  make test        run pytest"
	@echo "  make app         run the Streamlit UI"
	@echo "  make api         run the FastAPI service on :8766"
	@echo "  make demo        regenerate the synthetic cohort + train models"
	@echo "  make train       train all models on the current cohort"
	@echo "  make experiment  run the model-only comparison"
	@echo "  make lock        regenerate requirements-lock.txt"
	@echo "  make clean       remove caches and generated artefacts"

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements-lock.txt

test:
	. .venv/bin/activate && pytest -q

app:
	. .venv/bin/activate && streamlit run app/main.py

demo:
	. .venv/bin/activate && python run_demo.py

train:
	. .venv/bin/activate && python -c "from experiments.experiment_1 import run_experiment_from_data; from data.questions.question_bank import build_question_bank; from simulator.learning_simulator import SimulationConfig; from database.db import initialise_database; initialise_database(); import pandas as pd; from pathlib import Path; a = pd.read_csv('data/generated/attempts.csv'); t = pd.read_csv('data/generated/synthetic_truth.csv'); run_experiment_from_data(a, t)"

experiment:
	. .venv/bin/activate && python -m experiments.experiment_1

lock:
	. .venv/bin/activate && pip freeze > requirements-lock.txt

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	find . -name '*.pyc' -delete
