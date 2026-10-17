"""Shared paths, label names and constants used across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_FRACTION = 0.1  # carved out of the official train split, stratified

# Order matches the ClassLabel ids of the KINNEWS release (label - 1 in the CSV).
LABELS = [
    "politics", "sport", "economy", "health", "entertainment", "history",
    "technology", "tourism", "culture", "fashion", "religion", "environment",
    "education", "relationship",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# Kinyarwanda names for the categories (shown in the web app).
KIN_LABELS = {
    "politics": "politiki", "sport": "imikino", "economy": "ubukungu",
    "health": "ubuzima", "entertainment": "imyidagaduro", "history": "amateka",
    "technology": "ikoranabuhanga", "tourism": "ubukerarugendo", "culture": "umuco",
    "fashion": "imideri", "religion": "iyobokamana", "environment": "ibidukikije",
    "education": "uburezi", "relationship": "urukundo",
}
