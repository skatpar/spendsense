"""Project-wide settings. Every other module imports from here so that
paths, categories and the random seed are defined in exactly one place."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import os
from pathlib import Path

# One seed for everything, so anyone re-running the code gets the same numbers.
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RECEIPTS_DIR = DATA_DIR / "receipts"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = DATA_DIR / "cache"

HOUSEHOLD_CSV = RAW_DIR / "daily_household_transactions.csv"
PERSONAL_FINANCE_CSV = RAW_DIR / "personal_finance_transactions.csv"
OWN_EXPENSES_CSV = DATA_DIR / "own_expenses.csv"
CATEGORY_MAP_CSV = DATA_DIR / "category_map.csv"
RECEIPT_LABELS_CSV = RECEIPTS_DIR / "labels.csv"
# An environment variable lets tests and demos use a separate database file.
DB_PATH = Path(os.environ.get("SPENDSENSE_DB", DATA_DIR / "spendsense.db"))

CATEGORIES = [
    "Food", "Groceries", "Transport", "Shopping",
    "Bills", "Health", "Education", "Entertainment",
]

# Short descriptions used by the zero-shot classifier. The examples matter:
# MiniLM compares meaning, so concrete words a student would actually type
# give the model something to match against.
CATEGORY_DESCRIPTIONS = {
    "Food": "Food: restaurant, lunch, dinner, snacks, tea, coffee, fast food, eating out",
    "Groceries": "Groceries: grocery store, milk, eggs, vegetables, fruit, bread, flour, supermarket",
    "Transport": "Transport: rickshaw, bus, train, taxi, fuel, petrol, ride-hailing, careem, uber",
    "Shopping": "Shopping: clothes, shoes, electronics, amazon, household items, appliances",
    "Bills": "Bills: electricity, gas, water, internet, mobile recharge, rent, subscription",
    "Health": "Health: medicine, pharmacy, doctor, hospital, lab test, clinic",
    "Education": "Education: books, stationery, course fee, exam fee, tuition, university",
    "Entertainment": "Entertainment: movie, cinema, netflix, music, games, outing, festival",
}

# The pre-trained embedding model. If the student has saved a local copy
# (useful when Hugging Face is blocked), use it; otherwise download by name.
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_LOCAL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"


def minilm_path() -> str:
    """Return the local model folder if present, else the Hugging Face name."""
    if (MINILM_LOCAL_DIR / "model.safetensors").exists():
        return str(MINILM_LOCAL_DIR)
    return MINILM_NAME


# Fixed colour per category, so "Food" is the same blue in every chart and
# in the app. Order follows a colour-blind-checked palette.
CATEGORY_COLORS = {
    "Food": "#2a78d6", "Groceries": "#eb6834", "Transport": "#1baf7a", "Shopping": "#eda100",
    "Bills": "#e87ba4", "Health": "#008300", "Education": "#6250d6", "Entertainment": "#e34948",
}
SOURCE_COLORS = {"household": "#2a78d6", "personal_finance": "#eb6834", "own": "#1baf7a"}

TEST_FRACTION = 0.2

# Filled in after notebook 04 picks it from the accuracy-vs-confidence plot.
CONFIDENCE_THRESHOLD = 0.6
