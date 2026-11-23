"""Predict the category of one expense with the saved model."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import json
from datetime import date as date_type
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

from src import config

MODEL_PATH = config.MODELS_DIR / "model.joblib"
METADATA_PATH = config.MODELS_DIR / "metadata.json"


@lru_cache(maxsize=2)
def _load(mtime: float) -> tuple:
    """Cached by the file's modification time: loading happens once, but a
    retrained model (new file, new mtime) is picked up automatically."""
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text())


def load_model() -> tuple:
    """Return (pipeline, metadata) for the current model."""
    return _load(MODEL_PATH.stat().st_mtime)


def model_problem() -> str | None:
    """Return a plain-English reason the saved model can't be used, or None if it's fine."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return "No trained model was found in the models folder."
    metadata = json.loads(METADATA_PATH.read_text())
    saved = metadata.get("sklearn_version")
    if saved != sklearn.__version__:
        built_with = f"scikit-learn {saved}" if saved else "a different scikit-learn version"
        return (f"The saved model was built with {built_with}, but this computer has "
                f"scikit-learn {sklearn.__version__}, so it needs to be rebuilt here.")
    if "test_accuracy" not in metadata:
        return "The model has not been evaluated yet (run notebook 04, or rebuild it here)."
    try:
        load_model()
    except Exception as error:  # a broken or half-written file
        return f"The saved model could not be loaded ({type(error).__name__})."
    return None


def model_version() -> str:
    return load_model()[1]["model_version"]


def predict(description: str, amount: float | None = None, date: str | None = None) -> dict:
    """Return {"category", "confidence", "top3", "needs_review"}.

    Defaults for missing inputs:
    - amount: the median amount of the training data (stored in metadata.json),
      i.e. a "typical" expense, so a missing amount pushes towards no category.
    - date: today, because an expense is usually entered on the day it happens.
    """
    model, metadata = load_model()
    row = pd.DataFrame([{
        "description": description,
        "amount": metadata["default_amount"] if amount is None else float(amount),
        "date": pd.Timestamp(date) if date else pd.Timestamp(date_type.today()),
    }])
    probabilities = model.predict_proba(row)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
    category, confidence = ranked[0]
    return {
        "category": str(category),
        "confidence": float(confidence),
        "top3": [{"category": str(c), "confidence": float(p)} for c, p in ranked[:3]],
        "needs_review": bool(confidence < config.CONFIDENCE_THRESHOLD),
    }
