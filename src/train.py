"""Building, tuning, scoring and saving models. Used by notebook 03 and by
retrain.py, so a retrained model is built exactly like the original."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config, data
from src.features import INPUT_COLUMNS, NUMERIC_NAMES, embedding_features, tfidf_features

MODEL_PATH = config.MODELS_DIR / "model.joblib"
METADATA_PATH = config.MODELS_DIR / "metadata.json"
CANDIDATES_PATH = config.MODELS_DIR / "candidates.joblib"

FEATURE_SETS = {"tfidf_logreg": tfidf_features, "minilm_logreg": embedding_features}
FEATURE_DESCRIPTIONS = {
    "tfidf_logreg": ["tfidf(clean description, 1-2 grams, min_df=2)"] + NUMERIC_NAMES,
    "minilm_logreg": ["minilm_embedding(clean description, 384 dims)"] + NUMERIC_NAMES,
}
PARAM_GRID = {"clf__C": [0.1, 1, 10], "clf__class_weight": [None, "balanced"]}


def make_model(name: str, C: float = 1.0, class_weight: str | None = None) -> Pipeline:
    """Features -> Logistic Regression, in one Pipeline object."""
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000,
                             random_state=config.RANDOM_SEED)
    return Pipeline([("features", FEATURE_SETS[name]()), ("clf", clf)])


def make_baseline() -> Pipeline:
    """Always predicts the most common training category. Any real model must beat this."""
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])


def cv_splitter() -> StratifiedKFold:
    # Stratified so each fold keeps a few of the rare Education rows.
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)


def tune(name: str, train: pd.DataFrame) -> GridSearchCV:
    """5-fold CV over C and class_weight on the TRAINING data only, scored by macro-F1."""
    search = GridSearchCV(make_model(name), PARAM_GRID, scoring={"macro_f1": "f1_macro",
                          "accuracy": "accuracy"}, refit="macro_f1", cv=cv_splitter())
    return search.fit(train[INPUT_COLUMNS], train["category"])


def macro_f1(y_true, y_pred) -> float:
    """Average F1 over the categories that actually occur in y_true. A test set
    with no Education rows (e.g. personal finance) should not score 0 for Education."""
    present = [c for c in config.CATEGORIES if c in set(y_true)]
    return float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))


def scores(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": macro_f1(y_true, y_pred)}


def build_metadata(name: str, params: dict, train: pd.DataFrame, cv_macro_f1: float,
                   version: str) -> dict:
    return {
        "model_name": name,
        "model_version": version,
        "features": FEATURE_DESCRIPTIONS[name],
        "settings": {"C": params["clf__C"], "class_weight": params["clf__class_weight"]},
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "train_rows": int(len(train)),
        "label_distribution": {c: int((train["category"] == c).sum()) for c in config.CATEGORIES},
        "cv_macro_f1": round(float(cv_macro_f1), 4),
        "default_amount": float(train["amount"].median()),
        # A model pickled by one scikit-learn version may not load in another,
        # so the app compares this with the installed version before loading.
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_model(model: Pipeline, metadata: dict) -> None:
    """Save as the current model, plus a versioned copy so old versions are kept."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model, config.MODELS_DIR / f"model_{metadata['model_version']}.joblib")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def update_metadata(**fields) -> dict:
    """Add results computed later (e.g. test accuracy from notebook 04)."""
    metadata = load_metadata() | fields
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def train_and_save(version: str = "v1") -> dict:
    """Notebook 03's training in one call: tune both models with 5-fold CV,
    keep the better one, score it on the test split and save it.
    Used by the app's "Build model" button and by `python -m src.train`."""
    train, test = data.time_split(data.load_source("household"))
    searches = {name: tune(name, train) for name in FEATURE_SETS}
    name = max(searches, key=lambda n: searches[n].best_score_)
    best = searches[name]
    metadata = build_metadata(name, best.best_params_, train, best.best_score_, version)
    test_scores = scores(test["category"], best.best_estimator_.predict(test[INPUT_COLUMNS]))
    metadata.update({
        "test_accuracy": round(test_scores["accuracy"], 4),
        "test_macro_f1": round(test_scores["macro_f1"], 4),
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "test_date_range": [str(test["date"].min().date()), str(test["date"].max().date())],
    })
    save_model(best.best_estimator_, metadata)
    return metadata


if __name__ == "__main__":
    for key, value in train_and_save().items():
        print(f"{key}: {value}")
