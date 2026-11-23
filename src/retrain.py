"""Retrain the model with the user's confirmed and corrected expenses added.

Run from the repo root:  python -m src.retrain
The new model replaces the current one only if macro-F1 does not drop."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import csv
from datetime import datetime

import pandas as pd

from src import config, data, db
from src import train as T
from src.features import INPUT_COLUMNS
from src.predict import load_model

HISTORY_PATH = config.MODELS_DIR / "retrain_history.csv"
MIN_LOGGED_ROWS = 10  # with fewer, the held-out 20% would be one or two rows


def next_version(version: str) -> str:
    return f"v{int(version.lstrip('v')) + 1}"


def evaluate(model, sets: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Macro-F1 on each evaluation set."""
    return {name: T.scores(df["category"], model.predict(df[INPUT_COLUMNS]))["macro_f1"]
            for name, df in sets.items()}


def append_history(row: dict) -> None:
    new_file = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def split_logged(logged: pd.DataFrame, default_amount: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest 20% of logged expenses are held out for evaluation, never trained on."""
    logged = logged.assign(amount=logged["amount"].fillna(default_amount))
    return data.time_split(logged, test_fraction=0.2)


def retrain(conn=None) -> dict:
    conn = conn or db.connect()
    old_model, metadata = load_model()
    logged = db.logged_training_rows(conn)
    if len(logged) < MIN_LOGGED_ROWS:
        return {"kept": False, "reason": f"only {len(logged)} logged expenses; need {MIN_LOGGED_ROWS}"}

    logged_train, logged_holdout = split_logged(logged, metadata["default_amount"])
    train, test = data.time_split(data.load_source("household"))
    combined_train = pd.concat([train, logged_train], ignore_index=True)
    new_model = T.make_model(metadata["model_name"], **metadata["settings"])
    new_model.fit(combined_train[INPUT_COLUMNS], combined_train["category"])

    sets = {"test": test, "logged_holdout": logged_holdout,
            "combined": pd.concat([test, logged_holdout], ignore_index=True)}
    old_f1, new_f1 = evaluate(old_model, sets), evaluate(new_model, sets)
    # One decision number: macro-F1 on test + held-out logs together must not drop.
    kept = new_f1["combined"] >= old_f1["combined"]
    new_version = next_version(metadata["model_version"]) if kept else metadata["model_version"]
    if kept:
        save_new_model(new_model, metadata, combined_train, test, new_version)

    summary = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "old_version": metadata["model_version"], "new_version": new_version,
               "logged_rows_added": len(logged_train), "logged_rows_held_out": len(logged_holdout)}
    summary |= {f"old_{k}_macro_f1": round(v, 4) for k, v in old_f1.items()}
    summary |= {f"new_{k}_macro_f1": round(v, 4) for k, v in new_f1.items()}
    summary["kept"] = kept
    append_history(summary)
    return summary


def save_new_model(model, old_metadata: dict, combined_train: pd.DataFrame,
                   test: pd.DataFrame, version: str) -> None:
    """Same settings as before; CV score is carried over because we did not re-tune."""
    params = {"clf__C": old_metadata["settings"]["C"],
              "clf__class_weight": old_metadata["settings"]["class_weight"]}
    metadata = T.build_metadata(old_metadata["model_name"], params, combined_train,
                                old_metadata["cv_macro_f1"], version)
    test_scores = T.scores(test["category"], model.predict(test[INPUT_COLUMNS]))
    metadata |= {"test_accuracy": round(test_scores["accuracy"], 4),
                 "test_macro_f1": round(test_scores["macro_f1"], 4),
                 "confidence_threshold": old_metadata.get("confidence_threshold"),
                 "default_amount": old_metadata["default_amount"]}
    T.save_model(model, metadata)


if __name__ == "__main__":
    for key, value in retrain().items():
        print(f"{key}: {value}")
