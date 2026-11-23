"""How well is the model doing in real use? Everything here is computed from
the prediction log: one row per saved expense, holding the model's guess and
the user's final choice."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import pandas as pd

from src import config

RETRAIN_MIN_ROWS = 20   # don't judge the model on fewer predictions than this
RETRAIN_DROP = 0.10     # 10 percentage points below the test accuracy


def weekly_metrics(log: pd.DataFrame, threshold: float = config.CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """One row per week (weeks start on Monday). The user's corrections are the
    ground truth: accuracy = 1 - share of predictions the user corrected."""
    if log.empty:
        return pd.DataFrame(columns=["week", "predictions", "corrected_share", "accuracy",
                                     "avg_confidence", "needed_review_share"])
    weeks = log["timestamp"].dt.to_period("W-SUN").dt.start_time.rename("week")
    out = log.groupby(weeks).agg(
        predictions=("id", "size"),
        corrected_share=("corrected", "mean"),
        avg_confidence=("confidence", "mean"),
        needed_review_share=("confidence", lambda c: (c < threshold).mean()),
    ).reset_index()
    out.insert(3, "accuracy", 1 - out["corrected_share"])
    return out


def category_mix(log: pd.DataFrame, training_counts: dict[str, int]) -> pd.DataFrame:
    """Share of each category among live predictions vs in the training data.
    A big difference means the student's spending doesn't look like the training data."""
    training = pd.Series(training_counts, dtype=float)
    live = log["predicted"].value_counts().reindex(config.CATEGORIES, fill_value=0)
    return pd.DataFrame({
        "predicted (live)": live / max(live.sum(), 1),
        "training data": (training / training.sum()).reindex(config.CATEGORIES, fill_value=0),
    })


def recent_accuracy(log: pd.DataFrame, window: int = RETRAIN_MIN_ROWS) -> float | None:
    """Accuracy over the most recent `window` predictions, or None if there are fewer."""
    if len(log) < window:
        return None
    return float(1 - log.sort_values("timestamp")["corrected"].tail(window).mean())


def retrain_recommended(log: pd.DataFrame, test_accuracy: float) -> tuple[bool, float | None]:
    """The retrain rule: recent accuracy (last 20+ predictions) is 10 or more
    points below the accuracy the model had on the held-out test split."""
    recent = recent_accuracy(log)
    if recent is None:
        return False, None
    return recent <= test_accuracy - RETRAIN_DROP, recent
