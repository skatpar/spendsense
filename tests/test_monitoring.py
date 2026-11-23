# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import pandas as pd
import pytest

from src import db, monitoring


def fake_log(corrected: list[int], start: str = "2026-11-02") -> pd.DataFrame:
    n = len(corrected)
    return pd.DataFrame({
        "id": range(1, n + 1),
        "timestamp": pd.date_range(start, periods=n, freq="D"),
        "predicted": ["Food"] * n,
        "confidence": [0.9 if i % 2 == 0 else 0.4 for i in range(n)],
        "corrected": corrected,
    })


def test_weekly_accuracy_is_one_minus_corrected_share():
    # 2026-11-02 is a Monday: 7 rows in week 1, 3 rows in week 2
    log = fake_log([1, 0, 0, 0, 0, 0, 1] + [1, 1, 0])
    weekly = monitoring.weekly_metrics(log, threshold=0.6)
    assert weekly["predictions"].tolist() == [7, 3]
    assert weekly["accuracy"].tolist() == pytest.approx([5 / 7, 1 / 3])
    assert weekly["needed_review_share"].tolist() == pytest.approx([3 / 7, 2 / 3])


def test_retrain_rule():
    healthy = fake_log([0] * 18 + [1] * 2)       # 90% accurate
    drifted = fake_log([0] * 15 + [1] * 5)       # 75% accurate
    too_few = fake_log([1] * 19)
    assert monitoring.retrain_recommended(healthy, test_accuracy=0.9) == (False, 0.9)
    assert monitoring.retrain_recommended(drifted, test_accuracy=0.9) == (True, 0.75)
    assert monitoring.retrain_recommended(too_few, test_accuracy=0.9) == (False, None)


def test_saving_an_expense_writes_both_tables(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    expense = {"date": "2026-11-23", "description": "chai", "amount_pkr": 100.0, "category": "Food", "source": "manual"}
    prediction = {"category": "Groceries", "confidence": 0.55}
    expense_id = db.save_expense(conn, expense, prediction, "v1")
    log = db.read_log(conn)
    assert len(db.read_expenses(conn)) == 1 and len(log) == 1
    assert log.loc[0, "id"] == expense_id and log.loc[0, "corrected"] == 1
    db.delete_expense(conn, expense_id)
    assert db.read_expenses(conn).empty and len(db.read_log(conn)) == 1  # the log is kept
