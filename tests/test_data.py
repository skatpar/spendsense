# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import pandas as pd

from src import config, data


def test_time_split_never_puts_test_before_train():
    household = data.load_source("household")
    train, test = data.time_split(household, test_fraction=0.2)
    assert len(train) + len(test) == len(household)
    assert test["date"].min() > train["date"].max()
    assert abs(len(test) / len(household) - 0.2) < 0.02


def test_time_split_keeps_rows_with_the_same_date_together():
    dates = pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 4 + ["2026-01-03"] * 2)
    df = pd.DataFrame({"date": dates, "description": "x", "category": "Food"})
    train, test = data.time_split(df, test_fraction=0.3)
    assert test["date"].min() > train["date"].max()
    # The 30% cut falls inside 2026-01-02, so all four rows of that day go to test.
    assert len(test) == 6
    assert set(test["date"]) == {pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")}


def test_every_mapped_value_is_a_spendsense_category():
    cmap = data.load_category_map()
    assert set(cmap["category"]) <= set(config.CATEGORIES)
    for source in ["household", "personal_finance", "own"]:
        assert set(data.load_source(source)["category"]) <= set(config.CATEGORIES)


def test_exact_subcategory_match_beats_wildcard():
    cmap = data.load_category_map()
    rows = pd.DataFrame({"source": ["household", "household"], "source_category": ["Food", "Food"],
                         "source_subcategory": ["Milk", "Lunch"], "description": ["a", "b"]})
    mapped = data.map_categories(rows, cmap)
    assert mapped["category"].tolist() == ["Groceries", "Food"]


def test_income_is_never_loaded():
    household = data.load_household()
    raw = pd.read_csv(config.HOUSEHOLD_CSV)
    assert len(household) == (raw["Income/Expense"] == "Expense").sum()
