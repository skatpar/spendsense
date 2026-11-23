"""Load the three data sources into one shared shape, map their categories
to the 8 SpendSense categories, clean them, and split by time.

Every source ends up with the same columns:
    date, description, amount, currency, category, source
"""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import pandas as pd

from src import config

COLUMNS = ["date", "description", "amount", "currency", "category", "source"]


def load_household() -> pd.DataFrame:
    """Daily Household Transactions (India, INR). Expenses only."""
    raw = pd.read_csv(config.HOUSEHOLD_CSV)
    # Income and transfers are not spending, so they never reach the model.
    raw = raw[raw["Income/Expense"] == "Expense"]
    return pd.DataFrame({
        # Some rows have a time, some don't; dayfirst because dates are dd/mm/yyyy.
        "date": pd.to_datetime(raw["Date"], format="mixed", dayfirst=True),
        "description": raw["Note"],
        "amount": raw["Amount"],
        "currency": raw["Currency"],
        "source_category": raw["Category"],
        "source_subcategory": raw["Subcategory"],
        "source": "household",
    })


def load_personal_finance() -> pd.DataFrame:
    """Personal Finance Transaction Analysis (USA, USD). Debits only."""
    raw = pd.read_csv(config.PERSONAL_FINANCE_CSV)
    # Credits are paychecks and refunds: money coming in, not an expense.
    raw = raw[raw["Transaction Type"] == "debit"]
    return pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%m/%d/%Y"),
        "description": raw["Description"],
        "amount": raw["Amount"],
        "currency": "USD",
        "source_category": raw["Category"],
        "source_subcategory": None,
        "source": "personal_finance",
    })


def load_own_expenses() -> pd.DataFrame:
    """The student's own hand-labelled expenses (Pakistan, PKR).
    These already use the 8 SpendSense categories, so no mapping is needed."""
    raw = pd.read_csv(config.OWN_EXPENSES_CSV)
    return pd.DataFrame({
        "date": pd.to_datetime(raw["date"], format="%Y-%m-%d"),
        "description": raw["description"],
        "amount": raw["amount_pkr"],
        "currency": "PKR",
        "category": raw["category"],
        "source": "own",
    })


def load_category_map() -> pd.DataFrame:
    return pd.read_csv(config.CATEGORY_MAP_CSV, dtype=str)


def map_categories(df: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """Give each row a SpendSense category. A row first looks for an exact
    (category, subcategory) match, then falls back to (category, '*').
    Rows with no match don't fit any of the 8 categories and are dropped."""
    exact = {(r.source, r.source_category, r.source_subcategory): r.category
             for r in cmap.itertuples()}

    def lookup(row) -> str | None:
        key = (row.source, row.source_category, row.source_subcategory)
        wildcard = (row.source, row.source_category, "*")
        return exact.get(key, exact.get(wildcard))

    out = df.copy()
    out["category"] = [lookup(r) for r in out.itertuples()]
    dropped = out["category"].isna()
    for source, n in out[dropped].groupby("source").size().items():
        print(f"[map] {source}: dropped {n} of {(out['source'] == source).sum()} rows with no matching category")
    return out[~dropped].drop(columns=["source_category", "source_subcategory"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the model cannot learn from: empty descriptions and exact duplicates."""
    out = df.copy()
    out["description"] = out["description"].astype("string").str.strip()
    empty = out["description"].isna() | (out["description"] == "")
    before = len(out)
    out = out[~empty]
    out = out.drop_duplicates(subset=COLUMNS)
    print(f"[clean] removed {empty.sum()} empty-description rows and "
          f"{before - empty.sum() - len(out)} exact duplicates")
    out["description"] = out["description"].astype(str)
    return out[COLUMNS].sort_values("date").reset_index(drop=True)


def load_source(name: str) -> pd.DataFrame:
    """Load one source ('household', 'personal_finance' or 'own'), fully cleaned."""
    if name == "own":
        return clean(load_own_expenses())
    loader = {"household": load_household, "personal_finance": load_personal_finance}[name]
    return clean(map_categories(loader(), load_category_map()))


def time_split(df: pd.DataFrame, test_fraction: float = config.TEST_FRACTION
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest dates -> train, latest dates -> test. We never shuffle across
    time because the real app always predicts *future* expenses from past ones.
    All rows sharing the cut-off moment go to test, so no test row is ever
    earlier than a training row."""
    ordered = df.sort_values("date").reset_index(drop=True)
    cut_index = int(len(ordered) * (1 - test_fraction))
    cutoff = ordered["date"].iloc[cut_index]
    train = ordered[ordered["date"] < cutoff].reset_index(drop=True)
    test = ordered[ordered["date"] >= cutoff].reset_index(drop=True)
    return train, test
