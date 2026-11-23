"""SQLite storage for the app: saved expenses and a log of every prediction.

A saved expense and its prediction-log row share the same id, which links
them without an extra column. Deleting an expense keeps its log row, because
the log is the record of how well the model did."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- AUTOINCREMENT: ids are never reused after a delete
    date TEXT, description TEXT, amount_pkr REAL, category TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY, timestamp TEXT, description TEXT, predicted TEXT,
    confidence REAL, final_category TEXT, corrected INTEGER, model_version TEXT);
"""


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def save_expense(conn: sqlite3.Connection, expense: dict, prediction: dict, model_version: str) -> int:
    """Write the expense and its prediction-log row in one transaction.
    `expense` has date, description, amount_pkr, category (the user's final choice) and source."""
    with conn:  # commits both inserts together, or neither
        cur = conn.execute(
            "INSERT INTO expenses (date, description, amount_pkr, category, source) VALUES (?, ?, ?, ?, ?)",
            (expense["date"], expense["description"], expense["amount_pkr"], expense["category"], expense["source"]))
        expense_id = cur.lastrowid
        conn.execute(
            "INSERT INTO prediction_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (expense_id, datetime.now().isoformat(timespec="seconds"), expense["description"],
             prediction["category"], prediction["confidence"], expense["category"],
             int(prediction["category"] != expense["category"]), model_version))
    return expense_id


def delete_expense(conn: sqlite3.Connection, expense_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))


def read_expenses(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM expenses ORDER BY date DESC, id DESC", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def read_log(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM prediction_log ORDER BY timestamp", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def logged_training_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every confirmed or corrected expense, in the same columns as data.py.
    The user's final category is the label. If the expense was deleted, the
    log row is still used, with the log date and an unknown amount."""
    df = pd.read_sql("""
        SELECT COALESCE(e.date, substr(p.timestamp, 1, 10)) AS date, p.description,
               e.amount_pkr AS amount, p.final_category AS category
        FROM prediction_log p LEFT JOIN expenses e ON e.id = p.id""", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.assign(currency="PKR", source="app_log")
