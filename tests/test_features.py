# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import pandas as pd
import pytest

from src.features import clean_text, numeric_features


@pytest.mark.parametrize("raw, expected", [
    ("2 Place 5 to Place 0", "place to place"),
    ("Half lit  milk (1L)", "half lit milk l"),
    ("  UPPER case!!  ", "upper case"),
    ("Toothbrush (buy 2 get 1)", "toothbrush buy get"),
    ("12345", ""),
    ("", ""),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_numeric_features_weekend_flag_and_log_amount():
    df = pd.DataFrame({"amount": [0, 99], "date": ["2026-11-23", "2026-11-29"]})  # Monday, Sunday
    features = numeric_features(df)
    assert features.shape == (2, 3)
    assert features[0].tolist() == [0.0, 0.0, 0.0]
    assert features[1, 0] == pytest.approx(4.6052, abs=1e-3)  # log1p(99) = log(100)
    assert features[1, 1:].tolist() == [6.0, 1.0]
