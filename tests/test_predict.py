# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import pytest

from src import config
from src.predict import model_problem

# The saved model must exist and match this computer's scikit-learn version.
# If it doesn't, build it first: notebook 03, `python -m src.train`, or the app's setup screen.
PROBLEM = model_problem()
pytestmark = pytest.mark.skipif(PROBLEM is not None, reason=f"model not usable here: {PROBLEM}")


def test_predict_output_shape():
    from src.predict import predict
    result = predict("careem ride to university", 450, "2026-11-23")
    assert set(result) == {"category", "confidence", "top3", "needs_review"}
    assert result["category"] in config.CATEGORIES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top3"]) == 3
    assert result["top3"][0]["category"] == result["category"]
    confidences = [option["confidence"] for option in result["top3"]]
    assert confidences == sorted(confidences, reverse=True)
    assert result["needs_review"] == (result["confidence"] < config.CONFIDENCE_THRESHOLD)


def test_predict_works_without_amount_and_date():
    from src.predict import predict
    assert predict("milk")["category"] in config.CATEGORIES
