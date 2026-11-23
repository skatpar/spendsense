"""Turning an expense (description, amount, date) into numbers a model can use.

Every model input is built with a scikit-learn ColumnTransformer, so the
exact same steps run at training time and when the app makes a prediction."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config

EMBEDDING_CACHE = config.CACHE_DIR / "minilm_embeddings.joblib"
INPUT_COLUMNS = ["description", "amount", "date"]
NUMERIC_NAMES = ["log1p_amount", "weekday", "is_weekend"]


def clean_text(s: str) -> str:
    """Lower-case, drop digits and punctuation, collapse spaces.
    EDA showed quantities ('2 plates', '1lit') are noise, not category clues."""
    s = str(s).lower()
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------- MiniLM sentence embeddings ----------

@lru_cache(maxsize=1)
def get_encoder():
    """Load MiniLM once per process; loading takes a few seconds."""
    from sentence_transformers import SentenceTransformer  # slow import, so only when needed
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()  # keeps a "Loading weights" bar out of notebook outputs
    return SentenceTransformer(config.minilm_path(), device="cpu")


def _load_cache() -> dict[str, np.ndarray]:
    return joblib.load(EMBEDDING_CACHE) if EMBEDDING_CACHE.exists() else {}


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return one 384-number vector per text. Vectors are cached on disk by
    text, so each unique description is only ever encoded once."""
    cache = _load_cache()
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        # normalize_embeddings=True makes every vector length 1, so a dot
        # product between two vectors *is* their cosine similarity.
        # batch_size=1: in a batch, short texts are padded to the longest one,
        # which changes the numbers in the 7th decimal place. One at a time,
        # a text always gets exactly the same vector, so results are reproducible.
        vectors = get_encoder().encode(missing, batch_size=1, normalize_embeddings=True,
                                       show_progress_bar=False)
        cache.update(zip(missing, vectors))
        EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, EMBEDDING_CACHE)
    return np.vstack([cache[t] for t in texts])


def embed_descriptions(descriptions) -> np.ndarray:
    """ColumnTransformer hands us a column of raw descriptions."""
    return embed_texts([clean_text(t) for t in descriptions])


# ---------- amount and date ----------

def numeric_features(df: pd.DataFrame) -> np.ndarray:
    """log1p(amount), weekday (0 = Monday ... 6 = Sunday) and an is_weekend flag.
    log1p squashes the long tail (1 to 50,000) and still works for an amount of 0."""
    amount = np.log1p(pd.to_numeric(df["amount"]).clip(lower=0).to_numpy(dtype=float))
    weekday = pd.to_datetime(df["date"]).dt.day_of_week.to_numpy(dtype=float)
    return np.column_stack([amount, weekday, (weekday >= 5).astype(float)])


def _numeric_part():
    # Scaling puts the three numbers on a similar range to the text features,
    # which keeps Logistic Regression's regularisation fair across them.
    return make_pipeline(FunctionTransformer(numeric_features), StandardScaler())


# ---------- the two feature sets compared in Step 4 ----------

def tfidf_features() -> ColumnTransformer:
    """TF-IDF of the cleaned description (words and 2-word phrases seen in at
    least 2 descriptions) + the 3 numeric features."""
    tfidf = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2)
    return ColumnTransformer([("text", tfidf, "description"),
                              ("numeric", _numeric_part(), ["amount", "date"])])


def embedding_features() -> ColumnTransformer:
    """384 MiniLM numbers for the cleaned description + the 3 numeric features."""
    return ColumnTransformer([("text", FunctionTransformer(embed_descriptions), "description"),
                              ("numeric", _numeric_part(), ["amount", "date"])])
