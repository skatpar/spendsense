"""Zero-shot classifier: no training at all. Each category is described in
words, and an expense gets the category whose description means the most
similar thing, according to the pre-trained MiniLM model.

This is the score any trained model has to beat to be worth training."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import numpy as np

from src import config
from src.features import embed_texts


def category_vectors() -> np.ndarray:
    """One embedding per category, in the order of config.CATEGORIES."""
    texts = [config.CATEGORY_DESCRIPTIONS[c] for c in config.CATEGORIES]
    return embed_texts(texts)


def similarity_scores(descriptions: list[str]) -> np.ndarray:
    """Cosine similarity of every description to every category,
    shape (n_descriptions, 8). Vectors are unit length, so a dot product is enough."""
    return embed_texts(descriptions) @ category_vectors().T


def zero_shot_predict(descriptions: list[str]) -> list[str]:
    best = similarity_scores(descriptions).argmax(axis=1)
    return [config.CATEGORIES[i] for i in best]
