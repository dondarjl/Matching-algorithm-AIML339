"""
Model A: MiniLM sentence-embedding cosine similarity (off-the-shelf baseline).
Model B: structured compatibility features -> logistic regression.
Fusion:  [Model A score] + [Model B features] -> logistic regression, learns
         how much weight to give the embedding signal vs. structured signal.

IMPORTANT -- run location: EmbeddingModel downloads 'all-MiniLM-L6-v2' from
Hugging Face Hub on first use. This sandbox has no internet access to
huggingface.co, so EmbeddingModel methods cannot be exercised here. Everything
in this file that does NOT touch EmbeddingModel (StructuredModel, FusionModel)
is fully runnable and tested in this environment. Run the embedding parts on
your own machine -- no code changes needed, just `pip install
sentence-transformers` there and this will download the model on first call.
"""
from __future__ import annotations

import zlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel:
    """Model A: off-the-shelf sentence-embedding cosine similarity baseline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded, requires internet on first call

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def pairwise_similarity(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        """Cosine similarity for aligned pairs (texts_a[i] vs texts_b[i])."""
        emb_a = self.embed(texts_a)
        emb_b = self.embed(texts_b)
        # embeddings are normalized, so row-wise dot product == cosine similarity
        return np.sum(emb_a * emb_b, axis=1)

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Full pairwise cosine similarity matrix for a list of texts."""
        emb = self.embed(texts)
        return cosine_similarity(emb)


class MockEmbeddingModel:
    """
    Drop-in replacement for EmbeddingModel with the exact same interface
    (embed / pairwise_similarity / similarity_matrix), but deterministic and
    fully offline -- no Hugging Face download needed.

    Purpose: let the *rest* of the pipeline (feature assembly, model
    fitting, evaluation, statistical tests) be built and tested end-to-end
    in an environment without internet access. The embeddings themselves
    are meaningless (text-seeded random vectors) -- swap in the real
    EmbeddingModel on a machine with internet access and nothing else in
    the pipeline needs to change, since the interface is identical.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _text_to_vec(self, text: str) -> np.ndarray:
        seed = zlib.crc32(text.encode('utf-8'))  # stable across processes (str hash() is randomised)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=self.dim)
        return v / np.linalg.norm(v)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._text_to_vec(t) for t in texts])

    def pairwise_similarity(self, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        emb_a = self.embed(texts_a)
        emb_b = self.embed(texts_b)
        return np.sum(emb_a * emb_b, axis=1)

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        emb = self.embed(texts)
        return cosine_similarity(emb)


class StructuredModel:
    """Model B: structured compatibility features -> logistic regression."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StructuredModel":
        self.clf.fit(X[self.feature_names], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.feature_names])[:, 1]

    @property
    def coefficients(self) -> dict:
        return dict(zip(self.feature_names, self.clf.coef_[0]))


class FusionModel:
    """
    Fusion: takes Model A's embedding_cosine_similarity score alongside
    Model B's structured features, and learns their relative weights via
    logistic regression. This is the model whose performance answers the
    core research question (fusion vs. embedding-only baseline).
    """

    def __init__(self, structured_feature_names: list[str]):
        self.feature_names = ["embedding_cosine_similarity"] + list(structured_feature_names)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FusionModel":
        self.clf.fit(X[self.feature_names], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[self.feature_names])[:, 1]

    @property
    def coefficients(self) -> dict:
        """
        Inspect these after fitting to see how much weight the fusion model
        assigns to the embedding baseline vs. each structured feature --
        this is direct evidence for/against the research hypothesis.
        """
        return dict(zip(self.feature_names, self.clf.coef_[0]))


class EmbeddingOnlyBaseline:
    """
    The baseline the research question compares against: threshold the raw
    embedding_cosine_similarity score directly (no learned weighting at all).
    A single logistic regression on one feature is used instead of a fixed
    threshold so the baseline gets a fair, equally-tuned decision boundary --
    the comparison is about *fusion* vs *embedding alone*, not about who gets
    to calibrate their cutoff.
    """

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EmbeddingOnlyBaseline":
        self.clf.fit(X[["embedding_cosine_similarity"]], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[["embedding_cosine_similarity"]])[:, 1]


if __name__ == "__main__":
    # Smoke test StructuredModel and FusionModel with synthetic data --
    # doesn't touch EmbeddingModel, so this runs fine in the sandbox.
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "same_cohort": rng.integers(0, 2, n),
        "arrival_date_proximity": rng.random(n),
        "course_overlap_jaccard": rng.random(n),
        "embedding_cosine_similarity": rng.uniform(-1, 1, n),
    })
    # synthetic ground truth correlated with the features, plus noise
    logit = (2 * df["same_cohort"] + 1.5 * df["arrival_date_proximity"]
             + 1.0 * df["course_overlap_jaccard"] + 1.0 * df["embedding_cosine_similarity"]
             + rng.normal(0, 1, n))
    y = (logit > logit.median()).astype(int)

    structured = StructuredModel(["same_cohort", "arrival_date_proximity", "course_overlap_jaccard"])
    structured.fit(df, y)
    print("StructuredModel coefficients:", structured.coefficients)

    fusion = FusionModel(["same_cohort", "arrival_date_proximity", "course_overlap_jaccard"])
    fusion.fit(df, y)
    print("FusionModel coefficients:", fusion.coefficients)

    baseline = EmbeddingOnlyBaseline()
    baseline.fit(df, y)
    print("EmbeddingOnlyBaseline fit OK, sample predict_proba:", baseline.predict_proba(df.head(3)))

    print("\nEmbeddingModel: not tested here (needs huggingface.co access). "
          "Run this file's __main__ block on your machine after "
          "`pip install sentence-transformers` to also smoke-test embed()/pairwise_similarity().")