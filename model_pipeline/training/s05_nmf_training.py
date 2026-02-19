"""
s05_nmf_training.py

Step 05: NMF model training (pure training module; matches notebook intent)

Input:
- X: TF-IDF sparse matrix (from s04_vectorisation)

Output:
- model: fitted sklearn.decomposition.NMF
- W: document-topic matrix (n_docs x n_topics)
- H: topic-word matrix (n_topics x n_terms)

Design:
- Pipeline-facing functions are PURE (no file I/O, no topic mapping, no exports).
- Smoke test at bottom runs the FINAL model configuration (30 topics etc.)
  and prints quick sanity checks (including topic allocation counts) without
  saving anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NMFTrainingOutput:
    model: NMF
    W: np.ndarray          # document-topic weights
    H: np.ndarray          # topic-word weights (model.components_)
    reconstruction_err: float


# ─────────────────────────────────────────────────────────────────────────────
# Core training (PURE)
# ─────────────────────────────────────────────────────────────────────────────
def build_nmf(
    n_topics: int = 30,
    init: str = "nndsvd",
    random_state: int = 42,
    max_iter: int = 1000,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
) -> NMF:
    """
    Construct an sklearn NMF model (not fitted yet).
    Params mirror your notebook defaults + leave room for tuning.
    """
    return NMF(
        n_components=n_topics,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
    )


def train_nmf(
    X: sparse.spmatrix,
    n_topics: int = 30,
    init: str = "nndsvd",
    random_state: int = 42,
    max_iter: int = 1000,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
) -> NMFTrainingOutput:
    """
    Fit NMF on TF-IDF matrix X and return model + W/H matrices.

    Note:
      - W = model.fit_transform(X)
      - H = model.components_
    """
    if not sparse.issparse(X):
        raise TypeError(f"Expected a scipy sparse matrix for X, got: {type(X)}")

    logger.info("Step 05 (NMF): starting. X shape=%s", X.shape)

    model = build_nmf(
        n_topics=n_topics,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
    )

    W = model.fit_transform(X)
    H = model.components_

    logger.info("Step 05 (NMF): complete.")
    logger.info("W shape=%s | H shape=%s", W.shape, H.shape)
    logger.info("Reconstruction error: %.6f", float(model.reconstruction_err_))

    return NMFTrainingOutput(
        model=model,
        W=W,
        H=H,
        reconstruction_err=float(model.reconstruction_err_),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (FINAL model settings)
# Run as:
#   python -m model_pipeline.training.s05_nmf_training
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation

    logging.basicConfig(level=logging.INFO)

    # Use FULL dataset for "final model" smoke test (this will be slower than a 200-row sample)
    df = load_articles("full_retro")
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    vec_out = run_vectorisation(df)
    X = vec_out.X

    # FINAL model configuration (as per your notebook choice)
    out = train_nmf(
        X,
        n_topics=30,
        init="nndsvd",
        random_state=42,
        max_iter=1000,
    )

    # Sanity checks (prints only; still "pure" training module—no saving)
    W = out.W
    topic_num = W.argmax(axis=1)
    dominant_weight = W.max(axis=1)

    print("\n--- NMF Smoke Test (Final Model) ---")
    print("X shape:", X.shape)
    print("W shape:", W.shape)
    print("H shape:", out.H.shape)
    print("Reconstruction error:", out.reconstruction_err)

    # Allocation sanity: are topics being used?
    counts = np.bincount(topic_num, minlength=30)
    print("\nTopic usage (topic_id: count) for first 10 topics:")
    for i in range(10):
        print(f"{i}: {counts[i]}")

    print("\nDominant weight summary:")
    print("min:", float(dominant_weight.min()))
    print("mean:", float(dominant_weight.mean()))
    print("max:", float(dominant_weight.max()))
