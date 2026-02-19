"""
s05_nmf_training.py

Step 05: NMF topic model training (matches FINAL notebook model used for allocation/export)

Input:
- df with column: 'text_final' (from s03_spacy_processing)
- TF-IDF matrix X (from s04_vectorisation)

Output:
- nmf_model: fitted sklearn.decomposition.NMF
- W: document-topic matrix  (n_docs x 30)
- H: topic-word matrix      (30 x n_features)
- reconstruction_error: nmf_model.reconstruction_err_

FINAL NOTE:
- This is the "nmf_model_30" equivalent from your notebook:
  NMF(n_components=30, init='nndsvd', random_state=42, max_iter=1000)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


@dataclass
class NMFTrainingOutput:
    nmf_model: NMF
    W: np.ndarray
    H: np.ndarray
    reconstruction_error: float


def train_nmf(
    X: sparse.spmatrix,
    n_topics: int = 30,
    random_state: int = 42,
    init: str = "nndsvd",
    max_iter: int = 1000,
) -> NMFTrainingOutput:
    """
    Train final NMF model (notebook-equivalent nmf_model_30).
    """
    logger.info("Step 05 (NMF): starting. X shape=%s", X.shape)

    nmf_model = NMF(
        n_components=n_topics,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
    )

    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    err = float(nmf_model.reconstruction_err_)

    logger.info("Step 05 (NMF): complete.")
    logger.info("W shape=%s | H shape=%s", W.shape, H.shape)
    logger.info("Reconstruction error: %.6f", err)

    return NMFTrainingOutput(nmf_model=nmf_model, W=W, H=H, reconstruction_error=err)


def get_top_words_per_topic(
    model: NMF,
    feature_names: np.ndarray,
    n_top_words: int = 20,
) -> List[List[str]]:
    """
    Notebook-style topic word extraction.
    Returns list of topics, each topic is list of top words.
    """
    topics: List[List[str]] = []
    for topic_vec in model.components_:
        top_idx = topic_vec.argsort()[-n_top_words:][::-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def dominant_topic_stats(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - dominant_topic_id per doc
      - dominant_topic_weight per doc
    Matches notebook export logic (argmax + max).
    """
    dominant_topic_id = W.argmax(axis=1)
    dominant_topic_weight = W.max(axis=1)
    return dominant_topic_id, dominant_topic_weight


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (package mode ONLY) — trains final model and prints allocations
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation

    logging.basicConfig(level=logging.INFO)

    # Full corpus run (matches how you trained in the notebook)
    df = load_articles("full_retro")
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    vec_out = run_vectorisation(df)   # uses your notebook TF-IDF settings
    nmf_out = train_nmf(vec_out.X)    # uses final notebook NMF settings

    dom_topic, dom_weight = dominant_topic_stats(nmf_out.W)

    print("\n--- NMF Smoke Test (Final Notebook Model) ---")
    print("X shape:", vec_out.X.shape)
    print("W shape:", nmf_out.W.shape)
    print("H shape:", nmf_out.H.shape)
    print("Reconstruction error:", nmf_out.reconstruction_error)

    # Topic usage counts (same idea as your earlier print)
    counts = np.bincount(dom_topic, minlength=nmf_out.W.shape[1])
    print("\nTopic usage (topic_id: count) for first 10 topics:")
    for i in range(min(10, len(counts))):
        print(f"{i}: {counts[i]}")

    print("\nDominant weight summary:")
    print("min:", float(dom_weight.min()))
    print("mean:", float(dom_weight.mean()))
    print("max:", float(dom_weight.max()))

    # Optional: print top words per topic (like notebook display_topics)
    topics = get_top_words_per_topic(nmf_out.nmf_model, vec_out.feature_names, n_top_words=20)
    print("\nTop words per topic (first 5 topics):")
    for t in range(min(5, len(topics))):
        print(f"Topic {t}: {', '.join(topics[t])}")
