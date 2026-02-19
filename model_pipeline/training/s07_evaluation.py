"""
s07_evaluation.py

Step 07: Evaluation utilities (no file writing)
- Coherence sweep across topic counts (gensim c_v)
- Topic stability across random seeds (cosine similarity on H matrices)

This module returns DataFrames.
Saving is handled by S08.

Run (smoke test):
python -m model_pipeline.training.s07_evaluation
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _top_words_per_topic(model: NMF, feature_names: np.ndarray, n_top_words: int) -> list[list[str]]:
    topics: list[list[str]] = []
    for topic_vec in model.components_:
        top_idx = topic_vec.argsort()[-n_top_words:][::-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def evaluate_coherence_over_topic_range(
    *,
    X,
    feature_names: np.ndarray,
    texts_tokens: Sequence[Sequence[str]],
    topic_range: Iterable[int] = range(5, 80, 5),
    n_top_words: int = 10,
    random_state: int = 42,
    init: str = "nndsvd",
    max_iter: int = 1000,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: n_topics, coherence_cv
    """
    # generator-safe
    topic_range = list(topic_range)

    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError as e:
        raise ImportError(
            "gensim is required for coherence evaluation. Install with: pip install gensim"
        ) from e

    logger.info("Coherence sweep: topic_range=%s", topic_range)

    dictionary = Dictionary(texts_tokens)

    rows: list[dict] = []
    for n_topics in topic_range:
        logger.info("Testing coherence for n_topics=%d", n_topics)

        nmf = NMF(n_components=n_topics, init=init, random_state=random_state, max_iter=max_iter)
        nmf.fit(X)

        topics = _top_words_per_topic(nmf, feature_names, n_top_words=n_top_words)

        cm = CoherenceModel(
            topics=topics,
            texts=texts_tokens,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence = float(cm.get_coherence())

        rows.append({"n_topics": int(n_topics), "coherence_cv": coherence})

    return pd.DataFrame(rows)


def evaluate_topic_stability(
    *,
    X,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    n_topics: int = 30,
    init: str = "nndsvda",
    max_iter: int = 3000,
) -> pd.DataFrame:
    """
    Matches notebook approach:
    - Fit multiple NMF runs with different seeds
    - Compare each run’s H to the first run using cosine similarity
    - similarity = mean(max cosine similarity per topic)
    Returns DataFrame with columns: seed, stability_vs_seed0, avg_stability
    """
    seeds = list(seeds)
    if len(seeds) < 2:
        return pd.DataFrame([{"seed": seeds[0] if seeds else None, "stability_vs_seed0": np.nan, "avg_stability": np.nan}])

    def fit_H(seed: int) -> np.ndarray:
        model = NMF(n_components=n_topics, random_state=seed, max_iter=max_iter, init=init)
        model.fit(X)
        return model.components_

    H_runs = [fit_H(s) for s in seeds]

    base = H_runs[0]
    stabilities: list[float] = []
    rows: list[dict] = [{"seed": int(seeds[0]), "stability_vs_seed0": np.nan}]  # base row

    for seed, H in zip(seeds[1:], H_runs[1:]):
        sim_matrix = cosine_similarity(H, base)
        stability = float(sim_matrix.max(axis=1).mean())
        stabilities.append(stability)
        rows.append({"seed": int(seed), "stability_vs_seed0": stability})

    avg = float(np.mean(stabilities)) if stabilities else float("nan")
    for r in rows:
        r["avg_stability"] = avg

    return pd.DataFrame(rows)


def main() -> None:
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation

    logging.basicConfig(level=logging.INFO)

    # Silence extremely noisy gensim internal INFO logs
    logging.getLogger("gensim").setLevel(logging.WARNING)

    dataset_name = "full_retro"

    df = load_articles(dataset_name)
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    vec_out = run_vectorisation(df)

    # IMPORTANT: coherence uses tokens (not joined text)
    texts_tokens = df["tokens_final"].tolist()

    df_coh = evaluate_coherence_over_topic_range(
        X=vec_out.X,
        feature_names=vec_out.feature_names,
        texts_tokens=texts_tokens,
        topic_range=range(5, 80, 5),
        n_top_words=10,
        random_state=42,
        init="nndsvd",
        max_iter=1000,
    )
    print("\n--- Coherence sweep (head) ---")
    print(df_coh.head())

    df_stab = evaluate_topic_stability(
        X=vec_out.X,
        seeds=[1, 2, 3, 4, 5],
        n_topics=30,
        init="nndsvda",
        max_iter=3000,
    )
    print("\n--- Stability (all rows) ---")
    print(df_stab)


if __name__ == "__main__":
    main()