"""
s07_evaluation.py

Step 07: Model evaluation (matches notebook evaluation logic)

What this does (pure evaluation — NO file writing):
1) Coherence sweep over topic counts (gensim c_v coherence)
2) Topic stability across random seeds (cosine similarity of H matrices)

Inputs (typical pipeline run):
- df with:
    - tokens_final : list[list[str]]  (from s03)
- vectorizer + TF-IDF matrix X (from s04)
- helper to extract top words per topic given feature_names (from s05-style logic)

Outputs (returned, not saved):
- coherence_scores_df : DataFrame with coherence per n_topics
- stability_scores_df : DataFrame with per-seed stability + avg stability repeated on each row

Run:
python -m model_pipeline.training.s07_evaluation
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy dependency (gensim) — imported lazily inside functions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationOutputs:
    coherence_scores: pd.DataFrame
    stability_scores: pd.DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# Small private helper: fit NMF (for sweeps/stability checks only)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_nmf(
    X,
    n_topics: int,
    random_state: int = 42,
    init: str = "nndsvd",
    max_iter: int = 1000,
) -> NMF:
    model = NMF(
        n_components=n_topics,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
    )
    model.fit(X)
    return model


def get_top_words_per_topic(
    model: NMF,
    feature_names: np.ndarray,
    n_top_words: int = 10,
) -> List[List[str]]:
    """
    Notebook-style topic word extraction for gensim coherence.
    """
    topics: List[List[str]] = []
    for topic_vec in model.components_:
        top_idx = topic_vec.argsort()[-n_top_words:][::-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


# ─────────────────────────────────────────────────────────────────────────────
# 1) Coherence sweep
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_coherence_over_topic_range(
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

    IMPORTANT:
    - texts_tokens MUST be list[list[str]] (e.g. df["tokens_final"].tolist()).
    """
    # Avoid generator exhaustion + allow logging list safely
    topic_range = list(topic_range)

    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError as e:
        raise ImportError(
            "gensim is required for coherence evaluation. Install with: pip install gensim"
        ) from e

    logger.info("Step 07 (evaluation): coherence sweep starting. topic_range=%s", topic_range)

    texts = list(texts_tokens)
    dictionary = Dictionary(texts)

    rows = []
    for n in topic_range:
        logger.info("Coherence: fitting NMF with n_topics=%d", n)
        nmf_model = _fit_nmf(
            X=X,
            n_topics=n,
            random_state=random_state,
            init=init,
            max_iter=max_iter,
        )
        topics = get_top_words_per_topic(nmf_model, feature_names, n_top_words=n_top_words)

        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence = float(cm.get_coherence())
        rows.append({"n_topics": n, "coherence_cv": coherence})
        logger.info("Coherence n_topics=%d -> %.4f", n, coherence)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Topic stability across seeds
# ─────────────────────────────────────────────────────────────────────────────
def _topic_similarity(H1: np.ndarray, H2: np.ndarray) -> float:
    """
    Matches notebook: cosine_similarity(H1, H2).max(axis=1).mean()
    """
    sim = cosine_similarity(H1, H2)
    return float(sim.max(axis=1).mean())


def evaluate_topic_stability(
    X,
    n_topics: int = 30,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    init: str = "nndsvda",
    max_iter: int = 3000,
) -> pd.DataFrame:
    """
    Returns per-seed stability vs the first seed (baseline),
    with avg_stability repeated on every row (clean tabular output).
    """
    seeds = list(seeds)

    if len(seeds) < 2:
        logger.warning("Stability: need >=2 seeds; got %d. Returning empty DF.", len(seeds))
        return pd.DataFrame(columns=["baseline_seed", "seed", "stability", "avg_stability"])

    logger.info(
        "Step 07 (evaluation): stability starting. n_topics=%d seeds=%s init=%s max_iter=%d",
        n_topics,
        seeds,
        init,
        max_iter,
    )

    H_runs = []
    for seed in seeds:
        model = NMF(
            n_components=n_topics,
            random_state=seed,
            max_iter=max_iter,
            init=init,
        )
        W = model.fit_transform(X)
        _ = W  # not used; kept for symmetry with notebook
        H_runs.append(model.components_)

    baseline_seed = seeds[0]
    H0 = H_runs[0]

    rows = []
    stabilities = []
    for seed, H in zip(seeds[1:], H_runs[1:]):
        stab = _topic_similarity(H0, H)
        stabilities.append(stab)
        rows.append(
            {
                "baseline_seed": baseline_seed,
                "seed": seed,
                "stability": float(stab),
            }
        )

    avg_stability = float(np.mean(stabilities)) if stabilities else float("nan")
    for r in rows:
        r["avg_stability"] = avg_stability

    df = pd.DataFrame(rows)
    logger.info("Stability avg=%.4f (baseline=%d)", avg_stability, baseline_seed)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main smoke test (package mode)
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Silence gensim’s noisy internal INFO logs (optional but recommended)
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("gensim.topic_coherence").setLevel(logging.WARNING)

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation

    # Full corpus (matches notebook evaluation pattern)
    df = load_articles("full_retro")
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    vec_out = run_vectorisation(df)

    # Coherence sweep
    coherence_df = evaluate_coherence_over_topic_range(
        X=vec_out.X,
        feature_names=vec_out.feature_names,
        texts_tokens=df["tokens_final"].tolist(),  # ✅ token lists, not strings
        topic_range=range(5, 80, 5),
        n_top_words=10,
        random_state=42,
        init="nndsvd",
        max_iter=1000,
    )

    # Stability (final model topic count)
    stability_df = evaluate_topic_stability(
        X=vec_out.X,
        n_topics=30,
        seeds=(1, 2, 3, 4, 5),
        init="nndsvda",
        max_iter=3000,
    )

    logger.info("Coherence results head:\n%s", coherence_df.head(5))
    logger.info("Stability results:\n%s", stability_df)

    # Print a tiny human-readable summary
    best_row = coherence_df.sort_values("coherence_cv", ascending=False).head(1)
    if not best_row.empty:
        print("\n--- Coherence sweep (best) ---")
        print(best_row.to_string(index=False))

    if not stability_df.empty:
        print("\n--- Stability (avg) ---")
        print(stability_df[["baseline_seed", "avg_stability"]].head(1).to_string(index=False))


if __name__ == "__main__":
    main()