"""
s04_vectorisation.py

Step 04: TF-IDF vectorisation (matches notebook workflow)

Input:
- df with column: 'text_final' (from s03_spacy_processing)

Output:
- X: sparse TF-IDF matrix
- vectorizer: fitted TfidfVectorizer
- feature_names: numpy array of feature names
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# A small container to keep outputs tidy
@dataclass
class VectorisationOutput:
    X: sparse.spmatrix
    vectorizer: TfidfVectorizer
    feature_names: object  # numpy array of strings


def build_vectorizer(
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """
    Mirrors notebook vectorizer settings.
    """
    return TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )


def run_vectorisation(
    df: pd.DataFrame,
    text_col: str = "text_final",
    vectorizer: Optional[TfidfVectorizer] = None,
) -> VectorisationOutput:
    """
    Fit TF-IDF vectorizer on df[text_col] and return matrix + fitted vectorizer.

    If you pass an existing vectorizer, it will be fit here (training step).
    Later, in inference, you'll call vectorizer.transform(...) instead.
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Expected column '{text_col}' not found. Available: {list(df.columns)}"
        )

    texts = df[text_col].fillna("").astype(str)

    logger.info("Step 04 (vectorisation): starting. Input shape=%s", df.shape)

    if vectorizer is None:
        vectorizer = build_vectorizer()

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    logger.info("TF-IDF shape: %s", X.shape)
    logger.info(
        "Vectorizer params: min_df=%s max_df=%s max_features=%s ngram_range=%s",
        vectorizer.min_df,
        vectorizer.max_df,
        vectorizer.max_features,
        vectorizer.ngram_range,
    )
    logger.info("Sample features: %s", feature_names[:20].tolist())

    return VectorisationOutput(X=X, vectorizer=vectorizer, feature_names=feature_names)


if __name__ == "__main__":
    # This MUST print if the __main__ block is executing
    print(">>> s04_vectorisation __main__ running from:", __file__, flush=True)

    import logging

    # force=True ensures you see logs even if logging was configured earlier elsewhere
    logging.basicConfig(level=logging.INFO, force=True)

    # Run in package mode:
    # python -m model_pipeline.training.s04_vectorisation
    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing

    # Small sample while refactoring
    df = load_articles("full_retro").head(200)
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    out = run_vectorisation(df)

    print("TF-IDF matrix shape:", out.X.shape, flush=True)
    print("First 20 features:", out.feature_names[:20].tolist(), flush=True)
