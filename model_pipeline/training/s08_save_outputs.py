"""
s08_save_outputs.py

Step 08: Save outputs (versioned artifacts for reproducibility)

Saves into a versioned run folder:
- fitted TF-IDF vectorizer (joblib)
- trained NMF model (joblib)
- topic name mapping (json)
- run metadata (json)  <-- now includes TF-IDF params too
- copies optional evaluation outputs from s07 (if they exist)

Also runs S06 in main() so the analysis-ready CSV is produced.

Run:
python -m model_pipeline.training.s08_save_outputs
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from model_pipeline.training.s06_topic_allocation import (
    TOPIC_NAMES,
    export_analysis_ready_csv,
    run_topic_allocation,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Project root (robust paths no matter where command is run from)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Run metadata container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunMetadata:
    run_id: str
    created_utc: str
    dataset_name: str
    n_docs: int
    tfidf_shape: tuple[int, int]

    # TF-IDF params (audit trail)
    tfidf_min_df: Any
    tfidf_max_df: Any
    tfidf_max_features: Any
    tfidf_ngram_range: list[int]
    tfidf_vocab_size: Optional[int]

    # NMF params
    nmf_n_topics: int
    nmf_init: str
    nmf_random_state: int
    nmf_max_iter: int

    # Metrics
    reconstruction_error: float
    dominant_topic_weight_min: float
    dominant_topic_weight_mean: float
    dominant_topic_weight_max: float


def _make_run_id() -> str:
    # readable + sortable
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logger.info("Wrote JSON: %s", path)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        logger.info("Copied: %s -> %s", src, dst)
    else:
        logger.info("Optional file not found (skipped): %s", src)


# ─────────────────────────────────────────────────────────────────────────────
# Save function
# ─────────────────────────────────────────────────────────────────────────────
def save_run_outputs(
    run_dir: Path,
    vectorizer,
    nmf_model,
    X,
    topic_names: dict[int, str] = TOPIC_NAMES,
    dataset_name: str = "full_retro",
    reconstruction_error: Optional[float] = None,
    W: Optional[np.ndarray] = None,
) -> Path:
    """
    Persists artifacts + metadata into a versioned run directory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving run artifacts to: %s", run_dir)

    # 1) Save fitted artifacts
    vec_path = run_dir / "vectorizer.joblib"
    nmf_path = run_dir / "nmf_model.joblib"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(nmf_model, nmf_path)
    logger.info("Saved: %s", vec_path)
    logger.info("Saved: %s", nmf_path)

    # 2) Save topic mapping
    topic_map_path = run_dir / "topic_names.json"
    _write_json({str(k): v for k, v in topic_names.items()}, topic_map_path)

    # 3) Compute W for dominant stats if not provided
    if W is None:
        try:
            W = nmf_model.transform(X)
        except Exception:
            W = None

    if reconstruction_error is None:
        reconstruction_error = getattr(nmf_model, "reconstruction_err_", None)

    if W is not None:
        dom = W.max(axis=1)
        dom_min, dom_mean, dom_max = dom.min(), dom.mean(), dom.max()
        n_docs = int(W.shape[0])
        n_topics = int(W.shape[1])
    else:
        dom_min = dom_mean = dom_max = float("nan")
        n_docs = int(X.shape[0])
        n_topics = int(getattr(nmf_model, "n_components", len(topic_names)))

    # 4) TF-IDF params for audit trail (read off fitted vectorizer)
    tfidf_min_df = getattr(vectorizer, "min_df", None)
    tfidf_max_df = getattr(vectorizer, "max_df", None)
    tfidf_max_features = getattr(vectorizer, "max_features", None)
    ngram = getattr(vectorizer, "ngram_range", (1, 1))
    tfidf_ngram_range = [int(ngram[0]), int(ngram[1])]
    tfidf_vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else None

    meta = RunMetadata(
        run_id=run_dir.name,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_name=dataset_name,
        n_docs=n_docs,
        tfidf_shape=(int(X.shape[0]), int(X.shape[1])),
        tfidf_min_df=tfidf_min_df,
        tfidf_max_df=tfidf_max_df,
        tfidf_max_features=tfidf_max_features,
        tfidf_ngram_range=tfidf_ngram_range,
        tfidf_vocab_size=tfidf_vocab_size,
        nmf_n_topics=n_topics,
        nmf_init=str(getattr(nmf_model, "init", "")),
        nmf_random_state=int(getattr(nmf_model, "random_state", -1) or -1),
        nmf_max_iter=int(getattr(nmf_model, "max_iter", -1)),
        reconstruction_error=_safe_float(reconstruction_error),
        dominant_topic_weight_min=_safe_float(dom_min),
        dominant_topic_weight_mean=_safe_float(dom_mean),
        dominant_topic_weight_max=_safe_float(dom_max),
    )
    _write_json(asdict(meta), run_dir / "run_metadata.json")

    # 5) Copy optional s07 outputs into this run folder (if present)
    _copy_if_exists(
        PROJECT_ROOT / "data" / "Interrim" / "coherence_sweep.csv",
        run_dir / "evaluation" / "coherence_sweep.csv",
    )
    _copy_if_exists(
        PROJECT_ROOT / "data" / "Interrim" / "stability_seeds.csv",
        run_dir / "evaluation" / "stability_seeds.csv",
    )

    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main: run full training + topic allocation + save artifacts
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO)

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation
    from model_pipeline.training.s05_nmf_training import train_nmf

    run_id = _make_run_id()
    run_dir = PROJECT_ROOT / "experiments" / "outputs" / "runs" / run_id

    # 1) Load + preprocess
    df = load_articles("full_retro")
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    # 2) Vectorise (fit vectorizer)
    vec_out = run_vectorisation(df)

    # 3) Train final NMF (30 topics)
    nmf_out = train_nmf(vec_out.X)

    # 4) Topic allocation + analysis-ready export (S06)
    df_alloc = run_topic_allocation(
        df=df,
        nmf_model=nmf_out.nmf_model,
        vectorizer=vec_out.vectorizer,
    )

    out_csv = PROJECT_ROOT / "data" / "full_retro" / "retro_topics_analysis_ready.csv"
    export_analysis_ready_csv(df_alloc, out_csv)
    logger.info("Analysis-ready CSV written: %s", out_csv)

    # 5) Save run artifacts
    save_run_outputs(
        run_dir=run_dir,
        vectorizer=vec_out.vectorizer,
        nmf_model=nmf_out.nmf_model,
        X=vec_out.X,
        topic_names=TOPIC_NAMES,
        dataset_name="full_retro",
        reconstruction_error=nmf_out.reconstruction_error,
        W=nmf_out.W,
    )

    print("\n✅ Saved run artifacts to:")
    print(run_dir)
    print("\nContents:")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()