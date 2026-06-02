"""Baseline coherence methods: TF-IDF and Sentence-BERT.

These baselines accept text organized by channel (Dict[str, str]) and return
the same CoherenceResult dataclass used by the hidden-state pipeline, enabling
direct comparison across methods.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from coherence.metrics import CONTROLLED_CHANNELS, CoherenceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_pairwise_cosine(vectors: np.ndarray, names: list[str]) -> tuple[float, Optional[float]]:
    """Compute market and brand coherence from a matrix of vectors.

    Parameters
    ----------
    vectors:
        2-D array of shape ``(n_channels, D)`` — one row per channel.
    names:
        Channel names matching the row order of *vectors*.

    Returns
    -------
    (market_coherence, brand_coherence)
        brand_coherence is ``None`` if fewer than 2 controlled channels.
    """
    n = len(names)
    if n < 2:
        raise ValueError(f"At least 2 channels required, got {n}")

    # Full pairwise cosine similarity matrix (n x n)
    sim_matrix = sklearn_cosine_similarity(vectors)

    # Market coherence: mean of upper triangle (all pairs)
    pair_sims: list[float] = []
    for i, j in combinations(range(n), 2):
        pair_sims.append(float(sim_matrix[i, j]))
    market_coherence = float(np.mean(pair_sims))

    # Brand coherence: controlled channels only
    controlled_indices = [i for i, name in enumerate(names) if name in CONTROLLED_CHANNELS]
    if len(controlled_indices) >= 2:
        brand_sims: list[float] = []
        for i, j in combinations(controlled_indices, 2):
            brand_sims.append(float(sim_matrix[i, j]))
        brand_coherence: Optional[float] = float(np.mean(brand_sims))
    else:
        brand_coherence = None

    return market_coherence, brand_coherence


# ---------------------------------------------------------------------------
# TF-IDF baseline
# ---------------------------------------------------------------------------


def compute_tfidf_coherence(channel_texts: Dict[str, str]) -> CoherenceResult:
    """TF-IDF cosine similarity across channel text pairs.

    Fits a ``TfidfVectorizer`` on the channel texts and computes pairwise
    cosine similarity, then aggregates into market and brand coherence.

    Parameters
    ----------
    channel_texts:
        Mapping of channel name to document text.

    Returns
    -------
    CoherenceResult
        With ``method="tfidf"``.
    """
    if len(channel_texts) < 2:
        raise ValueError(f"At least 2 channels required, got {len(channel_texts)}")

    names = sorted(channel_texts.keys())
    texts = [channel_texts[name] for name in names]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    market_coherence, brand_coherence = _mean_pairwise_cosine(tfidf_matrix, names)

    return CoherenceResult(
        brand_coherence=brand_coherence,
        market_coherence=market_coherence,
        method="tfidf",
        vocab_narrowness_flag=None,
        ttr_values=None,
    )


# ---------------------------------------------------------------------------
# Sentence-BERT baseline
# ---------------------------------------------------------------------------

_sbert_model = None


def _get_sbert_model():
    """Lazy-load the sentence-BERT model (singleton)."""
    global _sbert_model  # noqa: PLW0603
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def compute_sbert_coherence(channel_texts: Dict[str, str]) -> CoherenceResult:
    """Sentence-BERT mean-embedding cosine similarity across channel text pairs.

    Uses the ``all-MiniLM-L6-v2`` model from sentence-transformers to encode
    each channel text, then computes pairwise cosine similarity.

    Parameters
    ----------
    channel_texts:
        Mapping of channel name to document text.

    Returns
    -------
    CoherenceResult
        With ``method="sbert"``.
    """
    if len(channel_texts) < 2:
        raise ValueError(f"At least 2 channels required, got {len(channel_texts)}")

    names = sorted(channel_texts.keys())
    texts = [channel_texts[name] for name in names]

    model = _get_sbert_model()
    embeddings = model.encode(texts, convert_to_numpy=True)

    market_coherence, brand_coherence = _mean_pairwise_cosine(embeddings, names)

    return CoherenceResult(
        brand_coherence=brand_coherence,
        market_coherence=market_coherence,
        method="sbert",
        vocab_narrowness_flag=None,
        ttr_values=None,
    )


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------


def compare_methods(
    method_results: Dict[str, Dict[str, CoherenceResult]],
) -> Dict[str, Any]:
    """Spearman correlation between all method pairs on ``market_coherence``.

    Parameters
    ----------
    method_results:
        ``{method_name: {product_id: CoherenceResult, ...}, ...}``

    Returns
    -------
    dict
        Keys like ``"method_a_vs_method_b"`` (alphabetically sorted) mapping to
        ``{"spearman_rho": float, "p_value": float, "n": int}``, plus a
        ``"summary"`` key. JSON-serializable.
    """
    method_names = sorted(method_results.keys())
    output: Dict[str, Any] = {}

    for m_a, m_b in combinations(method_names, 2):
        results_a = method_results[m_a]
        results_b = method_results[m_b]

        # Find common product ids
        common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))
        n = len(common_ids)

        if n < 3:
            output[f"{m_a}_vs_{m_b}"] = {
                "spearman_rho": None,
                "p_value": None,
                "n": n,
            }
            continue

        scores_a = [results_a[pid].market_coherence for pid in common_ids]
        scores_b = [results_b[pid].market_coherence for pid in common_ids]

        rho, p_value = stats.spearmanr(scores_a, scores_b)

        output[f"{m_a}_vs_{m_b}"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "n": n,
        }

    # Summary
    comparison_keys = [k for k in output if k != "summary"]
    valid_comparisons = [
        output[k] for k in comparison_keys if output[k]["spearman_rho"] is not None
    ]

    if valid_comparisons:
        mean_rho = float(np.mean([c["spearman_rho"] for c in valid_comparisons]))
        summary = (
            f"{len(valid_comparisons)} method pair(s) compared; "
            f"mean Spearman rho = {mean_rho:.3f}"
        )
    else:
        summary = "No valid comparisons (need >= 3 common products per pair)"

    output["summary"] = summary

    return output
