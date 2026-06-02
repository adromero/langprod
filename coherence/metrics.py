"""Coherence metrics for multi-channel brand language analysis.

All functions accept embeddings as numpy arrays with explicit channel labels.
Standard input: Dict[str, np.ndarray] mapping channel name to 1-D embedding of shape (D,).
Embeddings are expected to arrive pre-centered (anisotropy-corrected by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CONTROLLED_CHANNELS: set[str] = {"regulatory", "marketing", "retail", "social"}
AGGREGATION_METHODS: list[str] = ["mean_pairwise", "centroid_distance", "silhouette"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CoherenceResult:
    """Result of a coherence score computation."""

    brand_coherence: Optional[float]
    market_coherence: float
    method: str
    vocab_narrowness_flag: Optional[bool] = None
    ttr_values: Optional[dict[str, float]] = None


@dataclass
class OutlierResult:
    """Result of outlier channel identification."""

    outlier_channel: str
    mean_similarities: dict[str, float]
    gap: float


@dataclass
class AttributeCoherenceResult:
    """Per-attribute coherence across channels."""

    matrix: np.ndarray  # shape (n_attributes, n_channels)
    channel_names: list[str]  # sorted channel names matching column order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _validate_min_channels(channel_embeddings: dict[str, np.ndarray], minimum: int) -> None:
    """Raise ValueError if fewer than *minimum* channels provided."""
    if len(channel_embeddings) < minimum:
        raise ValueError(
            f"At least {minimum} channels required, got {len(channel_embeddings)}"
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_pairwise_coherence(
    channel_embeddings: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Cosine similarity for every unordered pair of channels.

    Dict keys are ``(channel_a, channel_b)`` with names sorted alphabetically.
    """
    _validate_min_channels(channel_embeddings, 2)
    names = sorted(channel_embeddings.keys())
    result: dict[tuple[str, str], float] = {}
    for a, b in combinations(names, 2):
        result[(a, b)] = _cosine_similarity(channel_embeddings[a], channel_embeddings[b])
    return result


def _aggregate_mean_pairwise(embeddings: dict[str, np.ndarray]) -> float:
    """Mean of all pairwise cosine similarities."""
    pairs = compute_pairwise_coherence(embeddings)
    if not pairs:
        return 1.0  # single channel — perfectly coherent with itself
    return float(np.mean(list(pairs.values())))


def _aggregate_centroid_distance(embeddings: dict[str, np.ndarray]) -> float:
    """Mean cosine similarity of each embedding to the centroid."""
    vecs = list(embeddings.values())
    centroid = np.mean(vecs, axis=0)
    sims = [_cosine_similarity(v, centroid) for v in vecs]
    return float(np.mean(sims))


def _aggregate_silhouette(embeddings: dict[str, np.ndarray]) -> float:
    """Single-cluster cohesion silhouette.

    1. d(i,j) = 1 - cosine_similarity(e_i, e_j)
    2. a_i = mean d(i,j) for j != i
    3. Return 1 - mean(a_i)
    """
    names = sorted(embeddings.keys())
    n = len(names)
    # Build pairwise cosine distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - _cosine_similarity(embeddings[names[i]], embeddings[names[j]])
            dist[i, j] = d
            dist[j, i] = d

    # a_i = mean distance to all other points
    a_values = []
    for i in range(n):
        others = [dist[i, j] for j in range(n) if j != i]
        a_values.append(float(np.mean(others)))

    return 1.0 - float(np.mean(a_values))


_AGGREGATORS = {
    "mean_pairwise": _aggregate_mean_pairwise,
    "centroid_distance": _aggregate_centroid_distance,
    "silhouette": _aggregate_silhouette,
}


def compute_coherence_score(
    channel_embeddings: dict[str, np.ndarray],
    method: str = "mean_pairwise",
    ttr_values: Optional[dict[str, float]] = None,
) -> CoherenceResult:
    """Compute brand and market coherence scores.

    Parameters
    ----------
    channel_embeddings:
        Mapping of channel name → 1-D embedding.
    method:
        One of ``"mean_pairwise"``, ``"centroid_distance"``, ``"silhouette"``.
    ttr_values:
        Optional type-token ratio values per channel.  For single-product
        calls the ``vocab_narrowness_flag`` is always ``None``.
    """
    _validate_min_channels(channel_embeddings, 2)
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {AGGREGATION_METHODS}")

    aggregator = _AGGREGATORS[method]

    # Market coherence — ALL channels
    market_coherence = aggregator(channel_embeddings)

    # Brand coherence — controlled channels only
    controlled = {
        k: v for k, v in channel_embeddings.items() if k in CONTROLLED_CHANNELS
    }
    if len(controlled) >= 2:
        brand_coherence: Optional[float] = aggregator(controlled)
    else:
        brand_coherence = None

    return CoherenceResult(
        brand_coherence=brand_coherence,
        market_coherence=market_coherence,
        method=method,
        vocab_narrowness_flag=None,  # set by batch_coherence_scores
        ttr_values=ttr_values,
    )


def batch_coherence_scores(
    products: dict[str, dict[str, np.ndarray]],
    method: str = "mean_pairwise",
    ttr_values: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, CoherenceResult]:
    """Compute coherence scores for a batch of products.

    Parameters
    ----------
    products:
        Mapping of product name → channel embeddings dict.
    method:
        Aggregation method forwarded to ``compute_coherence_score``.
    ttr_values:
        Optional mapping of product name → per-channel TTR values.
        Used to compute the vocab-narrowness flag when ``len(products) >= 5``.
    """
    results: dict[str, CoherenceResult] = {}
    for product_name, channel_embs in products.items():
        product_ttr = ttr_values.get(product_name) if ttr_values else None
        results[product_name] = compute_coherence_score(
            channel_embs, method=method, ttr_values=product_ttr
        )

    # Vocab narrowness flag logic
    if ttr_values is not None and len(products) >= 5:
        # Compute Spearman rho between market_coherence and mean inverse TTR
        product_names = sorted(results.keys())
        coherences = []
        mean_inv_ttrs = []
        for pn in product_names:
            coherences.append(results[pn].market_coherence)
            if pn in ttr_values and ttr_values[pn]:
                channel_ttrs = list(ttr_values[pn].values())
                inv_ttrs = [1.0 - t for t in channel_ttrs]
                mean_inv_ttrs.append(float(np.mean(inv_ttrs)))
            else:
                mean_inv_ttrs.append(0.0)

        rho, _ = stats.spearmanr(coherences, mean_inv_ttrs)
        flag = bool(rho > 0.5)

        for result in results.values():
            result.vocab_narrowness_flag = flag
    # else: flag stays None (already set by compute_coherence_score)

    return results


def identify_outlier_channel(
    channel_embeddings: dict[str, np.ndarray],
) -> OutlierResult:
    """Identify the channel with lowest mean cosine similarity to all others.

    Requires >= 3 channels.
    """
    _validate_min_channels(channel_embeddings, 3)

    names = sorted(channel_embeddings.keys())
    pairwise = compute_pairwise_coherence(channel_embeddings)

    mean_sims: dict[str, float] = {}
    for name in names:
        sims = []
        for a, b in pairwise:
            if a == name:
                sims.append(pairwise[(a, b)])
            elif b == name:
                sims.append(pairwise[(a, b)])
        mean_sims[name] = float(np.mean(sims)) if sims else 0.0

    # Outlier = lowest mean similarity
    outlier = min(mean_sims, key=lambda k: mean_sims[k])

    # Gap = difference between outlier's mean sim and next-lowest
    sorted_sims = sorted(mean_sims.values())
    if len(sorted_sims) >= 2:
        gap = sorted_sims[1] - sorted_sims[0]
    else:
        gap = 0.0

    return OutlierResult(
        outlier_channel=outlier,
        mean_similarities=mean_sims,
        gap=gap,
    )


def compute_attribute_coherence(
    attribute_probes: np.ndarray,
    attribute_names: list[str],
    channel_embeddings: dict[str, np.ndarray],
) -> AttributeCoherenceResult:
    """Compute cosine similarity of each attribute probe with each channel embedding.

    Parameters
    ----------
    attribute_probes:
        Array of shape ``(n_attributes, D)``.
    attribute_names:
        Names for each attribute row.
    channel_embeddings:
        Mapping of channel name → 1-D embedding of shape ``(D,)``.

    Returns
    -------
    AttributeCoherenceResult
        With ``matrix`` of shape ``(n_attributes, n_channels)`` and
        ``channel_names`` sorted alphabetically.
    """
    channel_names = sorted(channel_embeddings.keys())
    n_attributes = attribute_probes.shape[0]
    n_channels = len(channel_names)

    matrix = np.zeros((n_attributes, n_channels))
    for i in range(n_attributes):
        for j, ch_name in enumerate(channel_names):
            matrix[i, j] = _cosine_similarity(attribute_probes[i], channel_embeddings[ch_name])

    return AttributeCoherenceResult(matrix=matrix, channel_names=channel_names)
