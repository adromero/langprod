"""Analysis module — RDM computation, RSA, permutation testing, and sanity checks.

This module provides the core analysis functions for the language production pipeline:
    - correct_anisotropy(): anisotropy correction (none/mean_centering/whitening)
    - compute_rdm(): compute representational dissimilarity matrix
    - compute_rdms_all_layers(): per-layer RDMs from HDF5
    - build_product_model_rdm(): product identity model RDM
    - build_register_model_rdm(): register identity model RDM
    - build_within_category_model_rdm(): within-category discrimination RDM
    - rsa_correlation(): Spearman RSA between observed and model RDMs
    - partial_rsa(): partial Spearman RSA controlling for nuisance
    - build_length_nuisance_rdm(): pairwise token count difference RDM
    - build_lexical_nuisance_rdm(): Jaccard distance on token sets
    - compute_condition_similarities(): SP-DR, DP-SC, DC per layer
    - run_permutation_test_tiered(): 200 screen + 10000 full on top-k
    - apply_fdr_correction(): Benjamini-Hochberg FDR
    - rsa_sanity_check(): triage logic for RSA curves
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 4.1  Anisotropy Correction
# ---------------------------------------------------------------------------


def correct_anisotropy(
    representations: np.ndarray,
    method: str,
    n_components: int | None = None,
    epsilon: float = 1e-8,
    variance_threshold: float = 0.95,
) -> np.ndarray:
    """Apply anisotropy correction to a representation matrix.

    Parameters
    ----------
    representations : np.ndarray
        Shape (N, D) — N stimuli, D-dimensional embeddings.
    method : str
        One of "none", "mean_centering", "whitening".
    n_components : int or None
        Number of PCA components for whitening.  If None, defaults to
        min(N-1, D).
    epsilon : float
        Regularization term added to singular values during whitening.
    variance_threshold : float
        Warn if retained variance falls below this fraction.

    Returns
    -------
    np.ndarray
        Corrected representations, same first-axis size as input.
    """
    if method == "none":
        return representations.copy()

    N, D = representations.shape

    if method == "mean_centering":
        global_mean = representations.mean(axis=0)
        return representations - global_mean

    if method == "whitening":
        if n_components is None:
            n_components = min(N - 1, D)

        pca = PCA(n_components=n_components, whiten=True)
        whitened = pca.fit_transform(representations)

        # Check retained variance
        retained_variance = pca.explained_variance_ratio_.sum()
        if retained_variance < variance_threshold:
            warnings.warn(
                f"Whitening retained only {retained_variance:.3f} variance "
                f"(threshold={variance_threshold:.2f})",
                stacklevel=2,
            )

        # Check condition number
        singular_values = pca.singular_values_
        # Add epsilon to avoid division by zero
        condition_number = singular_values[0] / (singular_values[-1] + epsilon)
        if condition_number > 1e6:
            logger.warning(
                "High condition number %.2e detected — numerical instability risk.",
                condition_number,
            )

        return whitened

    raise ValueError(f"Unknown anisotropy method: {method!r}")


# ---------------------------------------------------------------------------
# 4.2  RDM & RSA Computation
# ---------------------------------------------------------------------------


def compute_rdm(
    representations: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute a representational dissimilarity matrix.

    Parameters
    ----------
    representations : np.ndarray
        Shape (N, D).
    metric : str
        Distance metric passed to ``scipy.spatial.distance.pdist``.

    Returns
    -------
    np.ndarray
        Symmetric (N, N) dissimilarity matrix.
    """
    distances = pdist(representations, metric=metric)
    return squareform(distances)


def compute_rdms_all_layers(
    h5_path: str,
    stimuli_meta: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[int, np.ndarray]:
    """Compute RDMs for every layer, reading one layer at a time from HDF5.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing hidden states.  Expected dataset
        ``/hidden_states_mean_no_special`` with shape (N, L+1, D).
    stimuli_meta : list[dict]
        Stimulus metadata records (used for ordering validation).
    config : dict
        Pipeline configuration.  Must contain ``anisotropy_methods``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from layer index to its (N, N) RDM.  One entry per layer,
        using the first anisotropy method in config.
    """
    anisotropy_method = config.get("anisotropy_methods", ["none"])[0]
    pca_epsilon = config.get("pca_epsilon", 1e-8)
    pca_variance_threshold = config.get("pca_variance_threshold", 0.95)

    rdms: dict[int, np.ndarray] = {}

    with h5py.File(h5_path, "r") as f:
        hs = f["hidden_states_mean_no_special"]
        n_stimuli, n_layers_plus_one, _dim = hs.shape

        logger.info(
            "Computing RDMs: %d stimuli, %d layers (incl. embedding layer)",
            n_stimuli,
            n_layers_plus_one,
        )

        # Validate stimulus count
        if n_stimuli != len(stimuli_meta):
            raise ValueError(
                f"HDF5 has {n_stimuli} stimuli but metadata has {len(stimuli_meta)}"
            )

        # Process one layer at a time to keep memory bounded
        for layer_idx in range(n_layers_plus_one):
            reps = hs[:, layer_idx, :]  # shape (N, D) — single layer read
            reps = np.array(reps, dtype=np.float64)

            reps = correct_anisotropy(
                reps,
                method=anisotropy_method,
                epsilon=pca_epsilon,
                variance_threshold=pca_variance_threshold,
            )

            rdms[layer_idx] = compute_rdm(reps, metric="cosine")

    return rdms


# ---------------------------------------------------------------------------
# Model RDMs
# ---------------------------------------------------------------------------


def build_product_model_rdm(stimuli_meta: list[dict[str, Any]]) -> np.ndarray:
    """Build the product-identity model RDM.

    Dissimilarity encoding:
        - same product      → 0.0
        - same category      → 0.5
        - different category → 1.0

    Parameters
    ----------
    stimuli_meta : list[dict]
        Each dict must have ``product_id`` and ``category``.

    Returns
    -------
    np.ndarray
        Symmetric (N, N) model RDM.
    """
    n = len(stimuli_meta)
    rdm = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            if stimuli_meta[i]["product_id"] == stimuli_meta[j]["product_id"]:
                d = 0.0
            elif stimuli_meta[i]["category"] == stimuli_meta[j]["category"]:
                d = 0.5
            else:
                d = 1.0
            rdm[i, j] = d
            rdm[j, i] = d

    return rdm


def build_register_model_rdm(stimuli_meta: list[dict[str, Any]]) -> np.ndarray:
    """Build the register-identity model RDM.

    Dissimilarity encoding:
        - same register      → 0
        - different register → 1

    Parameters
    ----------
    stimuli_meta : list[dict]
        Each dict must have ``register``.

    Returns
    -------
    np.ndarray
        Symmetric (N, N) model RDM.
    """
    n = len(stimuli_meta)
    rdm = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0 if stimuli_meta[i]["register"] == stimuli_meta[j]["register"] else 1.0
            rdm[i, j] = d
            rdm[j, i] = d

    return rdm


def build_within_category_model_rdm(
    stimuli_meta: list[dict[str, Any]],
) -> np.ndarray:
    """Build the within-category product discrimination model RDM.

    Only within-category pairs are assigned meaningful distances:
        - same product (within same category)       → 0
        - different product, same category           → 1

    Cross-category pairs are set to NaN (excluded from correlation).

    Parameters
    ----------
    stimuli_meta : list[dict]
        Each dict must have ``product_id`` and ``category``.

    Returns
    -------
    np.ndarray
        Symmetric (N, N) model RDM with NaN for cross-category pairs.
    """
    n = len(stimuli_meta)
    rdm = np.full((n, n), np.nan, dtype=np.float64)

    for i in range(n):
        rdm[i, i] = 0.0
        for j in range(i + 1, n):
            if stimuli_meta[i]["category"] == stimuli_meta[j]["category"]:
                if stimuli_meta[i]["product_id"] == stimuli_meta[j]["product_id"]:
                    d = 0.0
                else:
                    d = 1.0
                rdm[i, j] = d
                rdm[j, i] = d

    return rdm


# ---------------------------------------------------------------------------
# Nuisance RDMs
# ---------------------------------------------------------------------------


def build_length_nuisance_rdm(stimuli_meta: list[dict[str, Any]]) -> np.ndarray:
    """Build a nuisance RDM based on pairwise token count differences.

    Parameters
    ----------
    stimuli_meta : list[dict]
        Each dict must have ``token_count`` (int).

    Returns
    -------
    np.ndarray
        Symmetric (N, N) RDM of absolute token count differences.
    """
    counts = np.array([s["token_count"] for s in stimuli_meta], dtype=np.float64)
    diff = np.abs(counts[:, None] - counts[None, :])
    return diff


def build_lexical_nuisance_rdm(stimuli_meta: list[dict[str, Any]]) -> np.ndarray:
    """Build a nuisance RDM based on Jaccard distance of token sets.

    Jaccard distance = 1 - |A ∩ B| / |A ∪ B|.

    Parameters
    ----------
    stimuli_meta : list[dict]
        Each dict must have ``text`` (str).

    Returns
    -------
    np.ndarray
        Symmetric (N, N) RDM of Jaccard distances.
    """
    n = len(stimuli_meta)
    # Build token sets (simple whitespace split, lowercased)
    token_sets = [set(s["text"].lower().split()) for s in stimuli_meta]

    rdm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union == 0:
                d = 0.0
            else:
                d = 1.0 - intersection / union
            rdm[i, j] = d
            rdm[j, i] = d

    return rdm


# ---------------------------------------------------------------------------
# RSA Correlation
# ---------------------------------------------------------------------------


def _upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """Extract the upper triangle of a symmetric matrix as a 1-D vector."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def rsa_correlation(
    observed_rdm: np.ndarray,
    model_rdm: np.ndarray,
) -> float:
    """Compute Spearman rank correlation between two RDMs.

    Compares the upper triangles of both matrices.  If the model RDM
    contains NaN values (e.g., within-category model), only non-NaN pairs
    are correlated.

    Parameters
    ----------
    observed_rdm : np.ndarray
        (N, N) observed dissimilarity matrix.
    model_rdm : np.ndarray
        (N, N) model dissimilarity matrix (may contain NaN).

    Returns
    -------
    float
        Spearman r value.
    """
    obs_vec = _upper_triangle(observed_rdm)
    mod_vec = _upper_triangle(model_rdm)

    # Handle NaN entries in the model (e.g., within-category model)
    valid = ~np.isnan(mod_vec) & ~np.isnan(obs_vec)
    if valid.sum() < 3:
        return 0.0

    r, _ = spearmanr(obs_vec[valid], mod_vec[valid])
    return float(r)


def partial_rsa(
    observed_rdm: np.ndarray,
    model_rdm: np.ndarray,
    nuisance_rdms: list[np.ndarray],
) -> float:
    """Compute partial Spearman RSA controlling for nuisance RDMs.

    Regresses out the nuisance RDM vectors from both observed and model
    upper triangles using ordinary least squares, then correlates the
    residuals with Spearman rank correlation.

    Parameters
    ----------
    observed_rdm : np.ndarray
        (N, N) observed dissimilarity matrix.
    model_rdm : np.ndarray
        (N, N) model dissimilarity matrix.
    nuisance_rdms : list[np.ndarray]
        List of (N, N) nuisance dissimilarity matrices.

    Returns
    -------
    float
        Partial Spearman r.
    """
    obs_vec = _upper_triangle(observed_rdm)
    mod_vec = _upper_triangle(model_rdm)

    # Build nuisance design matrix
    nuisance_vecs = np.column_stack(
        [_upper_triangle(nrd) for nrd in nuisance_rdms]
    )  # shape (n_pairs, n_nuisance)

    # Add intercept
    n_pairs = obs_vec.shape[0]
    X = np.column_stack([np.ones(n_pairs), nuisance_vecs])

    # Regress out nuisance from observed
    beta_obs, _, _, _ = np.linalg.lstsq(X, obs_vec, rcond=None)
    residual_obs = obs_vec - X @ beta_obs

    # Regress out nuisance from model
    beta_mod, _, _, _ = np.linalg.lstsq(X, mod_vec, rcond=None)
    residual_mod = mod_vec - X @ beta_mod

    # Correlate residuals
    if np.std(residual_obs) < 1e-12 or np.std(residual_mod) < 1e-12:
        return 0.0

    r, _ = spearmanr(residual_obs, residual_mod)
    return float(r)


# ---------------------------------------------------------------------------
# Condition Similarities
# ---------------------------------------------------------------------------


def compute_condition_similarities(
    h5_path: str,
    stimuli_meta: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, list[float]]:
    """Compute mean cosine similarity for SP-DR, DP-SC, and DC conditions.

    Conditions:
        - SP-DR: Same Product, Different Register
        - DP-SC: Different Product, Same Category
        - DC: Different Category

    Reads one layer at a time from HDF5.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 with ``/hidden_states_mean_no_special``.
    stimuli_meta : list[dict]
        Each dict must have ``product_id``, ``category``, ``register``.
    config : dict
        Pipeline configuration (used for anisotropy settings).

    Returns
    -------
    dict[str, list[float]]
        Keys "SP-DR", "DP-SC", "DC", each mapping to a list of per-layer
        mean cosine similarities.
    """
    n = len(stimuli_meta)

    # Pre-classify all pairs
    sp_dr_pairs: list[tuple[int, int]] = []
    dp_sc_pairs: list[tuple[int, int]] = []
    dc_pairs: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            same_product = stimuli_meta[i]["product_id"] == stimuli_meta[j]["product_id"]
            same_category = stimuli_meta[i]["category"] == stimuli_meta[j]["category"]
            same_register = stimuli_meta[i]["register"] == stimuli_meta[j]["register"]

            if same_product and not same_register:
                sp_dr_pairs.append((i, j))
            elif not same_product and same_category:
                dp_sc_pairs.append((i, j))
            elif not same_category:
                dc_pairs.append((i, j))

    logger.info(
        "Condition pairs — SP-DR: %d, DP-SC: %d, DC: %d",
        len(sp_dr_pairs),
        len(dp_sc_pairs),
        len(dc_pairs),
    )

    anisotropy_method = config.get("anisotropy_methods", ["none"])[0]
    pca_epsilon = config.get("pca_epsilon", 1e-8)
    pca_variance_threshold = config.get("pca_variance_threshold", 0.95)

    results: dict[str, list[float]] = {"SP-DR": [], "DP-SC": [], "DC": []}

    with h5py.File(h5_path, "r") as f:
        hs = f["hidden_states_mean_no_special"]
        n_stimuli, n_layers_plus_one, _dim = hs.shape

        for layer_idx in range(n_layers_plus_one):
            reps = np.array(hs[:, layer_idx, :], dtype=np.float64)

            reps = correct_anisotropy(
                reps,
                method=anisotropy_method,
                epsilon=pca_epsilon,
                variance_threshold=pca_variance_threshold,
            )

            # Compute full cosine similarity matrix for this layer
            # cosine_sim = 1 - cosine_distance
            cosine_dists = squareform(pdist(reps, metric="cosine"))
            cosine_sims = 1.0 - cosine_dists

            # Mean similarity per condition
            for label, pairs in [
                ("SP-DR", sp_dr_pairs),
                ("DP-SC", dp_sc_pairs),
                ("DC", dc_pairs),
            ]:
                if pairs:
                    sims = [cosine_sims[i, j] for i, j in pairs]
                    results[label].append(float(np.mean(sims)))
                else:
                    results[label].append(0.0)

    return results


# ---------------------------------------------------------------------------
# Permutation Testing
# ---------------------------------------------------------------------------


def run_permutation_test_tiered(
    rdms: dict[int, np.ndarray],
    model_rdm: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run tiered permutation testing: fast screen + full test on top layers.

    Tier 1 — Fast screen (200 permutations) at all layers to identify
    candidates.  Tier 2 — Full test (10,000 permutations) at the top-k
    layers by RSA magnitude from the screen.

    The model RDM labels are shuffled (rows + columns simultaneously)
    on each permutation.

    Parameters
    ----------
    rdms : dict[int, np.ndarray]
        Mapping from layer index to observed (N, N) RDM.
    model_rdm : np.ndarray
        (N, N) model RDM.
    config : dict
        Must contain ``screen_permutations``, ``full_permutations``,
        ``top_k_layers_for_full_test``, and ``seed``.

    Returns
    -------
    dict[str, Any]
        {
            "observed_rsa": {layer: r},
            "screen_pvalues": {layer: p},
            "screen_pvalues_fdr": {layer: p_corrected},
            "full_pvalues": {layer: p},
            "full_null_distributions": {layer: np.ndarray},
        }
    """
    rng = np.random.default_rng(config.get("seed", 42))
    screen_n = config.get("screen_permutations", 200)
    full_n = config.get("full_permutations", 10000)
    top_k = config.get("top_k_layers_for_full_test", 5)

    n = model_rdm.shape[0]
    layer_indices = sorted(rdms.keys())

    # Compute observed RSA for all layers
    observed_rsa: dict[int, float] = {}
    for layer_idx in layer_indices:
        observed_rsa[layer_idx] = rsa_correlation(rdms[layer_idx], model_rdm)

    logger.info("Tier 1: screening %d layers with %d permutations", len(layer_indices), screen_n)

    # ---- Tier 1: Fast screen ----
    screen_pvalues: dict[int, float] = {}

    for layer_idx in layer_indices:
        obs_r = observed_rsa[layer_idx]
        count_ge = 0

        for _ in range(screen_n):
            perm = rng.permutation(n)
            shuffled_model = model_rdm[np.ix_(perm, perm)]
            perm_r = rsa_correlation(rdms[layer_idx], shuffled_model)
            if perm_r >= obs_r:
                count_ge += 1

        screen_pvalues[layer_idx] = (count_ge + 1) / (screen_n + 1)

    # FDR correction on screen p-values
    screen_layers_sorted = sorted(screen_pvalues.keys())
    raw_ps = np.array([screen_pvalues[l] for l in screen_layers_sorted])
    fdr_ps = apply_fdr_correction(raw_ps)
    screen_pvalues_fdr: dict[int, float] = {
        l: float(p) for l, p in zip(screen_layers_sorted, fdr_ps)
    }

    # ---- Tier 2: Full test on top-k layers ----
    # Select top-k layers by absolute RSA magnitude
    top_layers = sorted(layer_indices, key=lambda l: abs(observed_rsa[l]), reverse=True)[:top_k]

    logger.info("Tier 2: full test on %d layers with %d permutations", len(top_layers), full_n)

    full_pvalues: dict[int, float] = {}
    full_null_distributions: dict[int, np.ndarray] = {}

    for layer_idx in top_layers:
        obs_r = observed_rsa[layer_idx]
        null_dist = np.zeros(full_n, dtype=np.float64)
        count_ge = 0

        for perm_i in range(full_n):
            perm = rng.permutation(n)
            shuffled_model = model_rdm[np.ix_(perm, perm)]
            perm_r = rsa_correlation(rdms[layer_idx], shuffled_model)
            null_dist[perm_i] = perm_r
            if perm_r >= obs_r:
                count_ge += 1

        full_pvalues[layer_idx] = (count_ge + 1) / (full_n + 1)
        full_null_distributions[layer_idx] = null_dist

    return {
        "observed_rsa": observed_rsa,
        "screen_pvalues": screen_pvalues,
        "screen_pvalues_fdr": screen_pvalues_fdr,
        "full_pvalues": full_pvalues,
        "full_null_distributions": full_null_distributions,
    }


# ---------------------------------------------------------------------------
# FDR Correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------


def apply_fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to a vector of p-values.

    Parameters
    ----------
    pvalues : np.ndarray
        1-D array of uncorrected p-values.

    Returns
    -------
    np.ndarray
        1-D array of FDR-corrected p-values (same order as input).
    """
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=np.float64)

    # Sort p-values and track original indices
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]

    # Benjamini-Hochberg adjustment
    # adjusted_p[i] = p[i] * m / rank[i], then enforce monotonicity from right
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity: walk backward, take cumulative min
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(m, dtype=np.float64)
    result[sorted_idx] = adjusted

    return result


# ---------------------------------------------------------------------------
# RSA Sanity Check
# ---------------------------------------------------------------------------


def rsa_sanity_check(
    rsa_curves: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Triage RSA curves to decide whether to proceed, investigate, or flag.

    Three-way classification:
        - **flat**: no peak anywhere (max – min < 0.02 or std < 0.01)
        - **weak**: middle-layer peak exists but RSA r < 0.05
        - **clear**: middle-layer peak with RSA r >= 0.1

    The "middle" zone is defined by ``config["h3_layer_zone_pct"]`` which
    gives (start%, end%) of total layers.

    Parameters
    ----------
    rsa_curves : dict[str, np.ndarray]
        Keys are model-RDM names (e.g. "product_identity"), values are 1-D
        arrays of per-layer RSA r values.
    config : dict
        Must contain ``h1_min_rsa_r`` and ``h3_layer_zone_pct``.

    Returns
    -------
    dict[str, Any]
        {
            "overall_verdict": "flat" | "weak" | "clear",
            "per_model": {
                name: {
                    "verdict": str,
                    "peak_layer": int,
                    "peak_r": float,
                    "in_middle_zone": bool,
                }
            },
        }
    """
    min_rsa_r = config.get("h1_min_rsa_r", 0.1)
    zone_pct = config.get("h3_layer_zone_pct", (10, 70))

    per_model: dict[str, dict[str, Any]] = {}
    verdicts: list[str] = []

    for name, curve in rsa_curves.items():
        n_layers = len(curve)
        zone_start = int(n_layers * zone_pct[0] / 100)
        zone_end = int(n_layers * zone_pct[1] / 100)

        peak_layer = int(np.argmax(curve))
        peak_r = float(curve[peak_layer])
        in_middle = zone_start <= peak_layer <= zone_end

        curve_range = float(np.max(curve) - np.min(curve))
        curve_std = float(np.std(curve))

        # Classify
        if curve_range < 0.02 or curve_std < 0.01:
            verdict = "flat"
        elif peak_r < 0.05:
            verdict = "weak"
        elif peak_r >= min_rsa_r:
            verdict = "clear"
        else:
            verdict = "weak"

        per_model[name] = {
            "verdict": verdict,
            "peak_layer": peak_layer,
            "peak_r": peak_r,
            "in_middle_zone": in_middle,
        }
        verdicts.append(verdict)

    # Overall verdict: worst case
    if "flat" in verdicts:
        overall = "flat"
    elif "weak" in verdicts:
        overall = "weak"
    else:
        overall = "clear"

    return {
        "overall_verdict": overall,
        "per_model": per_model,
    }
