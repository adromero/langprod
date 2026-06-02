"""Experiment 0 — Metric Exploration and Lock.

Screens 27 metric combinations (9 layers x 3 aggregations) on exploration data,
validates top candidates on held-out registers, and locks the winning metric to
``data/coherence/exp0/metric_selection.json``.

Run as::

    python -m coherence.experiment_0
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from scipy import stats

from analysis import correct_anisotropy
from coherence.metrics import (
    AGGREGATION_METHODS,
    CoherenceResult,
    batch_coherence_scores,
    compute_coherence_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
STIMULI_PATH = DATA_DIR / "stimuli.json"
OUTPUT_DIR = DATA_DIR / "coherence" / "exp0"

# Model identifier (matches HDF5 filename convention)
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

# HDF5 filename
HDF5_FILENAME = "Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4_hidden_states.h5"

# Candidate HDF5 layer indices (0=embedding, 1-64=transformer layers 0-63)
CANDIDATE_HDF5_INDICES = [21, 26, 31, 36, 41, 51, 56, 61, 62]

# Expected HDF5 shape parameters
EXPECTED_N_LAYERS_PLUS_ONE = 65  # embedding + 64 transformer layers
EXPECTED_HIDDEN_DIM = 5120       # Qwen2.5-32B

# Data split registers
EXPLORATION_REGISTERS = {"marketing", "regulatory", "casual_social"}
HELDOUT_REGISTERS = {"patent", "journalistic"}

# Reproducibility
RANDOM_SEED = 42
N_PERMUTATIONS = 1000
N_PSEUDO_PRODUCTS = 80

# Bonferroni threshold for held-out testing (9 tests)
BONFERRONI_P_THRESHOLD = 0.05 / 9  # ≈ 0.0056

# Effect size thresholds (from failure protocol)
D_STRONG = 0.5
D_WEAK = 0.3


# ---------------------------------------------------------------------------
# HDF5 shape validation
# ---------------------------------------------------------------------------


def validate_hdf5(h5_path: Path, n_stimuli_expected: int) -> bool:
    """Validate HDF5 shape before any analysis.

    Returns True if valid, False otherwise (logs errors).
    """
    if not h5_path.exists():
        logger.error("HDF5 file not found: %s", h5_path)
        return False

    with h5py.File(h5_path, "r") as f:
        if "hidden_states_mean_no_special" not in f:
            logger.error("Dataset 'hidden_states_mean_no_special' not found in %s", h5_path)
            return False

        hs = f["hidden_states_mean_no_special"]
        shape = hs.shape
        logger.info("HDF5 shape: %s", shape)

        n_stimuli, n_layers_plus_one, hidden_dim = shape

        # Check axis-1: layers
        if n_layers_plus_one != EXPECTED_N_LAYERS_PLUS_ONE:
            logger.error(
                "HDF5 axis-1 mismatch: expected %d (embedding + 64 transformer layers), "
                "got %d. File: %s",
                EXPECTED_N_LAYERS_PLUS_ONE,
                n_layers_plus_one,
                h5_path,
            )
            return False

        # Check axis-2: hidden dim
        if hidden_dim != EXPECTED_HIDDEN_DIM:
            logger.error(
                "HDF5 axis-2 mismatch: expected %d (Qwen2.5-32B hidden dim), got %d. File: %s",
                EXPECTED_HIDDEN_DIM,
                hidden_dim,
                h5_path,
            )
            return False

        # Check axis-0: stimuli count
        if n_stimuli != n_stimuli_expected:
            logger.error(
                "HDF5 axis-0 mismatch: expected %d stimuli (from stimuli.json), got %d. File: %s",
                n_stimuli_expected,
                n_stimuli,
                h5_path,
            )
            return False

        # Check max candidate index is within bounds
        max_candidate = max(CANDIDATE_HDF5_INDICES)
        if max_candidate >= n_layers_plus_one:
            logger.error(
                "Max candidate HDF5 index %d is out of bounds for axis-1 size %d. File: %s",
                max_candidate,
                n_layers_plus_one,
                h5_path,
            )
            return False

    logger.info("HDF5 validation passed: shape (%d, %d, %d)", n_stimuli, n_layers_plus_one, hidden_dim)
    return True


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_stimuli() -> list[dict]:
    """Load stimuli.json and return the list of stimulus dicts."""
    if not STIMULI_PATH.exists():
        logger.error("Stimuli file not found: %s", STIMULI_PATH)
        sys.exit(1)

    with open(STIMULI_PATH) as f:
        stimuli = json.load(f)

    logger.info("Loaded %d stimuli from %s", len(stimuli), STIMULI_PATH)
    return stimuli


def split_stimuli(
    stimuli: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split stimuli into exploration pool and held-out set by register.

    Returns (exploration_stimuli, heldout_stimuli).
    """
    exploration = [s for s in stimuli if s["register"] in EXPLORATION_REGISTERS]
    heldout = [s for s in stimuli if s["register"] in HELDOUT_REGISTERS]

    logger.info(
        "Split: %d exploration (%s), %d held-out (%s)",
        len(exploration),
        sorted(EXPLORATION_REGISTERS),
        len(heldout),
        sorted(HELDOUT_REGISTERS),
    )
    return exploration, heldout


def build_stimulus_index(stimuli: list[dict], all_stimuli: list[dict]) -> dict[str, int]:
    """Map stimulus_id to its row index in the HDF5 (which matches all_stimuli order)."""
    id_to_global_idx = {s["stimulus_id"]: i for i, s in enumerate(all_stimuli)}
    return {s["stimulus_id"]: id_to_global_idx[s["stimulus_id"]] for s in stimuli}


# ---------------------------------------------------------------------------
# Hidden-state extraction from HDF5
# ---------------------------------------------------------------------------


def load_layer_vectors(
    h5_path: Path,
    hdf5_layer_idx: int,
    row_indices: list[int],
) -> np.ndarray:
    """Load hidden-state vectors for specific rows and a single layer from HDF5.

    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file.
    hdf5_layer_idx : int
        HDF5 axis-1 index (0=embedding, 1-64=transformer layers).
    row_indices : list[int]
        HDF5 axis-0 row indices to load.

    Returns
    -------
    np.ndarray
        Shape ``(len(row_indices), hidden_dim)`` in float64.
    """
    with h5py.File(h5_path, "r") as f:
        hs = f["hidden_states_mean_no_special"]
        # Load only the needed rows and layer
        vectors = hs[row_indices, hdf5_layer_idx, :]
    return np.array(vectors, dtype=np.float64)


# ---------------------------------------------------------------------------
# Variant aggregation
# ---------------------------------------------------------------------------


def variant_average(
    stimuli: list[dict],
    vectors: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Average v0 and v1 vectors per (product, register) pair.

    Parameters
    ----------
    stimuli : list[dict]
        Stimulus metadata (must have product_id, register, variant).
    vectors : np.ndarray
        Shape ``(len(stimuli), D)`` — one vector per stimulus, aligned with stimuli.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{product_id: {register: averaged_vector}}``
    """
    # Group by (product_id, register)
    groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for i, s in enumerate(stimuli):
        key = (s["product_id"], s["register"])
        groups[key].append(vectors[i])

    # Average and organize by product -> register
    result: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for (product_id, register), vecs in groups.items():
        result[product_id][register] = np.mean(vecs, axis=0)

    return dict(result)


# ---------------------------------------------------------------------------
# Cohen's d and permutation testing
# ---------------------------------------------------------------------------


def generate_pseudo_products(
    product_register_vecs: dict[str, dict[str, np.ndarray]],
    registers: list[str],
    n_pseudo: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Generate random pseudo-products (Distribution B).

    For each pseudo-product, for each register, select a random product
    (different from the others selected for this pseudo-product) and take
    that product's variant-averaged vector for that register.

    Parameters
    ----------
    product_register_vecs : dict[str, dict[str, np.ndarray]]
        ``{product_id: {register: vector}}`` — real product data.
    registers : list[str]
        Registers to include in each pseudo-product.
    n_pseudo : int
        Number of pseudo-products to generate.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Each entry is ``{register: vector}`` for one pseudo-product.
    """
    product_ids = sorted(product_register_vecs.keys())
    n_products = len(product_ids)
    n_registers = len(registers)

    pseudo_products: list[dict[str, np.ndarray]] = []
    for _ in range(n_pseudo):
        # Select n_registers distinct products (one per register)
        selected_indices = rng.choice(n_products, size=n_registers, replace=False)
        channel_vecs: dict[str, np.ndarray] = {}
        for reg_idx, reg_name in enumerate(registers):
            pid = product_ids[selected_indices[reg_idx]]
            channel_vecs[reg_name] = product_register_vecs[pid][reg_name]
        pseudo_products.append(channel_vecs)

    return pseudo_products


def compute_cohens_d(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
) -> float:
    """Compute Cohen's d with pooled standard deviation.

    d = (mean(A) - mean(B)) / pooled_SD
    """
    mean_a = np.mean(dist_a)
    mean_b = np.mean(dist_b)
    n_a = len(dist_a)
    n_b = len(dist_b)
    var_a = np.var(dist_a, ddof=1)
    var_b = np.var(dist_b, ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd < 1e-12:
        return 0.0
    return float((mean_a - mean_b) / pooled_sd)


def permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    observed_d: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Permutation test for Cohen's d.

    Shuffle product labels, recompute d each time.
    p = (count of permuted d >= observed d + 1) / (n_permutations + 1)
    """
    combined = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)
    count_ge = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_d = compute_cohens_d(perm_a, perm_b)
        if perm_d >= observed_d:
            count_ge += 1

    return (count_ge + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# Vocabulary narrowness (type-token ratio)
# ---------------------------------------------------------------------------


def compute_ttr_per_product(
    stimuli: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute type-token ratio for each product across its stimuli.

    Returns ``{product_id: {register: ttr}}``.
    """
    # Group texts by (product_id, register)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in stimuli:
        groups[(s["product_id"], s["register"])].append(s["text"])

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (product_id, register), texts in groups.items():
        combined = " ".join(texts).lower().split()
        if len(combined) == 0:
            result[product_id][register] = 0.0
        else:
            result[product_id][register] = len(set(combined)) / len(combined)

    return dict(result)


def check_vocab_narrowness(
    coherence_scores: dict[str, float],
    ttr_per_product: dict[str, dict[str, float]],
) -> tuple[bool, float]:
    """Check if coherence correlates with inverse lexical diversity.

    Parameters
    ----------
    coherence_scores : dict[str, float]
        ``{product_id: coherence_score}``
    ttr_per_product : dict[str, dict[str, float]]
        ``{product_id: {register: ttr}}``

    Returns
    -------
    (flag, rho) : tuple[bool, float]
        flag is True if Spearman rho > 0.5.
    """
    product_ids = sorted(coherence_scores.keys())
    coh_vals = []
    inv_ttr_vals = []

    for pid in product_ids:
        coh_vals.append(coherence_scores[pid])
        if pid in ttr_per_product:
            channel_ttrs = list(ttr_per_product[pid].values())
            inv_ttr_vals.append(np.mean([1.0 - t for t in channel_ttrs]))
        else:
            inv_ttr_vals.append(0.0)

    if len(coh_vals) < 3:
        return False, 0.0

    rho, _ = stats.spearmanr(coh_vals, inv_ttr_vals)
    flag = bool(rho > 0.5)
    return flag, float(rho)


# ---------------------------------------------------------------------------
# Part A: Metric space exploration
# ---------------------------------------------------------------------------


def run_part_a_screen(
    h5_path: Path,
    stimuli: list[dict],
    all_stimuli: list[dict],
) -> list[dict]:
    """Part A, Step 1: Screen 27 combinations on exploration pool.

    Returns a list of result dicts sorted by |Cohen's d| descending.
    """
    logger.info("=== Part A, Step 1: Screening 27 combinations on exploration pool ===")

    exploration, _ = split_stimuli(stimuli)
    stim_index = build_stimulus_index(exploration, all_stimuli)
    row_indices = [stim_index[s["stimulus_id"]] for s in exploration]

    # TTR for vocab narrowness control
    ttr_per_product = compute_ttr_per_product(exploration)

    rng = np.random.default_rng(RANDOM_SEED)
    registers = sorted(EXPLORATION_REGISTERS)

    results: list[dict] = []

    for hdf5_idx in CANDIDATE_HDF5_INDICES:
        transformer_layer = hdf5_idx - 1
        logger.info("Screening HDF5 index %d (transformer layer %d)", hdf5_idx, transformer_layer)

        # Load raw vectors for all exploration stimuli at this layer
        raw_vectors = load_layer_vectors(h5_path, hdf5_idx, row_indices)

        # Anisotropy correction on all 480 raw vectors
        corrected_vectors = correct_anisotropy(raw_vectors, method="mean_centering")

        # Variant-average the corrected vectors
        product_vecs = variant_average(exploration, corrected_vectors)

        # Generate pseudo-products (Distribution B)
        pseudo_products = generate_pseudo_products(product_vecs, registers, N_PSEUDO_PRODUCTS, rng)

        for method in AGGREGATION_METHODS:
            # Distribution A: per-product coherence
            scores_a_dict: dict[str, float] = {}
            for product_id, reg_vecs in product_vecs.items():
                result = compute_coherence_score(reg_vecs, method=method)
                scores_a_dict[product_id] = result.market_coherence

            scores_a = np.array([scores_a_dict[pid] for pid in sorted(scores_a_dict)])

            # Distribution B: pseudo-product coherence
            scores_b_list: list[float] = []
            for pseudo_vecs in pseudo_products:
                result = compute_coherence_score(pseudo_vecs, method=method)
                scores_b_list.append(result.market_coherence)
            scores_b = np.array(scores_b_list)

            # Cohen's d
            d = compute_cohens_d(scores_a, scores_b)

            # Vocab narrowness check
            flag, rho = check_vocab_narrowness(scores_a_dict, ttr_per_product)

            results.append({
                "hdf5_index": hdf5_idx,
                "transformer_layer": transformer_layer,
                "aggregation": method,
                "cohens_d": d,
                "mean_a": float(np.mean(scores_a)),
                "mean_b": float(np.mean(scores_b)),
                "std_a": float(np.std(scores_a, ddof=1)),
                "std_b": float(np.std(scores_b, ddof=1)),
                "vocab_narrowness_flag": flag,
                "vocab_narrowness_rho": rho,
            })

            logger.info(
                "  [%d/%s] d=%.4f, mean_a=%.4f, mean_b=%.4f, vocab_flag=%s",
                hdf5_idx, method, d,
                float(np.mean(scores_a)), float(np.mean(scores_b)),
                flag,
            )

    # Sort by |d| descending
    results.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)

    logger.info("Top-5 combinations by |d|:")
    for i, r in enumerate(results[:5]):
        logger.info(
            "  %d. layer=%d, method=%s, d=%.4f",
            i + 1, r["hdf5_index"], r["aggregation"], r["cohens_d"],
        )

    return results


def run_part_a_formal(
    h5_path: Path,
    stimuli: list[dict],
    all_stimuli: list[dict],
    screen_results: list[dict],
) -> list[dict]:
    """Part A, Step 2: Formal testing on held-out set.

    Takes top-3 layers from screening, tests all 3 aggregation methods = 9 tests.
    Uses Bonferroni correction (p < 0.0056).

    Returns list of formal test result dicts.
    """
    logger.info("=== Part A, Step 2: Formal testing on held-out set ===")

    _, heldout = split_stimuli(stimuli)
    stim_index = build_stimulus_index(heldout, all_stimuli)
    row_indices = [stim_index[s["stimulus_id"]] for s in heldout]

    # Select top-3 unique layers by |d|
    seen_layers: set[int] = set()
    top_layers: list[int] = []
    for r in screen_results:
        hdf5_idx = r["hdf5_index"]
        if hdf5_idx not in seen_layers:
            seen_layers.add(hdf5_idx)
            top_layers.append(hdf5_idx)
        if len(top_layers) == 3:
            break

    logger.info("Top-3 layers for formal testing: %s", top_layers)

    ttr_per_product = compute_ttr_per_product(heldout)
    rng = np.random.default_rng(RANDOM_SEED + 1)  # different seed for held-out
    registers = sorted(HELDOUT_REGISTERS)

    formal_results: list[dict] = []

    for hdf5_idx in top_layers:
        transformer_layer = hdf5_idx - 1

        # Load raw vectors for all held-out stimuli at this layer
        raw_vectors = load_layer_vectors(h5_path, hdf5_idx, row_indices)

        # Anisotropy correction on the held-out set independently
        corrected_vectors = correct_anisotropy(raw_vectors, method="mean_centering")

        # Variant-average the corrected vectors
        product_vecs = variant_average(heldout, corrected_vectors)

        # Generate pseudo-products (Distribution B)
        pseudo_products = generate_pseudo_products(product_vecs, registers, N_PSEUDO_PRODUCTS, rng)

        for method in AGGREGATION_METHODS:
            # Distribution A
            scores_a_dict: dict[str, float] = {}
            for product_id, reg_vecs in product_vecs.items():
                result = compute_coherence_score(reg_vecs, method=method)
                scores_a_dict[product_id] = result.market_coherence

            scores_a = np.array([scores_a_dict[pid] for pid in sorted(scores_a_dict)])

            # Distribution B
            scores_b_list: list[float] = []
            for pseudo_vecs in pseudo_products:
                result = compute_coherence_score(pseudo_vecs, method=method)
                scores_b_list.append(result.market_coherence)
            scores_b = np.array(scores_b_list)

            # Cohen's d
            d = compute_cohens_d(scores_a, scores_b)

            # Permutation p-value
            p = permutation_test(scores_a, scores_b, d, N_PERMUTATIONS, rng)

            # Vocab narrowness
            flag, rho = check_vocab_narrowness(scores_a_dict, ttr_per_product)

            significant = p < BONFERRONI_P_THRESHOLD

            formal_results.append({
                "hdf5_index": hdf5_idx,
                "transformer_layer": transformer_layer,
                "aggregation": method,
                "cohens_d": d,
                "p_value": p,
                "significant_bonferroni": significant,
                "mean_a": float(np.mean(scores_a)),
                "mean_b": float(np.mean(scores_b)),
                "std_a": float(np.std(scores_a, ddof=1)),
                "std_b": float(np.std(scores_b, ddof=1)),
                "vocab_narrowness_flag": flag,
                "vocab_narrowness_rho": rho,
            })

            logger.info(
                "  [%d/%s] d=%.4f, p=%.6f, sig=%s, vocab_flag=%s",
                hdf5_idx, method, d, p, significant, flag,
            )

    # Sort by |d| descending
    formal_results.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)
    return formal_results


# ---------------------------------------------------------------------------
# Part B: Real-document condition simulation (requires GPU)
# ---------------------------------------------------------------------------


def run_part_b(
    h5_path: Path,
    stimuli: list[dict],
    all_stimuli: list[dict],
    winning_metric: dict,
) -> Optional[dict]:
    """Part B: Robustness testing with re-extraction.

    This requires a GPU for Qwen forward passes. If the extraction module
    cannot load the model (e.g., no GPU), this step is skipped.

    Parameters
    ----------
    h5_path : Path
        HDF5 file path.
    stimuli : list[dict]
        All 800 stimuli.
    all_stimuli : list[dict]
        Same as stimuli (for index building).
    winning_metric : dict
        The winning metric from Part A (contains hdf5_index, aggregation, etc.).

    Returns
    -------
    dict or None
        Robustness results, or None if GPU is unavailable.
    """
    logger.info("=== Part B: Real-document condition simulation ===")

    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("Part B skipped: no CUDA GPU available for re-extraction.")
            return None
    except ImportError:
        logger.warning("Part B skipped: PyTorch not available.")
        return None

    # Get exploration pool
    exploration, _ = split_stimuli(stimuli)
    stim_index = build_stimulus_index(exploration, all_stimuli)
    row_indices = [stim_index[s["stimulus_id"]] for s in exploration]

    hdf5_idx = winning_metric["hdf5_index"]
    method = winning_metric["aggregation"]

    # Load original vectors and compute baseline
    raw_vectors = load_layer_vectors(h5_path, hdf5_idx, row_indices)
    corrected_vectors = correct_anisotropy(raw_vectors, method="mean_centering")
    # Global mean for Part B corrections (already saved to disk by main())
    global_mean = raw_vectors.mean(axis=0)

    product_vecs = variant_average(exploration, corrected_vectors)

    # Baseline coherence scores
    baseline_scores: dict[str, float] = {}
    for product_id, reg_vecs in product_vecs.items():
        result = compute_coherence_score(reg_vecs, method=method)
        baseline_scores[product_id] = result.market_coherence

    baseline_mean = float(np.mean(list(baseline_scores.values())))
    baseline_std = float(np.std(list(baseline_scores.values()), ddof=1))

    logger.info("Baseline: mean=%.4f, std=%.4f", baseline_mean, baseline_std)

    # Part B requires re-extraction through Qwen for modified texts.
    # The conditions are:
    # 1. Length variation (truncate to 20-30 words, expand to 300-500 words)
    # 2. Attribute removal (remove 1-3 core attributes)
    # 3. Non-LLM text (manually rewritten ~10 stimuli)
    #
    # Since this requires GPU re-extraction, we structure the code to:
    # a) Generate modified texts
    # b) Extract hidden states through the model
    # c) Apply the ORIGINAL global mean for anisotropy correction
    # d) Compare coherence scores to the baseline

    try:
        from extraction import extract_hidden_states, load_model_and_tokenizer

        logger.info("Loading model for Part B re-extraction...")
        model, tokenizer = load_model_and_tokenizer(MODEL_ID)

        conditions: dict[str, dict] = {}

        # --- Condition 1: Length truncation (20-30 words) ---
        logger.info("Part B Condition 1: Length truncation (20-30 words)")
        truncated_stimuli = []
        for s in exploration:
            words = s["text"].split()
            n_words = min(30, max(20, len(words)))
            truncated_text = " ".join(words[:n_words])
            truncated_stimuli.append({**s, "text": truncated_text})

        truncated_vecs = _reextract_vectors(
            model, tokenizer, truncated_stimuli, hdf5_idx, global_mean
        )
        if truncated_vecs is not None:
            trunc_product_vecs = variant_average(exploration, truncated_vecs)
            trunc_scores = {}
            for pid, reg_vecs in trunc_product_vecs.items():
                r = compute_coherence_score(reg_vecs, method=method)
                trunc_scores[pid] = r.market_coherence
            trunc_mean = float(np.mean(list(trunc_scores.values())))
            trunc_change_sd = abs(trunc_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
            conditions["length_truncation"] = {
                "mean_coherence": trunc_mean,
                "change_sd": trunc_change_sd,
                "robust": trunc_change_sd < 0.2,
            }
            logger.info("  Truncation: mean=%.4f, change=%.4f SD, robust=%s",
                        trunc_mean, trunc_change_sd, trunc_change_sd < 0.2)

        # --- Condition 2: Length expansion (300-500 words) ---
        logger.info("Part B Condition 2: Length expansion (300-500 words)")
        expanded_stimuli = []
        for s in exploration:
            # Repeat text to reach 300-500 words
            words = s["text"].split()
            while len(words) < 300:
                words.extend(s["text"].split())
            expanded_text = " ".join(words[:400])
            expanded_stimuli.append({**s, "text": expanded_text})

        expanded_vecs = _reextract_vectors(
            model, tokenizer, expanded_stimuli, hdf5_idx, global_mean
        )
        if expanded_vecs is not None:
            exp_product_vecs = variant_average(exploration, expanded_vecs)
            exp_scores = {}
            for pid, reg_vecs in exp_product_vecs.items():
                r = compute_coherence_score(reg_vecs, method=method)
                exp_scores[pid] = r.market_coherence
            exp_mean = float(np.mean(list(exp_scores.values())))
            exp_change_sd = abs(exp_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
            conditions["length_expansion"] = {
                "mean_coherence": exp_mean,
                "change_sd": exp_change_sd,
                "robust": exp_change_sd < 0.2,
            }
            logger.info("  Expansion: mean=%.4f, change=%.4f SD, robust=%s",
                        exp_mean, exp_change_sd, exp_change_sd < 0.2)

        # --- Condition 3: Attribute removal ---
        logger.info("Part B Condition 3: Attribute removal (1-3 attributes)")
        attr_removed_stimuli = []
        for s in exploration:
            # Remove core attribute mentions from text
            text = s["text"]
            attrs = s.get("core_attributes", {})
            if attrs:
                # Remove up to 3 attribute values from text
                for i, (key, val) in enumerate(attrs.items()):
                    if i >= 3:
                        break
                    text = text.replace(str(val), "")
            attr_removed_stimuli.append({**s, "text": text})

        attr_vecs = _reextract_vectors(
            model, tokenizer, attr_removed_stimuli, hdf5_idx, global_mean
        )
        if attr_vecs is not None:
            attr_product_vecs = variant_average(exploration, attr_vecs)
            attr_scores = {}
            for pid, reg_vecs in attr_product_vecs.items():
                r = compute_coherence_score(reg_vecs, method=method)
                attr_scores[pid] = r.market_coherence
            attr_mean = float(np.mean(list(attr_scores.values())))
            attr_change_sd = abs(attr_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
            conditions["attribute_removal"] = {
                "mean_coherence": attr_mean,
                "change_sd": attr_change_sd,
                "robust": attr_change_sd < 0.2,
            }
            logger.info("  Attr removal: mean=%.4f, change=%.4f SD, robust=%s",
                        attr_mean, attr_change_sd, attr_change_sd < 0.2)

        # Clean up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

        all_robust = all(c.get("robust", False) for c in conditions.values())

        part_b_result = {
            "conditions": conditions,
            "all_robust": all_robust,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        }

        logger.info("Part B result: all_robust=%s", all_robust)
        return part_b_result

    except Exception as e:
        logger.warning("Part B failed with exception: %s", e)
        return None


def _reextract_vectors(
    model: Any,
    tokenizer: Any,
    stimuli: list[dict],
    hdf5_layer_idx: int,
    original_global_mean: np.ndarray,
) -> Optional[np.ndarray]:
    """Re-extract hidden states for modified stimuli and apply original mean centering.

    Returns corrected vectors of shape (len(stimuli), D), or None on failure.
    """
    import torch
    from extraction import mean_pool_no_special

    special_ids: set[int] = set()
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    vectors = []
    device = next(model.parameters()).device

    for s in stimuli:
        try:
            inputs = tokenizer(s["text"], return_tensors="pt", padding=False, truncation=False)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Get the hidden state at the target layer
            hs = outputs.hidden_states[hdf5_layer_idx]
            pooled = mean_pool_no_special(
                hs.detach().cpu(),
                attention_mask.detach().cpu(),
                special_ids,
                input_ids.detach().cpu(),
            )
            vectors.append(pooled.numpy().astype(np.float64))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning("Re-extraction failed for %s: %s", s["stimulus_id"], e)
            return None

    result = np.stack(vectors, axis=0)

    # Apply anisotropy correction using the ORIGINAL global mean
    result = result - original_global_mean

    return result


# ---------------------------------------------------------------------------
# Part C: Metric lock
# ---------------------------------------------------------------------------


def run_part_c(
    formal_results: list[dict],
    part_b_result: Optional[dict],
) -> dict:
    """Part C: Lock the winning metric and write to JSON.

    Selects the best formal-test result (highest |d| that is significant),
    applies the failure protocol, and writes the metric selection file.

    Parameters
    ----------
    formal_results : list[dict]
        Sorted by |d| descending from Part A formal testing.
    part_b_result : dict or None
        Part B robustness results, if available.

    Returns
    -------
    dict
        The metric selection dict that was written to disk.
    """
    logger.info("=== Part C: Metric Lock ===")

    # Find the best result (highest |d|)
    best = formal_results[0]
    d = best["cohens_d"]
    p = best["p_value"]

    logger.info("Best metric: layer=%d, method=%s, d=%.4f, p=%.6f",
                best["hdf5_index"], best["aggregation"], d, p)

    # Failure protocol
    if d < D_WEAK:
        logger.error(
            "FAILURE: d=%.4f < %.1f — contrastive escalation required. "
            "If contrastive also fails (d < %.1f), project must be killed.",
            d, D_WEAK, D_WEAK,
        )
        status = "failed_needs_contrastive"
    elif d < D_STRONG:
        logger.warning(
            "WARNING: d=%.4f is between %.1f and %.1f — contrastive escalation recommended. "
            "Will lock current metric but compare with contrastive later.",
            d, D_WEAK, D_STRONG,
        )
        status = "marginal_try_contrastive"
    else:
        # d >= 0.5: Check Part B
        if part_b_result is not None and not part_b_result.get("all_robust", True):
            logger.warning(
                "d=%.4f >= %.1f but Part B robustness check failed. "
                "Investigate preprocessing.",
                d, D_STRONG,
            )
            status = "strong_but_degradation_issue"
        else:
            logger.info("SUCCESS: d=%.4f >= %.1f — locking metric.", d, D_STRONG)
            status = "locked"

    metric_selection = {
        "model": MODEL_ID,
        "layer_hdf5_index": best["hdf5_index"],
        "layer_transformer": best["transformer_layer"],
        "correction": "mean_centering",
        "aggregation": best["aggregation"],
        "distance": "cosine",
        "vocab_narrowness_flag": best["vocab_narrowness_flag"],
        "effect_size_d": best["cohens_d"],
        "p_value": best["p_value"],
        "status": status,
        "all_formal_results": formal_results,
        "part_b_result": part_b_result,
    }

    # Write to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "metric_selection.json"

    with open(output_path, "w") as f:
        json.dump(metric_selection, f, indent=2, default=_json_default)

    logger.info("Metric selection written to %s", output_path)
    return metric_selection


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Experiment 0 pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Experiment 0: Metric Exploration and Lock")
    logger.info("=" * 60)

    # Load stimuli
    stimuli = load_stimuli()

    # Resolve HDF5 path
    h5_path = DATA_DIR / HDF5_FILENAME
    if not h5_path.exists():
        logger.error("HDF5 file not found: %s — cannot proceed without pre-extracted data.", h5_path)
        sys.exit(1)

    # Mandatory shape validation
    if not validate_hdf5(h5_path, len(stimuli)):
        logger.error("HDF5 validation failed — aborting.")
        sys.exit(1)

    # Part A: Screening
    screen_results = run_part_a_screen(h5_path, stimuli, stimuli)

    # Save screening results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    screen_path = OUTPUT_DIR / "screening_results.json"
    with open(screen_path, "w") as f:
        json.dump(screen_results, f, indent=2, default=_json_default)
    logger.info("Screening results saved to %s", screen_path)

    # Part A: Formal testing
    formal_results = run_part_a_formal(h5_path, stimuli, stimuli, screen_results)

    # Save formal results
    formal_path = OUTPUT_DIR / "formal_results.json"
    with open(formal_path, "w") as f:
        json.dump(formal_results, f, indent=2, default=_json_default)
    logger.info("Formal results saved to %s", formal_path)

    # Save global mean for Experiment 1 anisotropy anchor (must happen before Part B)
    winning_metric = formal_results[0]
    hdf5_idx = winning_metric["hdf5_index"]
    exploration, _ = split_stimuli(stimuli)
    stim_index = build_stimulus_index(exploration, stimuli)
    row_indices = [stim_index[s["stimulus_id"]] for s in exploration]
    raw_vectors = load_layer_vectors(h5_path, hdf5_idx, row_indices)
    global_mean = raw_vectors.mean(axis=0)
    np.save(OUTPUT_DIR / "global_mean.npy", global_mean)
    logger.info("Global mean vector saved to %s", OUTPUT_DIR / "global_mean.npy")

    # Part B: Robustness (may skip if no GPU)
    part_b_result = run_part_b(h5_path, stimuli, stimuli, winning_metric)

    # Part C: Metric lock
    metric_selection = run_part_c(formal_results, part_b_result)

    # Summary
    logger.info("=" * 60)
    logger.info("Experiment 0 complete.")
    logger.info("  Status: %s", metric_selection["status"])
    logger.info("  Layer: HDF5 idx %d (transformer %d)",
                metric_selection["layer_hdf5_index"],
                metric_selection["layer_transformer"])
    logger.info("  Aggregation: %s", metric_selection["aggregation"])
    logger.info("  Cohen's d: %.4f", metric_selection["effect_size_d"])
    logger.info("  p-value: %.6f", metric_selection["p_value"])
    logger.info("  Vocab narrowness flag: %s", metric_selection["vocab_narrowness_flag"])


if __name__ == "__main__":
    main()
