"""Experiment 1 -- Real-Document Sensitivity.

The core validation experiment and primary go/no-go gate. Tests whether the
locked metric from Experiment 0 can distinguish known-consistent from
known-inconsistent real products using hidden-state embeddings.

Run as::

    python -m coherence.experiment_1
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

from coherence.baselines import compute_sbert_coherence, compute_tfidf_coherence
from coherence.ingest import (
    RealDocument,
    clean_document,
    load_documents,
    normalize_document_length,
    to_stimuli_format,
    truncate_document,
    validate_product_set,
)
from coherence.metrics import CoherenceResult, compute_coherence_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
EXP0_DIR = DATA_DIR / "coherence" / "exp0"
EXP1_DIR = DATA_DIR / "coherence" / "exp1"
PORTFOLIO_DIR = DATA_DIR / "coherence" / "portfolios" / "exp1"

METRIC_SELECTION_PATH = EXP0_DIR / "metric_selection.json"
GLOBAL_MEAN_PATH = EXP0_DIR / "global_mean.npy"

# Model identifier (matches Experiment 0)
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

# Product group sizes
N_CONSISTENT = 10
N_INCONSISTENT = 10
N_TOTAL = N_CONSISTENT + N_INCONSISTENT

# Statistical thresholds
MANN_WHITNEY_P_THRESHOLD = 0.05
COHENS_D_PASS = 1.0
COHENS_D_SUGGESTIVE = 0.8
ROC_AUC_THRESHOLD = 0.85
MAX_MISCLASSIFICATIONS = 2
DELONG_P_THRESHOLD = 0.10

# AUC diagnostic bands (failure protocol)
AUC_FUNDAMENTAL_FAILURE = 0.70
AUC_MARGINAL = 0.85


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_gates() -> tuple[dict, np.ndarray]:
    """Verify that Experiment 0 outputs exist and are loadable.

    Returns
    -------
    (metric_selection, global_mean) :
        The locked metric configuration and the global mean vector
        for anisotropy correction.

    Raises
    ------
    SystemExit
        If either gate file is missing or unloadable.
    """
    if not METRIC_SELECTION_PATH.exists():
        logger.error(
            "Gate failed: %s not found. Run Experiment 0 first.", METRIC_SELECTION_PATH
        )
        sys.exit(1)

    if not GLOBAL_MEAN_PATH.exists():
        logger.error(
            "Gate failed: %s not found. Run Experiment 0 first.", GLOBAL_MEAN_PATH
        )
        sys.exit(1)

    with open(METRIC_SELECTION_PATH) as f:
        metric_selection = json.load(f)

    global_mean = np.load(GLOBAL_MEAN_PATH)

    logger.info(
        "Gates passed: metric=%s layer=%d, global_mean shape=%s",
        metric_selection["aggregation"],
        metric_selection["layer_hdf5_index"],
        global_mean.shape,
    )

    return metric_selection, global_mean


# ---------------------------------------------------------------------------
# Document loading and preparation
# ---------------------------------------------------------------------------


def load_and_prepare_documents() -> tuple[
    dict[str, list[RealDocument]],
    dict[str, bool],
]:
    """Load real documents from the portfolio directory and validate them.

    Returns
    -------
    (product_docs, product_labels)
        product_docs: ``{product_id: [RealDocument, ...]}``
        product_labels: ``{product_id: True}`` for consistent, ``False`` for inconsistent.

    Raises
    ------
    SystemExit
        If the portfolio directory is missing or validation fails critically.
    """
    if not PORTFOLIO_DIR.exists():
        logger.error(
            "Portfolio directory not found: %s. "
            "Populate with product documents before running Experiment 1.",
            PORTFOLIO_DIR,
        )
        sys.exit(1)

    documents = load_documents(PORTFOLIO_DIR)
    if not documents:
        logger.error("No documents loaded from %s.", PORTFOLIO_DIR)
        sys.exit(1)

    # Clean, truncate, and normalize length
    cleaned: list[RealDocument] = []
    for doc in documents:
        doc = clean_document(doc)
        doc = truncate_document(doc)
        cleaned.append(doc)

    # Normalize all documents to 512 tokens to remove length as a confound.
    # Exp 0 Part B showed length variation shifts coherence scores by ~3 SD.
    cleaned = normalize_document_length(cleaned, target_tokens=512)

    # Validate product sets
    validation = validate_product_set(cleaned)

    # Load labels from manifest
    manifest_path = PORTFOLIO_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error(
            "Manifest not found: %s. "
            "Create a manifest.json with 'consistent' and 'inconsistent' product lists.",
            manifest_path,
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    consistent_ids = set(manifest.get("consistent", []))
    inconsistent_ids = set(manifest.get("inconsistent", []))

    logger.info(
        "Manifest: %d consistent, %d inconsistent",
        len(consistent_ids),
        len(inconsistent_ids),
    )

    # Build product_docs and labels
    product_docs: dict[str, list[RealDocument]] = defaultdict(list)
    for doc in cleaned:
        product_docs[doc.product_id].append(doc)

    product_labels: dict[str, bool] = {}
    for pid in product_docs:
        if pid in consistent_ids:
            product_labels[pid] = True
        elif pid in inconsistent_ids:
            product_labels[pid] = False
        else:
            logger.warning(
                "Product %s not in manifest consistent or inconsistent list; skipping.",
                pid,
            )

    # Filter to only labeled products
    product_docs = {
        pid: docs for pid, docs in product_docs.items() if pid in product_labels
    }

    n_consistent = sum(1 for v in product_labels.values() if v)
    n_inconsistent = sum(1 for v in product_labels.values() if not v)
    logger.info(
        "Labeled products: %d consistent, %d inconsistent, %d total",
        n_consistent,
        n_inconsistent,
        len(product_labels),
    )

    # Warn but do not abort if counts differ from expected
    if n_consistent != N_CONSISTENT:
        logger.warning(
            "Expected %d consistent products, got %d", N_CONSISTENT, n_consistent
        )
    if n_inconsistent != N_INCONSISTENT:
        logger.warning(
            "Expected %d inconsistent products, got %d",
            N_INCONSISTENT,
            n_inconsistent,
        )

    return dict(product_docs), product_labels


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_product_embeddings(
    product_docs: dict[str, list[RealDocument]],
    metric_selection: dict,
    global_mean: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract hidden-state embeddings for all products and apply anisotropy correction.

    For each product, each document is run through the model to obtain a
    hidden-state vector at the locked layer. The global mean from Experiment 0
    is subtracted (NOT the Experiment 1 batch mean) to maintain comparability
    with Experiment 0 calibration.

    Parameters
    ----------
    product_docs :
        ``{product_id: [RealDocument, ...]}``
    metric_selection :
        Locked metric from Experiment 0.
    global_mean :
        Global mean vector from Experiment 0 for anisotropy correction.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{product_id: {channel: embedding_vector}}``
    """
    import torch
    from extraction import load_model_and_tokenizer, mean_pool_no_special

    hdf5_layer_idx = metric_selection["layer_hdf5_index"]

    logger.info("Loading model %s for extraction...", MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)

    special_ids: set[int] = set()
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    device = next(model.parameters()).device

    product_embeddings: dict[str, dict[str, np.ndarray]] = {}

    for product_id, docs in product_docs.items():
        # Group documents by channel, average embeddings per channel
        channel_vectors: dict[str, list[np.ndarray]] = defaultdict(list)

        for doc in docs:
            stim = to_stimuli_format(doc)
            text = stim["text"]

            inputs = tokenizer(
                text, return_tensors="pt", padding=False, truncation=False
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            hs = outputs.hidden_states[hdf5_layer_idx]
            pooled = mean_pool_no_special(
                hs.detach().cpu(),
                attention_mask.detach().cpu(),
                special_ids,
                input_ids.detach().cpu(),
            )
            raw_vec = pooled.numpy().astype(np.float64)

            # CRITICAL: Subtract Exp 0 global mean (NOT Exp 1 batch mean)
            corrected_vec = raw_vec - global_mean

            channel_vectors[doc.channel].append(corrected_vec)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average vectors per channel
        channel_embeddings: dict[str, np.ndarray] = {}
        for channel, vecs in channel_vectors.items():
            channel_embeddings[channel] = np.mean(vecs, axis=0)

        product_embeddings[product_id] = channel_embeddings
        logger.info(
            "Product %s: %d channels (%s)",
            product_id,
            len(channel_embeddings),
            ", ".join(sorted(channel_embeddings.keys())),
        )

    # Clean up GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return product_embeddings


# ---------------------------------------------------------------------------
# Coherence scoring
# ---------------------------------------------------------------------------


def compute_product_coherences(
    product_embeddings: dict[str, dict[str, np.ndarray]],
    method: str,
) -> dict[str, CoherenceResult]:
    """Compute brand and market coherence for all products.

    Parameters
    ----------
    product_embeddings :
        ``{product_id: {channel: embedding_vector}}``
    method :
        Aggregation method from the locked metric.

    Returns
    -------
    dict[str, CoherenceResult]
    """
    results: dict[str, CoherenceResult] = {}
    for product_id, channel_embs in product_embeddings.items():
        if len(channel_embs) < 2:
            logger.warning(
                "Product %s has only %d channel(s); cannot compute coherence.",
                product_id,
                len(channel_embs),
            )
            continue
        results[product_id] = compute_coherence_score(channel_embs, method=method)
        logger.info(
            "Product %s: brand=%.4f, market=%.4f",
            product_id,
            results[product_id].brand_coherence
            if results[product_id].brand_coherence is not None
            else float("nan"),
            results[product_id].market_coherence,
        )
    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d with pooled standard deviation.

    d = (mean(A) - mean(B)) / pooled_SD
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    n_a = len(group_a)
    n_b = len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd < 1e-12:
        return 0.0
    return float((mean_a - mean_b) / pooled_sd)


def rank_biserial_correlation(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U statistic.

    r = 1 - 2U / (n1 * n2)
    """
    return float(1.0 - 2.0 * u_stat / (n1 * n2))


def compute_roc_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, int, float]:
    """Compute ROC AUC and misclassification count.

    Parameters
    ----------
    scores :
        Coherence scores (higher = more consistent).
    labels :
        Binary labels (1 = consistent, 0 = inconsistent).

    Returns
    -------
    (auc, misclassifications, optimal_threshold)
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(labels, scores))

    # Find optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])

    # Misclassification count at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    misclassifications = int(np.sum(predictions != labels))

    return auc, misclassifications, optimal_threshold


# ---------------------------------------------------------------------------
# DeLong test for comparing AUCs
# ---------------------------------------------------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute mid-ranks for ties in a sorted array.

    Parameters
    ----------
    x :
        Input array.

    Returns
    -------
    np.ndarray
        Array of mid-ranks (1-based).
    """
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        # Average rank for tied values (1-based)
        avg_rank = 0.5 * (i + j + 1)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _auc_variance_components(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample structural components for the DeLong AUC variance.

    Uses the placement-value formulation: for each positive sample, its
    structural component is the fraction of negative samples ranked below it,
    and vice versa.

    Parameters
    ----------
    scores :
        Predicted scores.
    labels :
        Binary labels (1 = positive, 0 = negative).

    Returns
    -------
    (v_pos, v_neg) :
        Placement values for positive and negative samples.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # Structural component for positive samples: V_{10}(X_j)
    # = fraction of negatives with score < X_j (using mid-ranks)
    combined = np.concatenate([pos_scores, neg_scores])
    ranks = _compute_midrank(combined)

    pos_ranks = ranks[:n_pos]
    neg_ranks = ranks[n_pos:]

    # V_{10,j} = (R_j - j) / n_neg for positives (j is 1-based among positives)
    # But we use the combined rank. The placement value for positive i is
    # (rank_i - (position among positives)) / n_neg
    # Simplified: V_10 = (ranks_pos - np.arange(1, n_pos+1)) / n_neg
    v_pos = (pos_ranks - np.arange(1, n_pos + 1, dtype=np.float64)) / n_neg

    # V_{01,k} for negatives:
    # Sort negatives by their score, use combined ranks
    v_neg = (neg_ranks - np.arange(1, n_neg + 1, dtype=np.float64)) / n_pos

    return v_pos, v_neg


def delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    """DeLong test for comparing two correlated AUCs.

    Implements DeLong et al. (1988) for comparing the areas under two
    correlated ROC curves derived from the same set of samples.

    Parameters
    ----------
    scores_a :
        Scores from method A (the proposed method).
    scores_b :
        Scores from method B (the baseline).
    labels :
        Binary labels (1 = positive, 0 = negative).

    Returns
    -------
    (z_stat, p_value_two_tailed, p_value_one_tailed)
        z_stat: standard normal test statistic
        p_value_two_tailed: two-tailed p-value
        p_value_one_tailed: one-tailed p-value (H1: AUC_a > AUC_b)
    """
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(labels, scores_a)
    auc_b = roc_auc_score(labels, scores_b)

    # Get structural components for each method
    v_pos_a, v_neg_a = _auc_variance_components(scores_a, labels)
    v_pos_b, v_neg_b = _auc_variance_components(scores_b, labels)

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    # Covariance matrix of (AUC_a, AUC_b)
    # S = (1/n_pos) * cov(V10_a, V10_b) + (1/n_neg) * cov(V01_a, V01_b)
    # Guard: np.cov requires >= 2 observations for ddof=1; fall back to zero
    # covariance when we have only 1 sample per class.
    if n_pos < 2 or n_neg < 2:
        # Cannot estimate variance with < 2 samples per class
        return 0.0, 1.0, 0.5

    cov_pos = np.cov(v_pos_a, v_pos_b)  # 2x2
    cov_neg = np.cov(v_neg_a, v_neg_b)  # 2x2

    # Variance of (AUC_a - AUC_b)
    # var = S[0,0] + S[1,1] - 2*S[0,1]
    # where S = cov_pos/n_pos + cov_neg/n_neg
    s_matrix = cov_pos / n_pos + cov_neg / n_neg
    var_diff = s_matrix[0, 0] + s_matrix[1, 1] - 2 * s_matrix[0, 1]

    if not np.isfinite(var_diff) or var_diff < 1e-12:
        # AUCs are identical, variance is degenerate, or NaN from small samples
        return 0.0, 1.0, 0.5

    z_stat = float((auc_a - auc_b) / np.sqrt(var_diff))
    p_two = float(2.0 * stats.norm.sf(abs(z_stat)))
    # One-tailed: H1: AUC_a > AUC_b
    p_one = float(stats.norm.sf(z_stat))

    return z_stat, p_two, p_one


# ---------------------------------------------------------------------------
# Baseline scoring
# ---------------------------------------------------------------------------


def compute_baseline_scores(
    product_docs: dict[str, list[RealDocument]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute TF-IDF and SBERT coherence for all products.

    For each product, concatenates all documents per channel into a single
    text, then passes the channel-text dict to the baseline functions.

    Returns
    -------
    (tfidf_scores, sbert_scores)
        Each is ``{product_id: market_coherence}``.
    """
    tfidf_scores: dict[str, float] = {}
    sbert_scores: dict[str, float] = {}

    for product_id, docs in product_docs.items():
        # Build channel_texts: {channel: concatenated_text}
        channel_texts: dict[str, str] = defaultdict(str)
        for doc in docs:
            if channel_texts[doc.channel]:
                channel_texts[doc.channel] += " " + doc.text
            else:
                channel_texts[doc.channel] = doc.text

        channel_texts = dict(channel_texts)

        if len(channel_texts) < 2:
            logger.warning(
                "Product %s has only %d channel(s); skipping baselines.",
                product_id,
                len(channel_texts),
            )
            continue

        try:
            tfidf_result = compute_tfidf_coherence(channel_texts)
            tfidf_scores[product_id] = tfidf_result.market_coherence
        except Exception as e:
            logger.warning("TF-IDF failed for %s: %s", product_id, e)

        try:
            sbert_result = compute_sbert_coherence(channel_texts)
            sbert_scores[product_id] = sbert_result.market_coherence
        except Exception as e:
            logger.warning("SBERT failed for %s: %s", product_id, e)

    return tfidf_scores, sbert_scores


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    coherence_results: dict[str, CoherenceResult],
    product_labels: dict[str, bool],
    tfidf_scores: dict[str, float],
    sbert_scores: dict[str, float],
) -> dict[str, Any]:
    """Run the full statistical analysis suite.

    Parameters
    ----------
    coherence_results :
        Hidden-state coherence results per product.
    product_labels :
        Ground-truth labels (True = consistent, False = inconsistent).
    tfidf_scores :
        TF-IDF baseline market coherence per product.
    sbert_scores :
        SBERT baseline market coherence per product.

    Returns
    -------
    dict
        Complete analysis results including all statistical tests and the verdict.
    """
    # Align scores and labels
    product_ids = sorted(coherence_results.keys())
    scores_arr = np.array([coherence_results[pid].market_coherence for pid in product_ids])
    labels_arr = np.array([1 if product_labels[pid] else 0 for pid in product_ids])

    consistent_scores = scores_arr[labels_arr == 1]
    inconsistent_scores = scores_arr[labels_arr == 0]

    n_consistent = len(consistent_scores)
    n_inconsistent = len(inconsistent_scores)

    logger.info(
        "Analysis: %d consistent (mean=%.4f), %d inconsistent (mean=%.4f)",
        n_consistent,
        float(np.mean(consistent_scores)) if n_consistent > 0 else 0.0,
        n_inconsistent,
        float(np.mean(inconsistent_scores)) if n_inconsistent > 0 else 0.0,
    )

    # 1. Mann-Whitney U test
    u_stat, mw_p = stats.mannwhitneyu(
        consistent_scores, inconsistent_scores, alternative="greater"
    )
    r_rb = rank_biserial_correlation(float(u_stat), n_consistent, n_inconsistent)

    logger.info(
        "Mann-Whitney U: U=%.2f, p=%.6f, rank-biserial r=%.4f",
        u_stat, mw_p, r_rb,
    )

    # 2. Cohen's d
    d = compute_cohens_d(consistent_scores, inconsistent_scores)
    logger.info("Cohen's d: %.4f", d)

    # 3. ROC AUC and misclassifications
    auc, misclass, threshold = compute_roc_auc(scores_arr, labels_arr)
    logger.info(
        "ROC AUC: %.4f, misclassifications: %d/%d, threshold: %.4f",
        auc, misclass, len(labels_arr), threshold,
    )

    # 4. Baselines ROC AUC
    baseline_results: dict[str, dict] = {}

    for baseline_name, baseline_scores_dict in [
        ("tfidf", tfidf_scores),
        ("sbert", sbert_scores),
    ]:
        baseline_pids = sorted(
            pid for pid in product_ids if pid in baseline_scores_dict
        )
        if len(baseline_pids) < len(product_ids):
            logger.warning(
                "Baseline %s missing %d products",
                baseline_name,
                len(product_ids) - len(baseline_pids),
            )

        if len(baseline_pids) >= 2:
            b_scores = np.array([baseline_scores_dict[pid] for pid in baseline_pids])
            b_labels = np.array(
                [1 if product_labels[pid] else 0 for pid in baseline_pids]
            )

            if len(np.unique(b_labels)) < 2:
                logger.warning(
                    "Baseline %s has only one class represented; skipping AUC.",
                    baseline_name,
                )
                baseline_results[baseline_name] = {
                    "auc": None,
                    "n_products": len(baseline_pids),
                }
                continue

            b_auc = float(roc_auc_score_safe(b_scores, b_labels))
            baseline_results[baseline_name] = {
                "auc": b_auc,
                "n_products": len(baseline_pids),
            }
            logger.info("Baseline %s AUC: %.4f", baseline_name, b_auc)
        else:
            baseline_results[baseline_name] = {
                "auc": None,
                "n_products": len(baseline_pids),
            }

    # 5. DeLong test vs baselines
    delong_results: dict[str, dict] = {}
    for baseline_name, baseline_scores_dict in [
        ("tfidf", tfidf_scores),
        ("sbert", sbert_scores),
    ]:
        common_pids = sorted(
            pid
            for pid in product_ids
            if pid in baseline_scores_dict
        )
        if len(common_pids) < 2:
            delong_results[baseline_name] = {
                "z_stat": None,
                "p_two_tailed": None,
                "p_one_tailed": None,
                "n_products": len(common_pids),
            }
            continue

        hs_scores = np.array(
            [coherence_results[pid].market_coherence for pid in common_pids]
        )
        bl_scores = np.array([baseline_scores_dict[pid] for pid in common_pids])
        dl_labels = np.array(
            [1 if product_labels[pid] else 0 for pid in common_pids]
        )

        if len(np.unique(dl_labels)) < 2:
            logger.warning(
                "DeLong test vs %s: only one class present; skipping.",
                baseline_name,
            )
            delong_results[baseline_name] = {
                "z_stat": None,
                "p_two_tailed": None,
                "p_one_tailed": None,
                "n_products": len(common_pids),
            }
            continue

        z, p_two, p_one = delong_test(hs_scores, bl_scores, dl_labels)
        delong_results[baseline_name] = {
            "z_stat": z,
            "p_two_tailed": p_two,
            "p_one_tailed": p_one,
            "n_products": len(common_pids),
        }
        logger.info(
            "DeLong vs %s: z=%.4f, p_one=%.6f",
            baseline_name, z, p_one,
        )

    # 6. Spearman rho across all products
    spearman_results: dict[str, dict] = {}
    for baseline_name, baseline_scores_dict in [
        ("tfidf", tfidf_scores),
        ("sbert", sbert_scores),
    ]:
        common_pids = sorted(
            pid for pid in product_ids if pid in baseline_scores_dict
        )
        if len(common_pids) >= 3:
            hs_vals = [coherence_results[pid].market_coherence for pid in common_pids]
            bl_vals = [baseline_scores_dict[pid] for pid in common_pids]
            rho, sp_p = stats.spearmanr(hs_vals, bl_vals)
            spearman_results[baseline_name] = {
                "rho": float(rho),
                "p_value": float(sp_p),
                "n": len(common_pids),
            }
            logger.info(
                "Spearman vs %s: rho=%.4f, p=%.6f", baseline_name, rho, sp_p
            )
        else:
            spearman_results[baseline_name] = {
                "rho": None,
                "p_value": None,
                "n": len(common_pids),
            }

    # 7. Per-product detail
    product_details: list[dict] = []
    for pid in product_ids:
        detail: dict[str, Any] = {
            "product_id": pid,
            "label": "consistent" if product_labels[pid] else "inconsistent",
            "market_coherence": coherence_results[pid].market_coherence,
            "brand_coherence": coherence_results[pid].brand_coherence,
        }
        if pid in tfidf_scores:
            detail["tfidf_coherence"] = tfidf_scores[pid]
        if pid in sbert_scores:
            detail["sbert_coherence"] = sbert_scores[pid]
        product_details.append(detail)

    # --- Verdict logic ---
    verdict = build_verdict(
        auc=auc,
        mw_p=float(mw_p),
        d=d,
        misclass=misclass,
        n_total=len(labels_arr),
        baseline_results=baseline_results,
        delong_results=delong_results,
    )

    return {
        "mann_whitney": {
            "u_statistic": float(u_stat),
            "p_value": float(mw_p),
            "rank_biserial_r": r_rb,
            "n_consistent": n_consistent,
            "n_inconsistent": n_inconsistent,
        },
        "cohens_d": {
            "d": d,
            "interpretation": _interpret_d(d, float(mw_p)),
        },
        "roc": {
            "auc": auc,
            "misclassifications": misclass,
            "n_total": len(labels_arr),
            "optimal_threshold": threshold,
        },
        "baselines": baseline_results,
        "delong": delong_results,
        "spearman_rank_order": spearman_results,
        "product_details": product_details,
        "verdict": verdict,
    }


def roc_auc_score_safe(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC, returning 0.5 if only one class is present."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _interpret_d(d: float, mw_p: float) -> str:
    """Interpret Cohen's d in the context of study power at n=10."""
    if d >= COHENS_D_PASS:
        return "pass"
    if d >= COHENS_D_SUGGESTIVE:
        if mw_p < MANN_WHITNEY_P_THRESHOLD:
            return "suggestive_but_significant"
        return "suggestive_underpowered"
    return "below_threshold"


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def build_verdict(
    *,
    auc: float,
    mw_p: float,
    d: float,
    misclass: int,
    n_total: int,
    baseline_results: dict[str, dict],
    delong_results: dict[str, dict],
) -> dict[str, Any]:
    """Build the pass/fail verdict for Experiment 1.

    Criteria:
        - AUC >= 0.85
        - Mann-Whitney p < 0.05
        - Misclassifications <= 2/20
        - DeLong p < 0.10 (one-tailed) vs at least one baseline

    Cohen's d interpretation:
        - d >= 1.0: PASS (powered)
        - 0.8 <= d < 1.0 AND p < 0.05: PASS (suggestive but significant)
        - 0.8 <= d < 1.0 AND p >= 0.05: SUGGESTIVE (underpowered)
        - d < 0.8: BELOW_THRESHOLD
    """
    criteria: dict[str, dict] = {}

    # AUC criterion
    auc_pass = auc >= ROC_AUC_THRESHOLD
    criteria["auc"] = {
        "value": auc,
        "threshold": ROC_AUC_THRESHOLD,
        "pass": auc_pass,
    }

    # Mann-Whitney p-value criterion
    mw_pass = mw_p < MANN_WHITNEY_P_THRESHOLD
    criteria["mann_whitney_p"] = {
        "value": mw_p,
        "threshold": MANN_WHITNEY_P_THRESHOLD,
        "pass": mw_pass,
    }

    # Misclassification criterion
    misclass_pass = misclass <= MAX_MISCLASSIFICATIONS
    criteria["misclassifications"] = {
        "value": misclass,
        "threshold": MAX_MISCLASSIFICATIONS,
        "n_total": n_total,
        "pass": misclass_pass,
    }

    # Cohen's d interpretation
    d_interp = _interpret_d(d, mw_p)
    d_pass = d_interp in ("pass", "suggestive_but_significant")
    criteria["cohens_d"] = {
        "value": d,
        "pass_threshold": COHENS_D_PASS,
        "suggestive_threshold": COHENS_D_SUGGESTIVE,
        "interpretation": d_interp,
        "pass": d_pass,
    }

    # DeLong criterion: p < 0.10 one-tailed vs at least one baseline
    delong_pass = False
    for bl_name, dl_result in delong_results.items():
        if dl_result.get("p_one_tailed") is not None:
            if dl_result["p_one_tailed"] < DELONG_P_THRESHOLD:
                delong_pass = True
                break

    criteria["delong_value_added"] = {
        "threshold": DELONG_P_THRESHOLD,
        "pass": delong_pass,
        "details": delong_results,
    }

    # Overall verdict
    # Primary criteria: AUC, Mann-Whitney, misclassifications
    # Cohen's d is interpreted with power caveats
    # DeLong is value-added (secondary)
    primary_pass = auc_pass and mw_pass and misclass_pass
    overall = "PASS" if primary_pass and delong_pass else "FAIL"

    # Suggestive-result protocol
    notes: list[str] = []
    if d_interp == "suggestive_underpowered":
        notes.append(
            "Effect size suggestive of separation (d={:.3f}) but study underpowered "
            "at n=10 to confirm statistically. Recommend collecting 5-7 additional "
            "products per group to reach n~17 (powered for d=0.8 at 80%) before "
            "making a final go/no-go decision.".format(d)
        )

    # If primary passes but DeLong fails
    if primary_pass and not delong_pass:
        overall = "PASS_NO_VALUE_ADDED"
        notes.append(
            "Primary criteria pass but hidden-state metric does not significantly "
            "outperform baselines by DeLong test. Consider adopting best baseline."
        )

    # Failure protocol diagnostics
    if not auc_pass:
        if auc < AUC_FUNDAMENTAL_FAILURE:
            notes.append(
                "FUNDAMENTAL FAILURE: AUC < {:.2f}. Debug anisotropy correction "
                "and register alignment.".format(AUC_FUNDAMENTAL_FAILURE)
            )
        elif auc < AUC_MARGINAL:
            notes.append(
                "MARGINAL: {:.2f} <= AUC < {:.2f}. Investigate confounds "
                "(channel leakage, label noise).".format(
                    AUC_FUNDAMENTAL_FAILURE, AUC_MARGINAL
                )
            )

    # Check if baselines match our AUC
    if auc_pass:
        for bl_name, bl_result in baseline_results.items():
            if bl_result.get("auc") is not None and bl_result["auc"] >= ROC_AUC_THRESHOLD:
                if not delong_pass:
                    notes.append(
                        f"Baseline {bl_name} achieves AUC >= {ROC_AUC_THRESHOLD} "
                        f"and DeLong test is not significant. Consider adopting "
                        f"baseline {bl_name}."
                    )

    return {
        "overall": overall,
        "criteria": criteria,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Collect channel texts (for baselines)
# ---------------------------------------------------------------------------


def collect_channel_texts(
    product_docs: dict[str, list[RealDocument]],
) -> dict[str, dict[str, str]]:
    """Build channel-text dicts for baseline methods.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{product_id: {channel: concatenated_text}}``
    """
    result: dict[str, dict[str, str]] = {}
    for product_id, docs in product_docs.items():
        channel_texts: dict[str, str] = defaultdict(str)
        for doc in docs:
            if channel_texts[doc.channel]:
                channel_texts[doc.channel] += " " + doc.text
            else:
                channel_texts[doc.channel] = doc.text
        result[product_id] = dict(channel_texts)
    return result


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


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
    """Run the full Experiment 1 pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Experiment 1: Real-Document Sensitivity")
    logger.info("=" * 60)

    # Gate checks
    metric_selection, global_mean = check_gates()

    # Load and prepare documents
    product_docs, product_labels = load_and_prepare_documents()

    # Extract embeddings using locked metric
    product_embeddings = extract_product_embeddings(
        product_docs, metric_selection, global_mean
    )

    # Compute hidden-state coherence scores
    method = metric_selection["aggregation"]
    coherence_results = compute_product_coherences(product_embeddings, method)

    # Compute baseline scores
    logger.info("Computing baseline scores...")
    tfidf_scores, sbert_scores = compute_baseline_scores(product_docs)

    # Run full analysis
    logger.info("Running statistical analysis...")
    results = run_analysis(coherence_results, product_labels, tfidf_scores, sbert_scores)

    # Write verdict and persist data for downstream experiments
    EXP1_DIR.mkdir(parents=True, exist_ok=True)

    # Persist per-product per-channel embeddings for Experiments 2-5
    embeddings_path = EXP1_DIR / "embeddings.npz"
    flat = {}
    for product_id, channel_embs in product_embeddings.items():
        for channel, vec in channel_embs.items():
            flat[f"{product_id}/{channel}"] = vec
    np.savez_compressed(embeddings_path, **flat)
    logger.info("Embeddings saved to %s (%d vectors)", embeddings_path, len(flat))

    # Persist product labels (consistent/inconsistent)
    labels_path = EXP1_DIR / "product_labels.json"
    with open(labels_path, "w") as f:
        json.dump(
            {pid: is_consistent for pid, is_consistent in product_labels.items()},
            f, indent=2,
        )
    logger.info("Product labels saved to %s", labels_path)

    verdict_path = EXP1_DIR / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    logger.info("Verdict written to %s", verdict_path)

    # Summary
    verdict = results["verdict"]
    logger.info("=" * 60)
    logger.info("Experiment 1 complete.")
    logger.info("  Overall verdict: %s", verdict["overall"])
    logger.info("  AUC: %.4f (pass=%s)", results["roc"]["auc"], verdict["criteria"]["auc"]["pass"])
    logger.info(
        "  Mann-Whitney p: %.6f (pass=%s)",
        results["mann_whitney"]["p_value"],
        verdict["criteria"]["mann_whitney_p"]["pass"],
    )
    logger.info(
        "  Cohen's d: %.4f (%s)",
        results["cohens_d"]["d"],
        results["cohens_d"]["interpretation"],
    )
    logger.info(
        "  Misclassifications: %d/%d (pass=%s)",
        results["roc"]["misclassifications"],
        results["roc"]["n_total"],
        verdict["criteria"]["misclassifications"]["pass"],
    )
    logger.info(
        "  DeLong value-added: %s",
        verdict["criteria"]["delong_value_added"]["pass"],
    )
    if verdict["notes"]:
        for note in verdict["notes"]:
            logger.info("  NOTE: %s", note)


if __name__ == "__main__":
    main()
