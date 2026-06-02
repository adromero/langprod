"""Experiment 2 -- Channel Attribution.

Reanalysis of Experiment 1 data. For each known-inconsistent product, identifies
the outlier channel (lowest mean cosine similarity to all other channels) and
compares against human-judged ground truth. No model loading or new document
collection -- operates entirely on pre-computed embeddings from Experiment 1.

Run as::

    python -m coherence.experiment_2
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from coherence.metrics import (
    OutlierResult,
    identify_outlier_channel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
EXP1_DIR = DATA_DIR / "coherence" / "exp1"

VERDICT_PATH = EXP1_DIR / "verdict.json"
EMBEDDINGS_PATH = EXP1_DIR / "embeddings.npz"
LABELS_PATH = EXP1_DIR / "product_labels.json"
ANNOTATIONS_PATH = EXP1_DIR / "outlier_annotations.json"

# Minimum number of product/channel vectors expected in embeddings.npz
MIN_VECTORS = 20

# Pass criterion: correct outlier identification for >= this many of 10 inconsistent products
PASS_THRESHOLD = 6

# Consistent-product outlier gap threshold: below this, we consider "no strong outlier"
CONSISTENT_GAP_THRESHOLD = 0.05

# Acceptable verdict values from Experiment 1
ACCEPTABLE_VERDICTS = {"PASS", "PASS_NO_VALUE_ADDED"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProductAttribution:
    """Attribution result for a single product."""

    product_id: str
    outlier_channel: str
    mean_similarities: dict[str, float]
    gap: float
    ground_truth_outlier: str | None = None
    correct: bool | None = None


@dataclass
class Experiment2Results:
    """Full results of Experiment 2."""

    # Inconsistent product analysis
    inconsistent_results: list[ProductAttribution]
    n_correct: int
    n_total_inconsistent: int
    accuracy: float
    pass_criterion: bool

    # Consistent product analysis (secondary)
    consistent_results: list[ProductAttribution]
    n_consistent_with_strong_outlier: int
    n_total_consistent: int
    consistent_gap_threshold: float

    # Overall
    verdict: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_gates() -> tuple[dict, np.lib.npyio.NpzFile, dict[str, bool]]:
    """Verify that Experiment 1 outputs exist and are valid.

    Returns
    -------
    (verdict, embeddings_npz, product_labels)

    Raises
    ------
    SystemExit
        If any gate file is missing, the verdict is not PASS/PASS_NO_VALUE_ADDED,
        or the embeddings contain fewer than MIN_VECTORS vectors.
    """
    # 1. Check verdict.json
    if not VERDICT_PATH.exists():
        logger.error("Gate failed: %s not found. Run Experiment 1 first.", VERDICT_PATH)
        sys.exit(1)

    with open(VERDICT_PATH) as f:
        verdict_data = json.load(f)

    overall = verdict_data.get("verdict", {}).get("overall", "UNKNOWN")
    if overall not in ACCEPTABLE_VERDICTS:
        logger.error(
            "Gate failed: Experiment 1 verdict is %s, expected one of %s.",
            overall,
            ACCEPTABLE_VERDICTS,
        )
        sys.exit(1)

    logger.info("Verdict gate passed: %s", overall)

    # 2. Check embeddings.npz
    if not EMBEDDINGS_PATH.exists():
        logger.error("Gate failed: %s not found. Run Experiment 1 first.", EMBEDDINGS_PATH)
        sys.exit(1)

    embeddings_npz = np.load(EMBEDDINGS_PATH)
    n_vectors = len(embeddings_npz.files)
    if n_vectors < MIN_VECTORS:
        logger.error(
            "Gate failed: embeddings.npz contains %d vectors, expected >= %d.",
            n_vectors,
            MIN_VECTORS,
        )
        sys.exit(1)

    logger.info("Embeddings gate passed: %d vectors", n_vectors)

    # 3. Check product_labels.json
    if not LABELS_PATH.exists():
        logger.error("Gate failed: %s not found. Run Experiment 1 first.", LABELS_PATH)
        sys.exit(1)

    with open(LABELS_PATH) as f:
        product_labels = json.load(f)

    logger.info(
        "Labels gate passed: %d products (%d consistent, %d inconsistent)",
        len(product_labels),
        sum(1 for v in product_labels.values() if v),
        sum(1 for v in product_labels.values() if not v),
    )

    return verdict_data, embeddings_npz, product_labels


# ---------------------------------------------------------------------------
# Embedding reconstruction
# ---------------------------------------------------------------------------


def reconstruct_embeddings(
    embeddings_npz: np.lib.npyio.NpzFile,
) -> dict[str, dict[str, np.ndarray]]:
    """Reconstruct per-product per-channel embedding dicts from flat .npz keys.

    Keys in the .npz file are ``"{product_id}/{channel}"``.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{product_id: {channel: embedding_vector}}``
    """
    product_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for key in embeddings_npz.files:
        product_id, channel = key.split("/", 1)
        product_embeddings.setdefault(product_id, {})[channel] = embeddings_npz[key]
    return product_embeddings


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_annotations(path: Path) -> dict[str, str]:
    """Load outlier annotations from JSON.

    Expected schema::

        {
            "annotations": [
                {
                    "product_id": "product_001",
                    "outlier_channel": "social",
                    "notes": "Social media tone is drastically different..."
                },
                ...
            ]
        }

    Returns
    -------
    dict[str, str]
        ``{product_id: outlier_channel}``

    Raises
    ------
    SystemExit
        If the file is missing or malformed.
    """
    if not path.exists():
        logger.error(
            "Ground truth annotations not found: %s. "
            "Create outlier_annotations.json with human-judged outlier channels "
            "BEFORE running analysis.",
            path,
        )
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if "annotations" not in data:
        logger.error(
            "Invalid annotations file: missing 'annotations' key in %s.", path
        )
        sys.exit(1)

    annotations: dict[str, str] = {}
    for entry in data["annotations"]:
        pid = entry.get("product_id")
        outlier = entry.get("outlier_channel")
        if pid is None or outlier is None:
            logger.warning(
                "Skipping incomplete annotation entry: %s", entry
            )
            continue
        annotations[pid] = outlier

    logger.info("Loaded %d outlier annotations from %s", len(annotations), path)
    return annotations


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_inconsistent_products(
    product_embeddings: dict[str, dict[str, np.ndarray]],
    product_labels: dict[str, bool],
    annotations: dict[str, str],
) -> list[ProductAttribution]:
    """Analyze inconsistent products: identify outlier channel, compare to ground truth.

    Parameters
    ----------
    product_embeddings :
        ``{product_id: {channel: embedding_vector}}``
    product_labels :
        ``{product_id: True/False}`` (True = consistent).
    annotations :
        ``{product_id: outlier_channel}`` ground truth.

    Returns
    -------
    list[ProductAttribution]
        One entry per inconsistent product with outlier identification results.
    """
    results: list[ProductAttribution] = []

    inconsistent_pids = sorted(
        pid for pid, is_consistent in product_labels.items() if not is_consistent
    )

    for pid in inconsistent_pids:
        if pid not in product_embeddings:
            logger.warning("Product %s not found in embeddings; skipping.", pid)
            continue

        channel_embs = product_embeddings[pid]
        if len(channel_embs) < 3:
            logger.warning(
                "Product %s has only %d channels; need >= 3 for outlier detection.",
                pid,
                len(channel_embs),
            )
            continue

        outlier_result: OutlierResult = identify_outlier_channel(channel_embs)

        gt_outlier = annotations.get(pid)
        correct = (
            outlier_result.outlier_channel == gt_outlier if gt_outlier is not None else None
        )

        attr = ProductAttribution(
            product_id=pid,
            outlier_channel=outlier_result.outlier_channel,
            mean_similarities=outlier_result.mean_similarities,
            gap=outlier_result.gap,
            ground_truth_outlier=gt_outlier,
            correct=correct,
        )
        results.append(attr)

        logger.info(
            "Product %s: predicted=%s, ground_truth=%s, correct=%s, gap=%.4f",
            pid,
            outlier_result.outlier_channel,
            gt_outlier,
            correct,
            outlier_result.gap,
        )

    return results


def analyze_consistent_products(
    product_embeddings: dict[str, dict[str, np.ndarray]],
    product_labels: dict[str, bool],
    gap_threshold: float = CONSISTENT_GAP_THRESHOLD,
) -> list[ProductAttribution]:
    """Analyze consistent products: verify no strong outlier channel.

    For consistent products, we expect all channels to be roughly equivalent
    (low gap between the lowest and second-lowest mean similarity).

    Parameters
    ----------
    product_embeddings :
        ``{product_id: {channel: embedding_vector}}``
    product_labels :
        ``{product_id: True/False}`` (True = consistent).
    gap_threshold :
        Gap below which we consider "no strong outlier".

    Returns
    -------
    list[ProductAttribution]
        One entry per consistent product.
    """
    results: list[ProductAttribution] = []

    consistent_pids = sorted(
        pid for pid, is_consistent in product_labels.items() if is_consistent
    )

    for pid in consistent_pids:
        if pid not in product_embeddings:
            logger.warning("Product %s not found in embeddings; skipping.", pid)
            continue

        channel_embs = product_embeddings[pid]
        if len(channel_embs) < 3:
            logger.warning(
                "Product %s has only %d channels; need >= 3 for outlier detection.",
                pid,
                len(channel_embs),
            )
            continue

        outlier_result: OutlierResult = identify_outlier_channel(channel_embs)

        attr = ProductAttribution(
            product_id=pid,
            outlier_channel=outlier_result.outlier_channel,
            mean_similarities=outlier_result.mean_similarities,
            gap=outlier_result.gap,
            ground_truth_outlier=None,
            correct=None,
        )
        results.append(attr)

        has_strong_outlier = outlier_result.gap >= gap_threshold
        logger.info(
            "Product %s (consistent): gap=%.4f, strong_outlier=%s",
            pid,
            outlier_result.gap,
            has_strong_outlier,
        )

    return results


# ---------------------------------------------------------------------------
# Build experiment results
# ---------------------------------------------------------------------------


def build_results(
    inconsistent_results: list[ProductAttribution],
    consistent_results: list[ProductAttribution],
    gap_threshold: float = CONSISTENT_GAP_THRESHOLD,
) -> Experiment2Results:
    """Combine analysis results and compute pass/fail verdict.

    Parameters
    ----------
    inconsistent_results :
        Attribution results for inconsistent products.
    consistent_results :
        Attribution results for consistent products.
    gap_threshold :
        Gap threshold for "strong outlier" in consistent products.

    Returns
    -------
    Experiment2Results
    """
    # Inconsistent products: count correct identifications
    n_correct = sum(
        1 for r in inconsistent_results if r.correct is True
    )
    n_total_inconsistent = len(inconsistent_results)
    accuracy = n_correct / n_total_inconsistent if n_total_inconsistent > 0 else 0.0
    pass_criterion = n_correct >= PASS_THRESHOLD

    # Consistent products: count those with strong outliers (should be few)
    n_consistent_with_strong_outlier = sum(
        1 for r in consistent_results if r.gap >= gap_threshold
    )
    n_total_consistent = len(consistent_results)

    # Notes
    notes: list[str] = []

    if pass_criterion:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        notes.append(
            f"Only {n_correct}/{n_total_inconsistent} correct outlier identifications "
            f"(need >= {PASS_THRESHOLD})."
        )

    # Check for missing ground truth
    n_missing_gt = sum(
        1 for r in inconsistent_results if r.ground_truth_outlier is None
    )
    if n_missing_gt > 0:
        notes.append(
            f"{n_missing_gt} inconsistent products missing ground-truth annotation."
        )

    # Consistent product secondary analysis
    if n_total_consistent > 0:
        strong_outlier_rate = n_consistent_with_strong_outlier / n_total_consistent
        if strong_outlier_rate > 0.5:
            notes.append(
                f"WARNING: {n_consistent_with_strong_outlier}/{n_total_consistent} "
                f"consistent products show strong outlier (gap >= {gap_threshold}). "
                f"The metric may have false-positive outlier detection."
            )
        else:
            notes.append(
                f"Secondary check: {n_consistent_with_strong_outlier}/{n_total_consistent} "
                f"consistent products show strong outlier (gap >= {gap_threshold})."
            )

    return Experiment2Results(
        inconsistent_results=inconsistent_results,
        n_correct=n_correct,
        n_total_inconsistent=n_total_inconsistent,
        accuracy=accuracy,
        pass_criterion=pass_criterion,
        consistent_results=consistent_results,
        n_consistent_with_strong_outlier=n_consistent_with_strong_outlier,
        n_total_consistent=n_total_consistent,
        consistent_gap_threshold=gap_threshold,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def results_to_dict(results: Experiment2Results) -> dict[str, Any]:
    """Convert Experiment2Results to a JSON-serializable dict."""
    def attribution_to_dict(attr: ProductAttribution) -> dict[str, Any]:
        return {
            "product_id": attr.product_id,
            "outlier_channel": attr.outlier_channel,
            "mean_similarities": attr.mean_similarities,
            "gap": attr.gap,
            "ground_truth_outlier": attr.ground_truth_outlier,
            "correct": attr.correct,
        }

    return {
        "inconsistent_analysis": {
            "results": [attribution_to_dict(r) for r in results.inconsistent_results],
            "n_correct": results.n_correct,
            "n_total": results.n_total_inconsistent,
            "accuracy": results.accuracy,
            "pass_threshold": PASS_THRESHOLD,
            "pass_criterion": results.pass_criterion,
        },
        "consistent_analysis": {
            "results": [attribution_to_dict(r) for r in results.consistent_results],
            "n_with_strong_outlier": results.n_consistent_with_strong_outlier,
            "n_total": results.n_total_consistent,
            "gap_threshold": results.consistent_gap_threshold,
        },
        "verdict": results.verdict,
        "notes": results.notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Experiment 2 pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Experiment 2: Channel Attribution")
    logger.info("=" * 60)

    # Gate checks
    verdict_data, embeddings_npz, product_labels = check_gates()

    # Reconstruct embeddings
    product_embeddings = reconstruct_embeddings(embeddings_npz)
    logger.info(
        "Reconstructed embeddings for %d products", len(product_embeddings)
    )

    # Load ground-truth annotations
    annotations = load_annotations(ANNOTATIONS_PATH)

    # Analyze inconsistent products
    logger.info("Analyzing inconsistent products...")
    inconsistent_results = analyze_inconsistent_products(
        product_embeddings, product_labels, annotations
    )

    # Analyze consistent products (secondary)
    logger.info("Analyzing consistent products (secondary check)...")
    consistent_results = analyze_consistent_products(
        product_embeddings, product_labels
    )

    # Build results
    results = build_results(inconsistent_results, consistent_results)

    # Write results
    EXP1_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXP1_DIR / "experiment_2_results.json"
    with open(output_path, "w") as f:
        json.dump(results_to_dict(results), f, indent=2, default=_json_default)

    logger.info("Results written to %s", output_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Experiment 2 complete.")
    logger.info("  Verdict: %s", results.verdict)
    logger.info(
        "  Correct outlier identifications: %d/%d (%.1f%%)",
        results.n_correct,
        results.n_total_inconsistent,
        results.accuracy * 100,
    )
    logger.info(
        "  Consistent products with strong outlier: %d/%d",
        results.n_consistent_with_strong_outlier,
        results.n_total_consistent,
    )
    if results.notes:
        for note in results.notes:
            logger.info("  NOTE: %s", note)


if __name__ == "__main__":
    main()
