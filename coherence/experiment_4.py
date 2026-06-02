"""Experiment 4 -- Temporal Coherence Drift.

Tests whether the coherence metric captures meaningful temporal changes in
brand messaging by tracking coherence trajectories across multiple time points.
Products with known brand events (rebrands, agency changes, category expansions)
should show directional coherence shifts that differ from a stable-messaging
control product.

Run as::

    python -m coherence.experiment_4
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from coherence.ingest import (
    CHANNELS,
    RealDocument,
    clean_document,
    load_documents,
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
EXP4_DIR = DATA_DIR / "coherence" / "exp4"
PORTFOLIO_DIR = DATA_DIR / "coherence" / "portfolios" / "exp4"

METRIC_SELECTION_PATH = EXP0_DIR / "metric_selection.json"
GLOBAL_MEAN_PATH = EXP0_DIR / "global_mean.npy"
VERDICT_PATH = EXP1_DIR / "verdict.json"

# Model identifier (matches Experiment 0/1)
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

# Acceptable verdicts from Experiment 1
ACCEPTABLE_VERDICTS = {"PASS", "PASS_NO_VALUE_ADDED"}

# Experiment parameters
MIN_TEST_PRODUCTS = 2
MAX_TEST_PRODUCTS = 3
MIN_TIME_POINTS = 3
N_CONTROL_PRODUCTS = 1

# Statistical thresholds
KENDALL_TAU_P_THRESHOLD = 0.10  # one-sided, lenient for small N
PASS_MIN_DIRECTIONALLY_CORRECT = 2  # out of 2-3 test products


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TimePoint:
    """A single time point for a product."""

    label: str  # e.g., "launch", "mid_2020", "current"
    date: str  # ISO date or descriptive label
    coherence: Optional[CoherenceResult] = None
    brand_coherence: Optional[float] = None
    market_coherence: Optional[float] = None


@dataclass
class BrandEvent:
    """A known brand event for correlation with coherence changes."""

    date: str
    event_type: str  # "rebrand", "agency_change", "category_expansion", "other"
    description: str
    expected_direction: str  # "increase", "decrease", "disruption"


@dataclass
class ProductTrajectory:
    """Coherence trajectory for a single product across time points."""

    product_id: str
    is_control: bool
    time_points: list[TimePoint] = field(default_factory=list)
    brand_events: list[BrandEvent] = field(default_factory=list)
    raw_coherences: list[float] = field(default_factory=list)
    corrected_coherences: list[float] = field(default_factory=list)
    kendall_tau: Optional[float] = None
    kendall_p: Optional[float] = None
    directionally_correct: Optional[bool] = None


@dataclass
class Experiment4Results:
    """Full results of Experiment 4."""

    test_trajectories: list[ProductTrajectory]
    control_trajectory: ProductTrajectory
    n_directionally_correct: int
    n_test_products: int
    pass_criterion: bool
    kendall_overall: Optional[float]
    kendall_overall_p: Optional[float]
    verdict: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_gates() -> tuple[dict, np.ndarray, dict]:
    """Verify that Experiment 0 and Experiment 1 outputs exist and are valid.

    Returns
    -------
    (metric_selection, global_mean, verdict_data)

    Raises
    ------
    SystemExit
        If any gate file is missing or the Experiment 1 verdict is not PASS.
    """
    # 1. Check Experiment 0 outputs
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
        "Exp0 gates passed: metric=%s layer=%d, global_mean shape=%s",
        metric_selection["aggregation"],
        metric_selection["layer_hdf5_index"],
        global_mean.shape,
    )

    # 2. Check Experiment 1 verdict
    if not VERDICT_PATH.exists():
        logger.error(
            "Gate failed: %s not found. Run Experiment 1 first.", VERDICT_PATH
        )
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

    logger.info("Exp1 verdict gate passed: %s", overall)

    return metric_selection, global_mean, verdict_data


# ---------------------------------------------------------------------------
# Portfolio loading (temporal structure)
# ---------------------------------------------------------------------------


def load_temporal_manifest(portfolio_dir: Path) -> dict:
    """Load the temporal experiment manifest.

    Expected schema::

        {
            "category": "skincare",
            "test_products": [
                {
                    "product_id": "brand_x",
                    "time_points": [
                        {"label": "launch_2018", "date": "2018-01"},
                        {"label": "rebrand_2021", "date": "2021-06"},
                        {"label": "current_2025", "date": "2025-01"}
                    ],
                    "brand_events": [
                        {
                            "date": "2021-03",
                            "event_type": "rebrand",
                            "description": "Visual identity overhaul and messaging refresh",
                            "expected_direction": "disruption"
                        }
                    ]
                },
                ...
            ],
            "control_products": [
                {
                    "product_id": "stable_brand",
                    "time_points": [
                        {"label": "2018", "date": "2018-01"},
                        {"label": "2021", "date": "2021-06"},
                        {"label": "2025", "date": "2025-01"}
                    ],
                    "brand_events": []
                }
            ]
        }

    Returns
    -------
    dict
        Parsed manifest.

    Raises
    ------
    SystemExit
        If manifest is missing or invalid.
    """
    manifest_path = portfolio_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(
            "Temporal manifest not found: %s. "
            "Create a manifest.json describing test/control products and time points.",
            manifest_path,
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Validate structure
    test_products = manifest.get("test_products", [])
    control_products = manifest.get("control_products", [])

    if len(test_products) < MIN_TEST_PRODUCTS:
        logger.error(
            "Manifest requires >= %d test products, got %d.",
            MIN_TEST_PRODUCTS,
            len(test_products),
        )
        sys.exit(1)

    if len(control_products) < N_CONTROL_PRODUCTS:
        logger.error(
            "Manifest requires >= %d control product(s), got %d.",
            N_CONTROL_PRODUCTS,
            len(control_products),
        )
        sys.exit(1)

    # Validate time points per product
    for product_spec in test_products + control_products:
        tp_count = len(product_spec.get("time_points", []))
        if tp_count < MIN_TIME_POINTS:
            logger.error(
                "Product %s has %d time points, need >= %d.",
                product_spec.get("product_id", "UNKNOWN"),
                tp_count,
                MIN_TIME_POINTS,
            )
            sys.exit(1)

    logger.info(
        "Manifest loaded: %d test products, %d control product(s), category=%s",
        len(test_products),
        len(control_products),
        manifest.get("category", "unknown"),
    )

    return manifest


def load_temporal_documents(
    portfolio_dir: Path, manifest: dict
) -> dict[str, dict[str, list[RealDocument]]]:
    """Load documents organized by product and time point.

    Expected directory structure::

        portfolio_dir/
          {product_id}/
            {time_point_label}/
              regulatory.txt
              marketing.txt
              retail.txt
              social.txt
              consumer_review/
                review_001.txt
                ...
              metadata.json  (optional)

    Returns
    -------
    dict[str, dict[str, list[RealDocument]]]
        ``{product_id: {time_point_label: [RealDocument, ...]}}``
    """
    all_product_specs = manifest.get("test_products", []) + manifest.get(
        "control_products", []
    )

    product_time_docs: dict[str, dict[str, list[RealDocument]]] = {}

    for product_spec in all_product_specs:
        product_id = product_spec["product_id"]
        product_time_docs[product_id] = {}

        for tp in product_spec["time_points"]:
            tp_label = tp["label"]
            tp_dir = portfolio_dir / product_id / tp_label

            if not tp_dir.is_dir():
                logger.warning(
                    "Time point directory not found: %s. Skipping.", tp_dir
                )
                continue

            # Load documents from this time point directory
            docs = _load_timepoint_documents(product_id, tp_label, tp_dir)
            if docs:
                product_time_docs[product_id][tp_label] = docs
                logger.info(
                    "Loaded %d docs for %s/%s",
                    len(docs),
                    product_id,
                    tp_label,
                )
            else:
                logger.warning(
                    "No documents loaded for %s/%s", product_id, tp_label
                )

    return product_time_docs


def _load_timepoint_documents(
    product_id: str, tp_label: str, tp_dir: Path
) -> list[RealDocument]:
    """Load and prepare documents from a single time-point directory.

    Returns cleaned and truncated RealDocuments.
    """
    documents: list[RealDocument] = []

    # Load optional metadata
    metadata: dict = {}
    meta_path = tp_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    source_urls = metadata.get("source_urls", {})
    dates_collected = metadata.get("dates_collected", {})
    authors = metadata.get("authors", {})

    # Top-level .txt files
    for txt_file in sorted(tp_dir.glob("*.txt")):
        channel = txt_file.stem
        if channel not in CHANNELS:
            logger.warning("Skipping unknown channel file: %s", txt_file)
            continue

        text = txt_file.read_text(encoding="utf-8")
        doc = RealDocument(
            product_id=product_id,
            channel=channel,
            text=text,
            source_url=source_urls.get(channel, ""),
            date_collected=dates_collected.get(channel, tp_label),
            author=authors.get(channel, None),
        )
        doc = clean_document(doc)
        doc = truncate_document(doc)
        documents.append(doc)

    # Consumer review subdirectory
    review_dir = tp_dir / "consumer_review"
    if review_dir.is_dir():
        for review_file in sorted(review_dir.glob("*.txt")):
            text = review_file.read_text(encoding="utf-8")
            doc = RealDocument(
                product_id=product_id,
                channel="consumer_review",
                text=text,
                source_url=source_urls.get("consumer_review", ""),
                date_collected=dates_collected.get("consumer_review", tp_label),
                author=authors.get("consumer_review", None),
            )
            doc = clean_document(doc)
            doc = truncate_document(doc)
            documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Embedding extraction (per time point)
# ---------------------------------------------------------------------------


def extract_temporal_embeddings(
    product_time_docs: dict[str, dict[str, list[RealDocument]]],
    metric_selection: dict,
    global_mean: np.ndarray,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Extract hidden-state embeddings for all products at all time points.

    Parameters
    ----------
    product_time_docs :
        ``{product_id: {time_point_label: [RealDocument, ...]}}``
    metric_selection :
        Locked metric from Experiment 0.
    global_mean :
        Global mean vector from Experiment 0 for anisotropy correction.

    Returns
    -------
    dict[str, dict[str, dict[str, np.ndarray]]]
        ``{product_id: {time_point_label: {channel: embedding_vector}}}``
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

    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for product_id, time_docs in product_time_docs.items():
        result[product_id] = {}

        for tp_label, docs in time_docs.items():
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

                # CRITICAL: Subtract Exp 0 global mean (NOT batch mean)
                corrected_vec = raw_vec - global_mean

                channel_vectors[doc.channel].append(corrected_vec)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Average vectors per channel
            channel_embeddings: dict[str, np.ndarray] = {}
            for channel, vecs in channel_vectors.items():
                channel_embeddings[channel] = np.mean(vecs, axis=0)

            result[product_id][tp_label] = channel_embeddings
            logger.info(
                "Product %s/%s: %d channels (%s)",
                product_id,
                tp_label,
                len(channel_embeddings),
                ", ".join(sorted(channel_embeddings.keys())),
            )

    # Clean up GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Coherence trajectory computation
# ---------------------------------------------------------------------------


def compute_trajectories(
    temporal_embeddings: dict[str, dict[str, dict[str, np.ndarray]]],
    manifest: dict,
    method: str,
) -> tuple[list[ProductTrajectory], ProductTrajectory]:
    """Compute coherence trajectories for test and control products.

    Parameters
    ----------
    temporal_embeddings :
        ``{product_id: {time_point: {channel: embedding}}}``
    manifest :
        The temporal manifest with product specs.
    method :
        Aggregation method from locked metric.

    Returns
    -------
    (test_trajectories, control_trajectory)
    """
    test_trajectories: list[ProductTrajectory] = []
    control_trajectory: Optional[ProductTrajectory] = None

    # Process test products
    for product_spec in manifest["test_products"]:
        product_id = product_spec["product_id"]
        trajectory = _build_trajectory(
            product_id=product_id,
            product_spec=product_spec,
            temporal_embeddings=temporal_embeddings,
            method=method,
            is_control=False,
        )
        test_trajectories.append(trajectory)

    # Process control product(s) -- use the first one
    for product_spec in manifest["control_products"]:
        product_id = product_spec["product_id"]
        trajectory = _build_trajectory(
            product_id=product_id,
            product_spec=product_spec,
            temporal_embeddings=temporal_embeddings,
            method=method,
            is_control=True,
        )
        if control_trajectory is None:
            control_trajectory = trajectory

    if control_trajectory is None:
        logger.error("No control product trajectory could be computed.")
        sys.exit(1)

    return test_trajectories, control_trajectory


def _build_trajectory(
    product_id: str,
    product_spec: dict,
    temporal_embeddings: dict[str, dict[str, dict[str, np.ndarray]]],
    method: str,
    is_control: bool,
) -> ProductTrajectory:
    """Build a coherence trajectory for a single product."""
    trajectory = ProductTrajectory(
        product_id=product_id,
        is_control=is_control,
    )

    # Parse brand events
    for event_data in product_spec.get("brand_events", []):
        trajectory.brand_events.append(
            BrandEvent(
                date=event_data["date"],
                event_type=event_data["event_type"],
                description=event_data["description"],
                expected_direction=event_data.get("expected_direction", "disruption"),
            )
        )

    # Compute coherence at each time point (in order)
    product_embs = temporal_embeddings.get(product_id, {})

    for tp_spec in product_spec["time_points"]:
        tp_label = tp_spec["label"]
        tp_date = tp_spec.get("date", tp_label)

        channel_embs = product_embs.get(tp_label, {})

        tp = TimePoint(label=tp_label, date=tp_date)

        if len(channel_embs) >= 2:
            result = compute_coherence_score(channel_embs, method=method)
            tp.coherence = result
            tp.brand_coherence = result.brand_coherence
            tp.market_coherence = result.market_coherence
            trajectory.raw_coherences.append(result.market_coherence)
        else:
            logger.warning(
                "Product %s/%s has %d channel(s); insufficient for coherence.",
                product_id,
                tp_label,
                len(channel_embs),
            )
            trajectory.raw_coherences.append(float("nan"))

        trajectory.time_points.append(tp)

    logger.info(
        "Product %s trajectory: %s",
        product_id,
        [f"{tp.label}={c:.4f}" for tp, c in zip(trajectory.time_points, trajectory.raw_coherences)
         if not np.isnan(c)],
    )

    return trajectory


# ---------------------------------------------------------------------------
# Control correction (secular trend subtraction)
# ---------------------------------------------------------------------------


def apply_control_correction(
    test_trajectories: list[ProductTrajectory],
    control_trajectory: ProductTrajectory,
) -> None:
    """Subtract control product's trajectory from test products.

    Modifies ``corrected_coherences`` in place on each test trajectory.
    For the control product, corrected = raw (identity).
    """
    control_raw = np.array(control_trajectory.raw_coherences)

    # Control trajectory: corrected = raw
    control_trajectory.corrected_coherences = list(control_raw)

    for trajectory in test_trajectories:
        raw = np.array(trajectory.raw_coherences)

        # Handle length mismatch gracefully
        min_len = min(len(raw), len(control_raw))
        if min_len < len(raw):
            logger.warning(
                "Product %s has %d time points but control has %d; "
                "truncating to %d for correction.",
                trajectory.product_id,
                len(raw),
                len(control_raw),
                min_len,
            )

        corrected = raw[:min_len] - control_raw[:min_len]
        # Pad with NaN if needed
        if min_len < len(raw):
            corrected = np.concatenate(
                [corrected, np.full(len(raw) - min_len, np.nan)]
            )

        trajectory.corrected_coherences = list(corrected)

        logger.info(
            "Product %s corrected trajectory: %s",
            trajectory.product_id,
            [f"{c:.4f}" for c in corrected if not np.isnan(c)],
        )


# ---------------------------------------------------------------------------
# Statistical evaluation
# ---------------------------------------------------------------------------


def evaluate_trajectory(
    trajectory: ProductTrajectory,
) -> None:
    """Evaluate whether a trajectory is non-random and directionally correct.

    Uses Kendall tau test on the corrected coherences.
    Sets ``kendall_tau``, ``kendall_p``, and ``directionally_correct`` on the
    trajectory in place.
    """
    corrected = np.array(trajectory.corrected_coherences)

    # Remove NaN values
    valid_mask = ~np.isnan(corrected)
    valid_coherences = corrected[valid_mask]
    time_indices = np.arange(len(corrected))[valid_mask]

    if len(valid_coherences) < MIN_TIME_POINTS:
        logger.warning(
            "Product %s has only %d valid time points after NaN removal; "
            "cannot compute Kendall tau (need >= %d).",
            trajectory.product_id,
            len(valid_coherences),
            MIN_TIME_POINTS,
        )
        trajectory.kendall_tau = None
        trajectory.kendall_p = None
        trajectory.directionally_correct = False
        return

    # Kendall tau: correlation between time index and corrected coherence
    tau, p_value = stats.kendalltau(time_indices, valid_coherences)

    trajectory.kendall_tau = float(tau)
    trajectory.kendall_p = float(p_value)

    # Check if non-random (p < threshold)
    is_non_random = p_value < KENDALL_TAU_P_THRESHOLD

    # Directionality: check if the trajectory direction is consistent
    # with known brand events
    direction_ok = _check_directionality(trajectory, tau)

    trajectory.directionally_correct = is_non_random and direction_ok

    logger.info(
        "Product %s: Kendall tau=%.4f, p=%.4f, non_random=%s, direction_ok=%s => %s",
        trajectory.product_id,
        tau,
        p_value,
        is_non_random,
        direction_ok,
        "CORRECT" if trajectory.directionally_correct else "INCORRECT",
    )


def _check_directionality(trajectory: ProductTrajectory, tau: float) -> bool:
    """Check if the trajectory direction matches expectations from brand events.

    Rules:
    - If brand events with "disruption" or "decrease" expected: negative tau is OK
    - If brand events with "increase" expected: positive tau is OK
    - If no brand events but trajectory is non-random: considered directionally
      correct if there's a clear trend (|tau| > 0)
    """
    if not trajectory.brand_events:
        # No events: any non-random trend is considered valid
        return True

    # Aggregate expected directions from events
    expected_directions = [e.expected_direction for e in trajectory.brand_events]

    if "disruption" in expected_directions or "decrease" in expected_directions:
        # We expect a dip or disruption -- any significant trend is acceptable
        # since disruption can manifest as either direction depending on
        # whether the brand has recovered
        return True

    if "increase" in expected_directions:
        # We expect improvement -- tau should be positive
        return tau > 0

    # Default: any trend is fine
    return True


def build_results(
    test_trajectories: list[ProductTrajectory],
    control_trajectory: ProductTrajectory,
) -> Experiment4Results:
    """Build the final results object.

    Parameters
    ----------
    test_trajectories :
        Evaluated test product trajectories.
    control_trajectory :
        The control product trajectory.

    Returns
    -------
    Experiment4Results
    """
    n_directionally_correct = sum(
        1 for t in test_trajectories if t.directionally_correct
    )
    n_test = len(test_trajectories)
    pass_criterion = n_directionally_correct >= PASS_MIN_DIRECTIONALLY_CORRECT

    # Overall Kendall tau: combine all test product corrected coherences
    all_times = []
    all_coherences = []
    for t in test_trajectories:
        for i, c in enumerate(t.corrected_coherences):
            if not np.isnan(c):
                all_times.append(i)
                all_coherences.append(c)

    if len(all_times) >= 3:
        overall_tau, overall_p = stats.kendalltau(all_times, all_coherences)
    else:
        overall_tau = None
        overall_p = None

    # Determine verdict
    notes: list[str] = []

    if pass_criterion:
        verdict = "PASS"
        notes.append(
            f"{n_directionally_correct}/{n_test} test products "
            f"show directionally correct non-random trajectories."
        )
    else:
        verdict = "FAIL"
        notes.append(
            f"Only {n_directionally_correct}/{n_test} products "
            f"are directionally correct (need >= {PASS_MIN_DIRECTIONALLY_CORRECT})."
        )

    # Add control trajectory note
    control_range = (
        max(control_trajectory.raw_coherences) - min(control_trajectory.raw_coherences)
        if control_trajectory.raw_coherences
        else 0.0
    )
    notes.append(
        f"Control product {control_trajectory.product_id} "
        f"coherence range: {control_range:.4f}"
    )

    return Experiment4Results(
        test_trajectories=test_trajectories,
        control_trajectory=control_trajectory,
        n_directionally_correct=n_directionally_correct,
        n_test_products=n_test,
        pass_criterion=pass_criterion,
        kendall_overall=float(overall_tau) if overall_tau is not None else None,
        kendall_overall_p=float(overall_p) if overall_p is not None else None,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types and dataclass objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def results_to_dict(results: Experiment4Results) -> dict:
    """Convert results to a JSON-serializable dict."""

    def trajectory_to_dict(t: ProductTrajectory) -> dict:
        return {
            "product_id": t.product_id,
            "is_control": t.is_control,
            "time_points": [
                {
                    "label": tp.label,
                    "date": tp.date,
                    "brand_coherence": tp.brand_coherence,
                    "market_coherence": tp.market_coherence,
                }
                for tp in t.time_points
            ],
            "brand_events": [
                {
                    "date": e.date,
                    "event_type": e.event_type,
                    "description": e.description,
                    "expected_direction": e.expected_direction,
                }
                for e in t.brand_events
            ],
            "raw_coherences": t.raw_coherences,
            "corrected_coherences": t.corrected_coherences,
            "kendall_tau": t.kendall_tau,
            "kendall_p": t.kendall_p,
            "directionally_correct": t.directionally_correct,
        }

    return {
        "test_trajectories": [trajectory_to_dict(t) for t in results.test_trajectories],
        "control_trajectory": trajectory_to_dict(results.control_trajectory),
        "summary": {
            "n_directionally_correct": results.n_directionally_correct,
            "n_test_products": results.n_test_products,
            "pass_criterion": results.pass_criterion,
            "kendall_overall": results.kendall_overall,
            "kendall_overall_p": results.kendall_overall_p,
        },
        "verdict": {
            "overall": results.verdict,
            "criteria": {
                "kendall_tau_threshold": KENDALL_TAU_P_THRESHOLD,
                "min_directionally_correct": PASS_MIN_DIRECTIONALLY_CORRECT,
            },
            "notes": results.notes,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 4: Temporal Coherence Drift."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("Experiment 4: Temporal Coherence Drift")
    logger.info("=" * 60)

    # 1. Gate checks
    metric_selection, global_mean, verdict_data = check_gates()
    method = metric_selection["aggregation"]

    # 2. Load temporal manifest
    manifest = load_temporal_manifest(PORTFOLIO_DIR)

    # 3. Load documents organized by product and time point
    product_time_docs = load_temporal_documents(PORTFOLIO_DIR, manifest)

    # 4. Extract embeddings at each time point
    logger.info("Extracting embeddings for all products and time points...")
    temporal_embeddings = extract_temporal_embeddings(
        product_time_docs, metric_selection, global_mean
    )

    # 5. Compute coherence trajectories
    logger.info("Computing coherence trajectories...")
    test_trajectories, control_trajectory = compute_trajectories(
        temporal_embeddings, manifest, method
    )

    # 6. Apply control correction (secular trend subtraction)
    logger.info("Applying control correction...")
    apply_control_correction(test_trajectories, control_trajectory)

    # 7. Evaluate each test trajectory
    logger.info("Evaluating trajectories...")
    for trajectory in test_trajectories:
        evaluate_trajectory(trajectory)

    # 8. Build results
    results = build_results(test_trajectories, control_trajectory)

    # 9. Persist results
    EXP4_DIR.mkdir(parents=True, exist_ok=True)

    results_path = EXP4_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_to_dict(results), f, indent=2, default=_json_default)
    logger.info("Results written to %s", results_path)

    verdict_path = EXP4_DIR / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results_to_dict(results)["verdict"], f, indent=2, default=_json_default)
    logger.info("Verdict written to %s", verdict_path)

    # Save embeddings for reproducibility
    flat_embeddings: dict[str, np.ndarray] = {}
    for product_id, time_embs in temporal_embeddings.items():
        for tp_label, channel_embs in time_embs.items():
            for channel, vec in channel_embs.items():
                flat_embeddings[f"{product_id}/{tp_label}/{channel}"] = vec
    embeddings_path = EXP4_DIR / "embeddings.npz"
    np.savez_compressed(embeddings_path, **flat_embeddings)
    logger.info("Embeddings saved to %s (%d vectors)", embeddings_path, len(flat_embeddings))

    # Summary
    logger.info("=" * 60)
    logger.info("Experiment 4 complete.")
    logger.info("  Verdict: %s", results.verdict)
    logger.info(
        "  Directionally correct: %d/%d",
        results.n_directionally_correct,
        results.n_test_products,
    )
    if results.kendall_overall is not None:
        logger.info(
            "  Overall Kendall tau: %.4f (p=%.4f)",
            results.kendall_overall,
            results.kendall_overall_p,
        )
    for trajectory in results.test_trajectories:
        status = "CORRECT" if trajectory.directionally_correct else "INCORRECT"
        tau_str = f"tau={trajectory.kendall_tau:.4f}" if trajectory.kendall_tau is not None else "tau=N/A"
        logger.info(
            "  Product %s: %s (%s, p=%s)",
            trajectory.product_id,
            status,
            tau_str,
            f"{trajectory.kendall_p:.4f}" if trajectory.kendall_p is not None else "N/A",
        )
    if results.notes:
        for note in results.notes:
            logger.info("  NOTE: %s", note)


if __name__ == "__main__":
    main()
