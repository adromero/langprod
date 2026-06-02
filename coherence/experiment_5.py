"""Experiment 5 -- Competitive Coherence Benchmarking.

Tests whether the coherence metric produces brand rankings that agree with
expert judgment. Selects one CPG category with 5+ competing brands, computes
coherence scores, and compares the metric ranking to expert rankings using
Kendall tau. Expert evaluation forms are generated BEFORE metric results are
computed to prevent anchoring.

Run as::

    python -m coherence.experiment_5
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
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
EXP5_DIR = DATA_DIR / "coherence" / "exp5"
PORTFOLIO_DIR = DATA_DIR / "coherence" / "portfolios" / "exp5"

METRIC_SELECTION_PATH = EXP0_DIR / "metric_selection.json"
GLOBAL_MEAN_PATH = EXP0_DIR / "global_mean.npy"
VERDICT_PATH = EXP1_DIR / "verdict.json"

# Model identifier (matches Experiment 0/1)
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

# Acceptable verdicts from Experiment 1
ACCEPTABLE_VERDICTS = {"PASS", "PASS_NO_VALUE_ADDED"}

# Experiment parameters
MIN_BRANDS = 5
N_EXPERTS = 3
REQUIRED_CHANNELS = {"regulatory", "marketing", "retail", "social", "consumer_review"}

# Pass criteria
MIN_PAIRWISE_AGREEMENT = 7  # out of C(5,2) = 10 pairwise comparisons
KENDALL_TAU_P_THRESHOLD = 0.10  # lenient given small N


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BrandScore:
    """Coherence score for a single brand."""

    brand_id: str
    brand_name: str
    coherence_result: Optional[CoherenceResult] = None
    market_coherence: Optional[float] = None
    brand_coherence: Optional[float] = None
    metric_rank: Optional[int] = None  # 1 = most coherent


@dataclass
class ExpertRanking:
    """A single expert's ranking of brands."""

    expert_id: str
    ranking: list[str]  # brand_ids from most to least coherent
    rank_map: dict[str, int] = field(default_factory=dict)  # brand_id -> rank (1-based)


@dataclass
class PairwiseComparison:
    """Comparison of a brand pair across metric and experts."""

    brand_a: str
    brand_b: str
    metric_winner: str  # brand with higher coherence
    expert_majority_winner: str  # brand preferred by majority of experts
    n_experts_agree_with_metric: int
    agrees_with_majority: bool


@dataclass
class Experiment5Results:
    """Full results of Experiment 5."""

    # Brand scores
    brand_scores: list[BrandScore]
    metric_ranking: list[str]  # brand_ids from most to least coherent

    # Expert rankings
    expert_rankings: list[ExpertRanking]

    # Agreement metrics
    metric_expert_taus: list[float]  # Kendall tau per expert
    mean_metric_expert_tau: float
    inter_expert_taus: list[float]  # pairwise Kendall tau between experts
    mean_inter_expert_tau: float

    # Pairwise comparison
    pairwise_comparisons: list[PairwiseComparison]
    n_pairwise_agree: int
    n_pairwise_total: int

    # Pass criteria
    tau_criterion_pass: bool  # metric-expert >= inter-expert
    pairwise_criterion_pass: bool  # >= 7/10 pairwise agree

    # Overall
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
# Portfolio loading (competitive structure)
# ---------------------------------------------------------------------------


def load_competitive_manifest(portfolio_dir: Path) -> dict:
    """Load the competitive experiment manifest.

    Expected schema::

        {
            "category": "oral_care",
            "brands": [
                {
                    "brand_id": "colgate_total",
                    "brand_name": "Colgate Total"
                },
                ...
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
            "Competitive manifest not found: %s. "
            "Create a manifest.json listing brands in the target category.",
            manifest_path,
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    brands = manifest.get("brands", [])
    if len(brands) < MIN_BRANDS:
        logger.error(
            "Manifest requires >= %d brands, got %d.", MIN_BRANDS, len(brands)
        )
        sys.exit(1)

    logger.info(
        "Manifest loaded: %d brands in category %s",
        len(brands),
        manifest.get("category", "unknown"),
    )

    return manifest


def load_brand_documents(
    portfolio_dir: Path, manifest: dict
) -> dict[str, list[RealDocument]]:
    """Load documents organized by brand.

    Expected directory structure::

        portfolio_dir/
          {brand_id}/
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
    dict[str, list[RealDocument]]
        ``{brand_id: [RealDocument, ...]}``
    """
    brand_docs: dict[str, list[RealDocument]] = {}

    for brand_spec in manifest["brands"]:
        brand_id = brand_spec["brand_id"]
        brand_dir = portfolio_dir / brand_id

        if not brand_dir.is_dir():
            logger.warning("Brand directory not found: %s. Skipping.", brand_dir)
            continue

        docs = load_documents(brand_dir.parent)
        # filter to just this brand
        brand_specific = [d for d in docs if d.product_id == brand_id]

        if not brand_specific:
            # Try loading directly from the brand_dir as a portfolio
            # with a single product subdirectory
            brand_specific = _load_brand_direct(brand_id, brand_dir)

        # Clean and truncate
        cleaned = []
        for doc in brand_specific:
            doc = clean_document(doc)
            doc = truncate_document(doc)
            cleaned.append(doc)

        brand_docs[brand_id] = cleaned

        channels_present = {d.channel for d in cleaned}
        logger.info(
            "Brand %s: %d documents, channels: %s",
            brand_id,
            len(cleaned),
            sorted(channels_present),
        )

        missing = REQUIRED_CHANNELS - channels_present
        if missing:
            logger.warning(
                "Brand %s missing channels: %s", brand_id, sorted(missing)
            )

    return brand_docs


def _load_brand_direct(brand_id: str, brand_dir: Path) -> list[RealDocument]:
    """Load documents directly from a brand directory (flat structure).

    Handles the case where brand_dir contains .txt files directly
    instead of a product subdirectory.
    """
    documents: list[RealDocument] = []

    # Load optional metadata
    metadata: dict = {}
    meta_path = brand_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    source_urls = metadata.get("source_urls", {})
    dates_collected = metadata.get("dates_collected", {})
    authors = metadata.get("authors", {})

    # Top-level .txt files
    for txt_file in sorted(brand_dir.glob("*.txt")):
        channel = txt_file.stem
        if channel not in CHANNELS:
            continue

        text = txt_file.read_text(encoding="utf-8")
        documents.append(
            RealDocument(
                product_id=brand_id,
                channel=channel,
                text=text,
                source_url=source_urls.get(channel, ""),
                date_collected=dates_collected.get(channel, ""),
                author=authors.get(channel, None),
            )
        )

    # Consumer review subdirectory
    review_dir = brand_dir / "consumer_review"
    if review_dir.is_dir():
        for review_file in sorted(review_dir.glob("*.txt")):
            text = review_file.read_text(encoding="utf-8")
            documents.append(
                RealDocument(
                    product_id=brand_id,
                    channel="consumer_review",
                    text=text,
                    source_url=source_urls.get("consumer_review", ""),
                    date_collected=dates_collected.get("consumer_review", ""),
                    author=authors.get("consumer_review", None),
                )
            )

    return documents


# ---------------------------------------------------------------------------
# Expert evaluation form generation
# ---------------------------------------------------------------------------


def generate_expert_form(
    manifest: dict,
    output_dir: Path,
) -> Path:
    """Generate the expert evaluation form BEFORE computing metric results.

    The form asks each expert to rank brands by perceived brand message
    coherence -- how consistently the brand communicates its core identity
    across channels. This MUST be generated and distributed before any metric
    results are known to prevent anchoring.

    Parameters
    ----------
    manifest :
        Competitive manifest with brand information.
    output_dir :
        Directory to write the form to.

    Returns
    -------
    Path
        Path to the generated form JSON.
    """
    brands = manifest["brands"]
    category = manifest.get("category", "unknown")

    form = {
        "experiment": "Experiment 5: Competitive Coherence Benchmarking",
        "category": category,
        "instructions": (
            "You are evaluating the brand message coherence of the following brands "
            f"in the {category} category. Brand coherence measures how consistently "
            "a brand communicates its core identity and key messages across all "
            "communication channels (regulatory filings, marketing materials, retail "
            "listings, social media, and consumer reviews).\n\n"
            "Please rank the brands from MOST coherent (rank 1) to LEAST coherent "
            f"(rank {len(brands)}). This is a forced-choice ranking: no ties allowed.\n\n"
            "Base your ranking on your professional knowledge and assessment of each "
            "brand's messaging consistency, NOT on brand quality or personal preference."
        ),
        "brands": [
            {
                "brand_id": b["brand_id"],
                "brand_name": b["brand_name"],
                "rank": None,  # to be filled by expert
            }
            for b in brands
        ],
        "expert_id": None,  # to be filled
        "date_completed": None,  # to be filled
        "pre_registered": True,
        "notes": (
            "IMPORTANT: This form must be completed BEFORE seeing any metric results. "
            "Your ranking should reflect your independent professional judgment."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    form_path = output_dir / "expert_evaluation_form.json"
    with open(form_path, "w") as f:
        json.dump(form, f, indent=2)

    logger.info("Expert evaluation form generated: %s", form_path)
    logger.info(
        "IMPORTANT: Distribute this form to %d experts BEFORE computing metric results.",
        N_EXPERTS,
    )

    return form_path


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_brand_embeddings(
    brand_docs: dict[str, list[RealDocument]],
    metric_selection: dict,
    global_mean: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract hidden-state embeddings for all brands.

    Parameters
    ----------
    brand_docs :
        ``{brand_id: [RealDocument, ...]}``
    metric_selection :
        Locked metric from Experiment 0.
    global_mean :
        Global mean vector for anisotropy correction.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{brand_id: {channel: embedding_vector}}``
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

    brand_embeddings: dict[str, dict[str, np.ndarray]] = {}

    for brand_id, docs in brand_docs.items():
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

        brand_embeddings[brand_id] = channel_embeddings
        logger.info(
            "Brand %s: %d channels (%s)",
            brand_id,
            len(channel_embeddings),
            ", ".join(sorted(channel_embeddings.keys())),
        )

    # Clean up GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return brand_embeddings


# ---------------------------------------------------------------------------
# Coherence scoring and ranking
# ---------------------------------------------------------------------------


def compute_brand_scores(
    brand_embeddings: dict[str, dict[str, np.ndarray]],
    manifest: dict,
    method: str,
) -> list[BrandScore]:
    """Compute coherence scores for all brands and rank them.

    Parameters
    ----------
    brand_embeddings :
        ``{brand_id: {channel: embedding_vector}}``
    manifest :
        Competitive manifest with brand info.
    method :
        Aggregation method from locked metric.

    Returns
    -------
    list[BrandScore]
        Sorted by market coherence (highest first = rank 1).
    """
    brand_name_map = {
        b["brand_id"]: b["brand_name"] for b in manifest["brands"]
    }

    scores: list[BrandScore] = []
    for brand_id, channel_embs in brand_embeddings.items():
        brand_name = brand_name_map.get(brand_id, brand_id)

        if len(channel_embs) < 2:
            logger.warning(
                "Brand %s has only %d channel(s); cannot compute coherence.",
                brand_id,
                len(channel_embs),
            )
            scores.append(BrandScore(brand_id=brand_id, brand_name=brand_name))
            continue

        result = compute_coherence_score(channel_embs, method=method)
        scores.append(
            BrandScore(
                brand_id=brand_id,
                brand_name=brand_name,
                coherence_result=result,
                market_coherence=result.market_coherence,
                brand_coherence=result.brand_coherence,
            )
        )

    # Sort by market coherence (descending) and assign ranks
    scores.sort(
        key=lambda s: s.market_coherence if s.market_coherence is not None else float("-inf"),
        reverse=True,
    )
    for rank, score in enumerate(scores, 1):
        score.metric_rank = rank

    for s in scores:
        logger.info(
            "Brand %s (%s): market=%.4f, brand=%s, rank=%d",
            s.brand_id,
            s.brand_name,
            s.market_coherence if s.market_coherence is not None else float("nan"),
            f"{s.brand_coherence:.4f}" if s.brand_coherence is not None else "N/A",
            s.metric_rank if s.metric_rank is not None else -1,
        )

    return scores


# ---------------------------------------------------------------------------
# Expert ranking loading
# ---------------------------------------------------------------------------


def load_expert_rankings(exp5_dir: Path) -> list[ExpertRanking]:
    """Load completed expert rankings from JSON files.

    Expected files::

        exp5_dir/
          expert_ranking_1.json
          expert_ranking_2.json
          expert_ranking_3.json

    Each file schema::

        {
            "expert_id": "expert_1",
            "ranking": ["brand_a", "brand_b", "brand_c", "brand_d", "brand_e"]
        }

    Returns
    -------
    list[ExpertRanking]

    Raises
    ------
    SystemExit
        If fewer than N_EXPERTS ranking files are found.
    """
    rankings: list[ExpertRanking] = []

    for i in range(1, N_EXPERTS + 1):
        ranking_path = exp5_dir / f"expert_ranking_{i}.json"
        if not ranking_path.exists():
            logger.error(
                "Expert ranking file not found: %s. "
                "Collect all %d expert rankings before running analysis.",
                ranking_path,
                N_EXPERTS,
            )
            sys.exit(1)

        with open(ranking_path) as f:
            data = json.load(f)

        expert_id = data.get("expert_id", f"expert_{i}")
        ranking = data["ranking"]

        rank_map = {brand_id: rank + 1 for rank, brand_id in enumerate(ranking)}

        rankings.append(
            ExpertRanking(
                expert_id=expert_id,
                ranking=ranking,
                rank_map=rank_map,
            )
        )
        logger.info(
            "Loaded expert %s ranking: %s",
            expert_id,
            " > ".join(ranking),
        )

    return rankings


# ---------------------------------------------------------------------------
# Agreement computation
# ---------------------------------------------------------------------------


def compute_metric_expert_taus(
    metric_ranking: list[str],
    expert_rankings: list[ExpertRanking],
) -> list[float]:
    """Compute Kendall tau between metric ranking and each expert ranking.

    Parameters
    ----------
    metric_ranking :
        Brand IDs sorted by metric (most coherent first).
    expert_rankings :
        Expert rankings.

    Returns
    -------
    list[float]
        Kendall tau values for each expert.
    """
    metric_ranks = {brand_id: rank + 1 for rank, brand_id in enumerate(metric_ranking)}
    taus = []

    for expert in expert_rankings:
        # Align brand IDs
        common_brands = sorted(set(metric_ranks.keys()) & set(expert.rank_map.keys()))
        if len(common_brands) < 2:
            logger.warning(
                "Expert %s has fewer than 2 common brands with metric; tau=0",
                expert.expert_id,
            )
            taus.append(0.0)
            continue

        metric_order = [metric_ranks[b] for b in common_brands]
        expert_order = [expert.rank_map[b] for b in common_brands]

        tau, _ = stats.kendalltau(metric_order, expert_order)
        taus.append(float(tau))

        logger.info(
            "Metric vs %s: Kendall tau = %.4f",
            expert.expert_id,
            tau,
        )

    return taus


def compute_inter_expert_taus(
    expert_rankings: list[ExpertRanking],
) -> list[float]:
    """Compute pairwise Kendall tau between all expert pairs.

    Returns
    -------
    list[float]
        Kendall tau for each expert pair.
    """
    taus = []

    for (i, exp_a), (j, exp_b) in combinations(enumerate(expert_rankings), 2):
        common_brands = sorted(
            set(exp_a.rank_map.keys()) & set(exp_b.rank_map.keys())
        )
        if len(common_brands) < 2:
            logger.warning(
                "Experts %s and %s have fewer than 2 common brands; tau=0",
                exp_a.expert_id,
                exp_b.expert_id,
            )
            taus.append(0.0)
            continue

        ranks_a = [exp_a.rank_map[b] for b in common_brands]
        ranks_b = [exp_b.rank_map[b] for b in common_brands]

        tau, _ = stats.kendalltau(ranks_a, ranks_b)
        taus.append(float(tau))

        logger.info(
            "%s vs %s: Kendall tau = %.4f",
            exp_a.expert_id,
            exp_b.expert_id,
            tau,
        )

    return taus


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def compute_pairwise_comparisons(
    metric_ranking: list[str],
    expert_rankings: list[ExpertRanking],
) -> list[PairwiseComparison]:
    """Compare every brand pair: does the metric agree with expert majority?

    Parameters
    ----------
    metric_ranking :
        Brand IDs sorted by metric (most coherent first).
    expert_rankings :
        Expert rankings.

    Returns
    -------
    list[PairwiseComparison]
    """
    metric_ranks = {brand_id: rank + 1 for rank, brand_id in enumerate(metric_ranking)}
    comparisons: list[PairwiseComparison] = []

    for brand_a, brand_b in combinations(metric_ranking, 2):
        # Metric winner (lower rank = more coherent = winner)
        metric_winner = brand_a if metric_ranks[brand_a] < metric_ranks[brand_b] else brand_b

        # Expert majority vote
        expert_votes_a = 0
        expert_votes_b = 0
        for expert in expert_rankings:
            rank_a = expert.rank_map.get(brand_a, len(metric_ranking) + 1)
            rank_b = expert.rank_map.get(brand_b, len(metric_ranking) + 1)
            if rank_a < rank_b:
                expert_votes_a += 1
            else:
                expert_votes_b += 1

        expert_majority_winner = brand_a if expert_votes_a > expert_votes_b else brand_b
        agrees = metric_winner == expert_majority_winner

        n_experts_agree = (
            expert_votes_a if metric_winner == brand_a else expert_votes_b
        )

        comparisons.append(
            PairwiseComparison(
                brand_a=brand_a,
                brand_b=brand_b,
                metric_winner=metric_winner,
                expert_majority_winner=expert_majority_winner,
                n_experts_agree_with_metric=n_experts_agree,
                agrees_with_majority=agrees,
            )
        )

    return comparisons


# ---------------------------------------------------------------------------
# Results building
# ---------------------------------------------------------------------------


def build_results(
    brand_scores: list[BrandScore],
    metric_ranking: list[str],
    expert_rankings: list[ExpertRanking],
    metric_expert_taus: list[float],
    inter_expert_taus: list[float],
    pairwise_comparisons: list[PairwiseComparison],
) -> Experiment5Results:
    """Build the final results object.

    Parameters
    ----------
    brand_scores :
        Scored and ranked brands.
    metric_ranking :
        Brand IDs sorted by metric.
    expert_rankings :
        Expert rankings.
    metric_expert_taus :
        Kendall tau per expert.
    inter_expert_taus :
        Pairwise inter-expert Kendall tau.
    pairwise_comparisons :
        Pairwise brand comparisons.

    Returns
    -------
    Experiment5Results
    """
    mean_metric_expert_tau = float(np.mean(metric_expert_taus))
    mean_inter_expert_tau = float(np.mean(inter_expert_taus)) if inter_expert_taus else 0.0

    n_pairwise_agree = sum(1 for c in pairwise_comparisons if c.agrees_with_majority)
    n_pairwise_total = len(pairwise_comparisons)

    # Criterion 1: metric-expert agreement >= inter-expert agreement
    tau_criterion_pass = mean_metric_expert_tau >= mean_inter_expert_tau

    # Criterion 2: >= 7/10 pairwise comparisons agree with expert majority
    pairwise_criterion_pass = n_pairwise_agree >= MIN_PAIRWISE_AGREEMENT

    # Overall verdict
    notes: list[str] = []

    if tau_criterion_pass and pairwise_criterion_pass:
        verdict = "PASS"
    elif tau_criterion_pass or pairwise_criterion_pass:
        verdict = "PARTIAL_PASS"
        if not tau_criterion_pass:
            notes.append(
                f"Tau criterion failed: mean metric-expert tau "
                f"({mean_metric_expert_tau:.4f}) < inter-expert tau "
                f"({mean_inter_expert_tau:.4f})"
            )
        if not pairwise_criterion_pass:
            notes.append(
                f"Pairwise criterion failed: {n_pairwise_agree}/{n_pairwise_total} "
                f"(need >= {MIN_PAIRWISE_AGREEMENT})"
            )
    else:
        verdict = "FAIL"
        notes.append(
            f"Both criteria failed: tau ({mean_metric_expert_tau:.4f} vs "
            f"{mean_inter_expert_tau:.4f}), pairwise ({n_pairwise_agree}/{n_pairwise_total})"
        )

    notes.append(
        f"Mean metric-expert tau: {mean_metric_expert_tau:.4f}, "
        f"mean inter-expert tau: {mean_inter_expert_tau:.4f}"
    )
    notes.append(
        f"Pairwise agreement: {n_pairwise_agree}/{n_pairwise_total}"
    )

    return Experiment5Results(
        brand_scores=brand_scores,
        metric_ranking=metric_ranking,
        expert_rankings=expert_rankings,
        metric_expert_taus=metric_expert_taus,
        mean_metric_expert_tau=mean_metric_expert_tau,
        inter_expert_taus=inter_expert_taus,
        mean_inter_expert_tau=mean_inter_expert_tau,
        pairwise_comparisons=pairwise_comparisons,
        n_pairwise_agree=n_pairwise_agree,
        n_pairwise_total=n_pairwise_total,
        tau_criterion_pass=tau_criterion_pass,
        pairwise_criterion_pass=pairwise_criterion_pass,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def results_to_dict(results: Experiment5Results) -> dict:
    """Convert results to a JSON-serializable dict."""
    return {
        "brand_scores": [
            {
                "brand_id": s.brand_id,
                "brand_name": s.brand_name,
                "market_coherence": s.market_coherence,
                "brand_coherence": s.brand_coherence,
                "metric_rank": s.metric_rank,
            }
            for s in results.brand_scores
        ],
        "metric_ranking": results.metric_ranking,
        "expert_rankings": [
            {
                "expert_id": e.expert_id,
                "ranking": e.ranking,
            }
            for e in results.expert_rankings
        ],
        "agreement": {
            "metric_expert_taus": results.metric_expert_taus,
            "mean_metric_expert_tau": results.mean_metric_expert_tau,
            "inter_expert_taus": results.inter_expert_taus,
            "mean_inter_expert_tau": results.mean_inter_expert_tau,
        },
        "pairwise_comparisons": [
            {
                "brand_a": c.brand_a,
                "brand_b": c.brand_b,
                "metric_winner": c.metric_winner,
                "expert_majority_winner": c.expert_majority_winner,
                "n_experts_agree_with_metric": c.n_experts_agree_with_metric,
                "agrees_with_majority": c.agrees_with_majority,
            }
            for c in results.pairwise_comparisons
        ],
        "summary": {
            "n_pairwise_agree": results.n_pairwise_agree,
            "n_pairwise_total": results.n_pairwise_total,
            "tau_criterion_pass": results.tau_criterion_pass,
            "pairwise_criterion_pass": results.pairwise_criterion_pass,
        },
        "verdict": {
            "overall": results.verdict,
            "criteria": {
                "tau_criterion": {
                    "pass": results.tau_criterion_pass,
                    "mean_metric_expert_tau": results.mean_metric_expert_tau,
                    "mean_inter_expert_tau": results.mean_inter_expert_tau,
                },
                "pairwise_criterion": {
                    "pass": results.pairwise_criterion_pass,
                    "n_agree": results.n_pairwise_agree,
                    "n_total": results.n_pairwise_total,
                    "threshold": MIN_PAIRWISE_AGREEMENT,
                },
            },
            "notes": results.notes,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 5: Competitive Coherence Benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("Experiment 5: Competitive Coherence Benchmarking")
    logger.info("=" * 60)

    # 1. Gate checks
    metric_selection, global_mean, verdict_data = check_gates()
    method = metric_selection["aggregation"]

    # 2. Load competitive manifest
    manifest = load_competitive_manifest(PORTFOLIO_DIR)

    # 3. Generate expert evaluation form BEFORE computing metric results
    logger.info("Generating expert evaluation form (pre-registration)...")
    form_path = generate_expert_form(manifest, EXP5_DIR)
    logger.info(
        "Expert form written to %s. Distribute to experts BEFORE proceeding.", form_path
    )

    # 4. Load brand documents
    brand_docs = load_brand_documents(PORTFOLIO_DIR, manifest)

    # 5. Extract embeddings
    logger.info("Extracting embeddings for all brands...")
    brand_embeddings = extract_brand_embeddings(
        brand_docs, metric_selection, global_mean
    )

    # 6. Compute coherence scores and rank brands
    logger.info("Computing brand coherence scores...")
    brand_scores = compute_brand_scores(brand_embeddings, manifest, method)
    metric_ranking = [s.brand_id for s in brand_scores]

    # 7. Save metric results before loading expert rankings
    EXP5_DIR.mkdir(parents=True, exist_ok=True)
    metric_results_path = EXP5_DIR / "metric_scores.json"
    with open(metric_results_path, "w") as f:
        json.dump(
            {
                "metric_ranking": metric_ranking,
                "scores": [
                    {
                        "brand_id": s.brand_id,
                        "brand_name": s.brand_name,
                        "market_coherence": s.market_coherence,
                        "brand_coherence": s.brand_coherence,
                        "metric_rank": s.metric_rank,
                    }
                    for s in brand_scores
                ],
            },
            f,
            indent=2,
            default=_json_default,
        )
    logger.info("Metric scores saved to %s", metric_results_path)

    # Save embeddings for reproducibility
    flat_embeddings: dict[str, np.ndarray] = {}
    for brand_id, channel_embs in brand_embeddings.items():
        for channel, vec in channel_embs.items():
            flat_embeddings[f"{brand_id}/{channel}"] = vec
    embeddings_path = EXP5_DIR / "embeddings.npz"
    np.savez_compressed(embeddings_path, **flat_embeddings)
    logger.info("Embeddings saved to %s (%d vectors)", embeddings_path, len(flat_embeddings))

    # 8. Load expert rankings (must have been collected after form generation)
    logger.info("Loading expert rankings...")
    expert_rankings = load_expert_rankings(EXP5_DIR)

    # 9. Compute agreement metrics
    logger.info("Computing metric-expert agreement...")
    metric_expert_taus = compute_metric_expert_taus(metric_ranking, expert_rankings)

    logger.info("Computing inter-expert agreement...")
    inter_expert_taus = compute_inter_expert_taus(expert_rankings)

    # 10. Pairwise comparisons
    logger.info("Computing pairwise comparisons...")
    pairwise_comparisons = compute_pairwise_comparisons(
        metric_ranking, expert_rankings
    )

    # 11. Build results
    results = build_results(
        brand_scores=brand_scores,
        metric_ranking=metric_ranking,
        expert_rankings=expert_rankings,
        metric_expert_taus=metric_expert_taus,
        inter_expert_taus=inter_expert_taus,
        pairwise_comparisons=pairwise_comparisons,
    )

    # 12. Persist results
    results_path = EXP5_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_to_dict(results), f, indent=2, default=_json_default)
    logger.info("Results written to %s", results_path)

    verdict_path = EXP5_DIR / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results_to_dict(results)["verdict"], f, indent=2, default=_json_default)
    logger.info("Verdict written to %s", verdict_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Experiment 5 complete.")
    logger.info("  Verdict: %s", results.verdict)
    logger.info("  Metric ranking: %s", " > ".join(results.metric_ranking))
    logger.info(
        "  Mean metric-expert tau: %.4f", results.mean_metric_expert_tau
    )
    logger.info(
        "  Mean inter-expert tau: %.4f", results.mean_inter_expert_tau
    )
    logger.info(
        "  Tau criterion: %s (metric-expert >= inter-expert: %.4f >= %.4f)",
        "PASS" if results.tau_criterion_pass else "FAIL",
        results.mean_metric_expert_tau,
        results.mean_inter_expert_tau,
    )
    logger.info(
        "  Pairwise criterion: %s (%d/%d >= %d)",
        "PASS" if results.pairwise_criterion_pass else "FAIL",
        results.n_pairwise_agree,
        results.n_pairwise_total,
        MIN_PAIRWISE_AGREEMENT,
    )
    for expert in expert_rankings:
        tau_idx = expert_rankings.index(expert)
        logger.info(
            "  Expert %s tau: %.4f, ranking: %s",
            expert.expert_id,
            metric_expert_taus[tau_idx],
            " > ".join(expert.ranking),
        )
    if results.notes:
        for note in results.notes:
            logger.info("  NOTE: %s", note)


if __name__ == "__main__":
    main()
