"""Experiment 3 -- Attribute-Level Drill-Down.

Tests whether the attribute-coherence metric can identify which product
attributes are present or absent in which channels, using sentence-level
and paragraph-level probes.  This is the most commercially valuable test:
if the metric can resolve *attribute x channel* gaps, it can power
automated content audits.

Run as::

    python -m coherence.experiment_3
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from coherence.metrics import AttributeCoherenceResult, compute_attribute_coherence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
EXP1_DIR = DATA_DIR / "coherence" / "exp1"
EXP3_DIR = DATA_DIR / "coherence" / "exp3"

VERDICT_PATH = EXP1_DIR / "verdict.json"
EMBEDDINGS_PATH = EXP1_DIR / "embeddings.npz"
PRODUCT_LABELS_PATH = EXP1_DIR / "product_labels.json"

GLOBAL_MEAN_PATH = DATA_DIR / "coherence" / "exp0" / "global_mean.npy"
METRIC_SELECTION_PATH = DATA_DIR / "coherence" / "exp0" / "metric_selection.json"

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

# Accuracy thresholds
ACCURACY_THRESHOLD = 0.70          # 70 % attribute-channel pairs correct
MIN_PRODUCTS_PASSING = 2           # out of 3
PROBE_LEVEL_DELTA_WARN = 0.15      # 15 pp delta triggers a warning

# Similarity threshold for classifying present/absent.
# Cells above this are predicted "present"; below are "absent".
SIMILARITY_CUTOFF = 0.5


# ---------------------------------------------------------------------------
# Attribute annotation schema
# ---------------------------------------------------------------------------


def _default_attribute_annotations() -> List[Dict[str, Any]]:
    """Return the per-product attribute annotation for the 3 test products.

    Schema per entry::

        {
            "product_id": str,
            "category": str,               # product category (for wrong-product control)
            "wrong_product_id": str,        # different product in same category
            "attributes": [
                {
                    "name": str,
                    "sentence_probe": str,  # single sentence mentioning product + attribute
                    "paragraph_probe": str, # 3-4 sentences with product context
                    "synonyms": [str, ...], # alternative phrasings
                    "ground_truth": {       # channel -> bool (True = present)
                        "marketing": True,
                        "regulatory": False,
                        ...
                    }
                },
                ...
            ]
        }

    The 3 products are chosen to have *known* attribute-level messaging gaps:
    attributes that appear in some channels but not others.
    """
    return [
        {
            "product_id": "product_A",
            "category": "pharmaceuticals",
            "wrong_product_id": "product_B",
            "attributes": [
                {
                    "name": "efficacy_data",
                    "sentence_probe": "Product A demonstrates strong clinical efficacy in randomized trials.",
                    "paragraph_probe": (
                        "Product A has been evaluated in multiple randomized controlled trials. "
                        "The clinical efficacy data show significant improvement over placebo. "
                        "These results were consistent across patient subgroups. "
                        "The efficacy profile supports its use as a first-line therapy."
                    ),
                    "synonyms": [
                        "Product A shows proven effectiveness in clinical studies.",
                        "Product A delivers strong clinical results in controlled research.",
                    ],
                    "ground_truth": {
                        "regulatory": True,
                        "marketing": True,
                        "retail": False,
                        "social": False,
                    },
                },
                {
                    "name": "safety_profile",
                    "sentence_probe": "Product A has a well-characterized safety profile with manageable side effects.",
                    "paragraph_probe": (
                        "The safety profile of Product A has been extensively studied. "
                        "Adverse events are generally mild and transient. "
                        "Long-term safety data confirm a favorable risk-benefit ratio. "
                        "Healthcare providers can prescribe Product A with confidence in its tolerability."
                    ),
                    "synonyms": [
                        "Product A has a favorable and well-known safety record.",
                        "Product A's side effects are minimal and well-documented.",
                    ],
                    "ground_truth": {
                        "regulatory": True,
                        "marketing": True,
                        "retail": True,
                        "social": False,
                    },
                },
                {
                    "name": "lifestyle_benefit",
                    "sentence_probe": "Product A helps patients maintain an active and fulfilling lifestyle.",
                    "paragraph_probe": (
                        "Patients using Product A report improvements in daily quality of life. "
                        "The product allows them to stay active and engaged in their routines. "
                        "Many users highlight the lifestyle benefits as a key differentiator. "
                        "Product A empowers patients to live life on their own terms."
                    ),
                    "synonyms": [
                        "Product A supports a better quality of daily living.",
                        "Product A enables patients to enjoy their everyday activities.",
                    ],
                    "ground_truth": {
                        "regulatory": False,
                        "marketing": True,
                        "retail": True,
                        "social": True,
                    },
                },
                {
                    "name": "cost_effectiveness",
                    "sentence_probe": "Product A offers cost-effective treatment compared to alternatives.",
                    "paragraph_probe": (
                        "Economic analyses demonstrate that Product A is cost-effective. "
                        "When compared to existing therapies, the total cost of care is lower. "
                        "Payers and providers recognize the value proposition. "
                        "Product A delivers clinical value at a competitive price point."
                    ),
                    "synonyms": [
                        "Product A provides affordable therapy relative to competitors.",
                        "Product A delivers good value for the treatment cost.",
                    ],
                    "ground_truth": {
                        "regulatory": False,
                        "marketing": True,
                        "retail": True,
                        "social": False,
                    },
                },
            ],
        },
        {
            "product_id": "product_C",
            "category": "medical_devices",
            "wrong_product_id": "product_D",
            "attributes": [
                {
                    "name": "ease_of_use",
                    "sentence_probe": "Product C is designed for intuitive and easy use by patients.",
                    "paragraph_probe": (
                        "Product C features an intuitive user interface designed for patients. "
                        "Minimal training is required to operate the device effectively. "
                        "User studies confirm high satisfaction with the ease of use. "
                        "Even first-time users can operate Product C without difficulty."
                    ),
                    "synonyms": [
                        "Product C is simple and user-friendly for patients.",
                        "Product C requires minimal training to operate effectively.",
                    ],
                    "ground_truth": {
                        "regulatory": False,
                        "marketing": True,
                        "retail": True,
                        "social": True,
                    },
                },
                {
                    "name": "clinical_accuracy",
                    "sentence_probe": "Product C provides highly accurate clinical measurements.",
                    "paragraph_probe": (
                        "Validation studies show that Product C delivers precise clinical measurements. "
                        "The accuracy exceeds regulatory requirements for the device class. "
                        "Comparative studies against gold-standard methods confirm reliability. "
                        "Clinicians trust Product C for accurate diagnostic information."
                    ),
                    "synonyms": [
                        "Product C delivers precise and reliable clinical readings.",
                        "Product C meets stringent accuracy standards for measurements.",
                    ],
                    "ground_truth": {
                        "regulatory": True,
                        "marketing": True,
                        "retail": False,
                        "social": False,
                    },
                },
                {
                    "name": "connectivity",
                    "sentence_probe": "Product C connects seamlessly to smartphones and health platforms.",
                    "paragraph_probe": (
                        "Product C integrates with popular smartphone apps and health platforms. "
                        "Wireless connectivity allows automatic data synchronization. "
                        "Patients can share readings with their healthcare team in real time. "
                        "The connected experience makes Product C a modern health companion."
                    ),
                    "synonyms": [
                        "Product C syncs wirelessly with phones and health apps.",
                        "Product C offers digital integration with mobile health platforms.",
                    ],
                    "ground_truth": {
                        "regulatory": False,
                        "marketing": True,
                        "retail": True,
                        "social": True,
                    },
                },
            ],
        },
        {
            "product_id": "product_E",
            "category": "otc_supplements",
            "wrong_product_id": "product_F",
            "attributes": [
                {
                    "name": "natural_ingredients",
                    "sentence_probe": "Product E is formulated with all-natural, plant-based ingredients.",
                    "paragraph_probe": (
                        "Product E uses carefully sourced natural and plant-based ingredients. "
                        "The formulation avoids artificial additives and preservatives. "
                        "Each ingredient is selected for its evidence-based health benefits. "
                        "Product E appeals to consumers seeking clean, natural supplementation."
                    ),
                    "synonyms": [
                        "Product E contains only natural, plant-derived components.",
                        "Product E is made with clean, plant-based ingredients.",
                    ],
                    "ground_truth": {
                        "regulatory": True,
                        "marketing": True,
                        "retail": True,
                        "social": True,
                    },
                },
                {
                    "name": "clinical_backing",
                    "sentence_probe": "Product E is supported by clinical studies demonstrating its benefits.",
                    "paragraph_probe": (
                        "Clinical research supports the health benefits of Product E. "
                        "Peer-reviewed studies show meaningful improvements in key biomarkers. "
                        "The clinical evidence differentiates Product E from unproven supplements. "
                        "Healthcare professionals can recommend Product E based on solid science."
                    ),
                    "synonyms": [
                        "Product E has clinical evidence backing its health claims.",
                        "Product E's benefits are validated by scientific research.",
                    ],
                    "ground_truth": {
                        "regulatory": True,
                        "marketing": True,
                        "retail": False,
                        "social": False,
                    },
                },
                {
                    "name": "taste_experience",
                    "sentence_probe": "Product E has a pleasant taste that makes daily supplementation enjoyable.",
                    "paragraph_probe": (
                        "Product E is available in delicious flavors that consumers love. "
                        "The taste experience makes it easy to incorporate into daily routines. "
                        "Unlike many supplements, Product E is something people look forward to taking. "
                        "Flavor innovation sets Product E apart in the supplement category."
                    ),
                    "synonyms": [
                        "Product E tastes great and is enjoyable to take daily.",
                        "Product E comes in flavors that make supplementation a pleasure.",
                    ],
                    "ground_truth": {
                        "regulatory": False,
                        "marketing": True,
                        "retail": True,
                        "social": True,
                    },
                },
                {
                    "name": "third_party_testing",
                    "sentence_probe": "Product E undergoes rigorous third-party testing for purity and potency.",
                    "paragraph_probe": (
                        "Every batch of Product E is independently tested by accredited laboratories. "
                        "Third-party testing verifies purity, potency, and absence of contaminants. "
                        "Certificates of analysis are available for transparency. "
                        "Product E meets the highest quality standards through independent verification."
                    ),
                    "synonyms": [
                        "Product E is independently tested for quality and purity.",
                        "Product E passes rigorous third-party quality verification.",
                    ],
                    "ground_truth": {
                        "regulatory": True,
                        "marketing": False,
                        "retail": False,
                        "social": False,
                    },
                },
                {
                    "name": "influencer_endorsement",
                    "sentence_probe": "Product E is recommended by popular health and wellness influencers.",
                    "paragraph_probe": (
                        "Leading health and wellness influencers recommend Product E. "
                        "Social media endorsements have driven strong consumer awareness. "
                        "Influencer partnerships highlight the product's appeal to health-conscious consumers. "
                        "Product E has built a passionate community of advocates online."
                    ),
                    "synonyms": [
                        "Product E is endorsed by well-known wellness influencers.",
                        "Product E has strong influencer support in the wellness space.",
                    ],
                    "ground_truth": {
                        "regulatory": False,
                        "marketing": False,
                        "retail": False,
                        "social": True,
                    },
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_gates() -> dict:
    """Verify that Experiment 1 produced a passing verdict.

    Returns
    -------
    dict
        The full verdict JSON from Experiment 1.

    Raises
    ------
    SystemExit
        If the verdict file is missing or does not show PASS.
    """
    if not VERDICT_PATH.exists():
        logger.error(
            "Gate failed: %s not found. Run Experiment 1 first.", VERDICT_PATH
        )
        sys.exit(1)

    with open(VERDICT_PATH) as f:
        verdict_data = json.load(f)

    overall = verdict_data.get("verdict", {}).get("overall", "UNKNOWN")

    if overall not in ("PASS", "PASS_NO_VALUE_ADDED"):
        logger.error(
            "Gate failed: Experiment 1 verdict is %s (must be PASS or PASS_NO_VALUE_ADDED). "
            "Fix upstream issues before running Experiment 3.",
            overall,
        )
        sys.exit(1)

    logger.info("Gate passed: Experiment 1 verdict=%s", overall)
    return verdict_data


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def load_exp1_embeddings() -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-product, per-channel embeddings from Experiment 1.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{product_id: {channel: embedding_vector}}``

    Raises
    ------
    SystemExit
        If the embeddings file is missing.
    """
    if not EMBEDDINGS_PATH.exists():
        logger.error(
            "Embeddings not found: %s. Run Experiment 1 first.", EMBEDDINGS_PATH
        )
        sys.exit(1)

    data = np.load(EMBEDDINGS_PATH)
    product_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        product_id, channel = key.split("/", 1)
        if product_id not in product_embeddings:
            product_embeddings[product_id] = {}
        product_embeddings[product_id][channel] = data[key]

    logger.info(
        "Loaded embeddings for %d products from %s",
        len(product_embeddings),
        EMBEDDINGS_PATH,
    )
    return product_embeddings


def embed_probes(
    texts: List[str],
    global_mean: np.ndarray,
    metric_selection: dict,
) -> np.ndarray:
    """Embed probe texts through Qwen and apply anisotropy correction.

    Parameters
    ----------
    texts :
        List of probe strings to embed.
    global_mean :
        Global mean vector from Experiment 0 for anisotropy correction.
    metric_selection :
        Locked metric configuration from Experiment 0.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(texts), D)`` with corrected embeddings.
    """
    import torch
    from extraction import load_model_and_tokenizer, mean_pool_no_special

    hdf5_layer_idx = metric_selection["layer_hdf5_index"]

    logger.info("Loading model %s for probe embedding...", MODEL_ID)
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)

    special_ids: set[int] = set()
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    device = next(model.parameters()).device

    embeddings = []
    for text in texts:
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

        # Apply Exp 0 global mean correction (same as Exp 1)
        corrected_vec = raw_vec - global_mean
        embeddings.append(corrected_vec)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clean up GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# Ground truth comparison
# ---------------------------------------------------------------------------


def evaluate_accuracy(
    result: AttributeCoherenceResult,
    attributes: List[Dict[str, Any]],
    cutoff: float = SIMILARITY_CUTOFF,
) -> tuple[float, int, int]:
    """Compare an attribute-coherence matrix against ground truth.

    Parameters
    ----------
    result :
        The ``AttributeCoherenceResult`` from ``compute_attribute_coherence()``.
    attributes :
        The attribute annotation list (with ``ground_truth`` dicts).
    cutoff :
        Similarity threshold above which a cell is classified as "present".

    Returns
    -------
    (accuracy, n_correct, n_total)
        Fraction correct, count correct, count total.
    """
    n_correct = 0
    n_total = 0
    for i, attr in enumerate(attributes):
        gt = attr["ground_truth"]
        for j, ch_name in enumerate(result.channel_names):
            if ch_name not in gt:
                continue
            predicted_present = result.matrix[i, j] >= cutoff
            actual_present = gt[ch_name]
            if predicted_present == actual_present:
                n_correct += 1
            n_total += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return accuracy, n_correct, n_total


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_experiment(
    annotations: List[Dict[str, Any]],
    product_embeddings: Dict[str, Dict[str, np.ndarray]],
    embed_fn: Any = None,
    global_mean: Optional[np.ndarray] = None,
    metric_selection: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run the Experiment 3 analysis.

    Parameters
    ----------
    annotations :
        Per-product attribute annotations (from ``_default_attribute_annotations()``).
    product_embeddings :
        Channel embeddings per product from Experiment 1.
    embed_fn :
        Callable ``(texts, global_mean, metric_selection) -> np.ndarray``.
        Defaults to ``embed_probes`` when ``None``.
    global_mean :
        Global mean vector (required if embed_fn is None).
    metric_selection :
        Locked metric config (required if embed_fn is None).

    Returns
    -------
    dict
        Full results including per-product details, verdict, and warnings.
    """
    if embed_fn is None:
        embed_fn = embed_probes

    product_results: List[Dict[str, Any]] = []

    for annotation in annotations:
        product_id = annotation["product_id"]
        wrong_product_id = annotation["wrong_product_id"]
        attributes = annotation["attributes"]
        attribute_names = [a["name"] for a in attributes]

        logger.info(
            "Processing product %s (%d attributes, wrong-product=%s)",
            product_id,
            len(attributes),
            wrong_product_id,
        )

        # --- Build probe texts ---
        sentence_probes = [a["sentence_probe"] for a in attributes]
        paragraph_probes = [a["paragraph_probe"] for a in attributes]

        # Also build synonym probes (for synonym concern reporting)
        synonym_texts: List[str] = []
        for a in attributes:
            synonym_texts.extend(a.get("synonyms", []))

        # --- Embed probes ---
        all_texts = sentence_probes + paragraph_probes + synonym_texts
        all_embeddings = embed_fn(all_texts, global_mean, metric_selection)

        n_attrs = len(attributes)
        n_sentence = n_attrs
        n_paragraph = n_attrs

        sentence_embs = all_embeddings[:n_sentence]
        paragraph_embs = all_embeddings[n_sentence : n_sentence + n_paragraph]
        synonym_embs = all_embeddings[n_sentence + n_paragraph :]

        # --- Get channel embeddings for target product ---
        if product_id not in product_embeddings:
            logger.warning(
                "Product %s not found in Experiment 1 embeddings; skipping.",
                product_id,
            )
            continue

        target_ch_embs = product_embeddings[product_id]

        # --- Compute attribute-coherence at both probe levels ---
        result_sentence = compute_attribute_coherence(
            sentence_embs, attribute_names, target_ch_embs
        )
        result_paragraph = compute_attribute_coherence(
            paragraph_embs, attribute_names, target_ch_embs
        )

        # --- Evaluate accuracy at each level ---
        acc_sentence, correct_s, total_s = evaluate_accuracy(
            result_sentence, attributes
        )
        acc_paragraph, correct_p, total_p = evaluate_accuracy(
            result_paragraph, attributes
        )

        logger.info(
            "  %s sentence accuracy: %.1f%% (%d/%d)",
            product_id,
            acc_sentence * 100,
            correct_s,
            total_s,
        )
        logger.info(
            "  %s paragraph accuracy: %.1f%% (%d/%d)",
            product_id,
            acc_paragraph * 100,
            correct_p,
            total_p,
        )

        # --- Wrong-product control ---
        wrong_control = {"sentence": {}, "paragraph": {}}

        if wrong_product_id in product_embeddings:
            wrong_ch_embs = product_embeddings[wrong_product_id]

            wrong_result_sentence = compute_attribute_coherence(
                sentence_embs, attribute_names, wrong_ch_embs
            )
            wrong_result_paragraph = compute_attribute_coherence(
                paragraph_embs, attribute_names, wrong_ch_embs
            )

            target_mean_s = float(np.mean(result_sentence.matrix))
            wrong_mean_s = float(np.mean(wrong_result_sentence.matrix))
            target_mean_p = float(np.mean(result_paragraph.matrix))
            wrong_mean_p = float(np.mean(wrong_result_paragraph.matrix))

            wrong_control["sentence"] = {
                "target_mean": target_mean_s,
                "wrong_mean": wrong_mean_s,
                "pass": wrong_mean_s < target_mean_s,
            }
            wrong_control["paragraph"] = {
                "target_mean": target_mean_p,
                "wrong_mean": wrong_mean_p,
                "pass": wrong_mean_p < target_mean_p,
            }

            logger.info(
                "  %s wrong-product control (sentence): target=%.4f, wrong=%.4f, pass=%s",
                product_id,
                target_mean_s,
                wrong_mean_s,
                wrong_control["sentence"]["pass"],
            )
            logger.info(
                "  %s wrong-product control (paragraph): target=%.4f, wrong=%.4f, pass=%s",
                product_id,
                target_mean_p,
                wrong_mean_p,
                wrong_control["paragraph"]["pass"],
            )
        else:
            logger.warning(
                "Wrong product %s not found in embeddings; wrong-product control skipped.",
                wrong_product_id,
            )
            wrong_control["sentence"] = {"target_mean": None, "wrong_mean": None, "pass": False}
            wrong_control["paragraph"] = {"target_mean": None, "wrong_mean": None, "pass": False}

        # --- Synonym concern: check that synonym probes produce similar results ---
        synonym_report: Optional[Dict[str, Any]] = None
        if len(synonym_embs) > 0:
            # Build per-attribute synonym cosines against the target product channels
            synonym_idx = 0
            synonym_diffs: List[float] = []
            for i, attr in enumerate(attributes):
                n_syns = len(attr.get("synonyms", []))
                if n_syns == 0:
                    continue
                attr_syn_embs = synonym_embs[synonym_idx : synonym_idx + n_syns]
                synonym_idx += n_syns

                # Compare mean synonym-channel similarity to sentence-probe similarity
                for j, ch_name in enumerate(result_sentence.channel_names):
                    sent_sim = result_sentence.matrix[i, j]
                    syn_sims = [
                        float(
                            np.dot(attr_syn_embs[k], target_ch_embs[ch_name])
                            / (
                                np.linalg.norm(attr_syn_embs[k])
                                * np.linalg.norm(target_ch_embs[ch_name])
                                + 1e-12
                            )
                        )
                        for k in range(n_syns)
                    ]
                    mean_syn_sim = float(np.mean(syn_sims))
                    synonym_diffs.append(abs(sent_sim - mean_syn_sim))

            synonym_report = {
                "mean_abs_diff": float(np.mean(synonym_diffs)) if synonym_diffs else 0.0,
                "max_abs_diff": float(np.max(synonym_diffs)) if synonym_diffs else 0.0,
                "n_comparisons": len(synonym_diffs),
            }
            logger.info(
                "  %s synonym concern: mean_diff=%.4f, max_diff=%.4f (%d comparisons)",
                product_id,
                synonym_report["mean_abs_diff"],
                synonym_report["max_abs_diff"],
                synonym_report["n_comparisons"],
            )

        # --- Probe-level agreement ---
        delta = abs(acc_sentence - acc_paragraph)
        probe_level_warning = delta > PROBE_LEVEL_DELTA_WARN

        if probe_level_warning:
            logger.warning(
                "  %s probe-level delta=%.1f pp (>%.0f pp threshold). "
                "Suggests probe-length sensitivity.",
                product_id,
                delta * 100,
                PROBE_LEVEL_DELTA_WARN * 100,
            )

        product_results.append(
            {
                "product_id": product_id,
                "category": annotation["category"],
                "wrong_product_id": wrong_product_id,
                "n_attributes": len(attributes),
                "sentence": {
                    "accuracy": acc_sentence,
                    "n_correct": correct_s,
                    "n_total": total_s,
                    "matrix": result_sentence.matrix.tolist(),
                    "channel_names": result_sentence.channel_names,
                },
                "paragraph": {
                    "accuracy": acc_paragraph,
                    "n_correct": correct_p,
                    "n_total": total_p,
                    "matrix": result_paragraph.matrix.tolist(),
                    "channel_names": result_paragraph.channel_names,
                },
                "wrong_product_control": wrong_control,
                "synonym_report": synonym_report,
                "probe_level_delta": delta,
                "probe_level_warning": probe_level_warning,
            }
        )

    # ---------------------------------------------------------------------------
    # Build verdict
    # ---------------------------------------------------------------------------
    verdict = build_verdict(product_results)

    return {
        "experiment": "experiment_3_attribute_drill_down",
        "n_products": len(product_results),
        "products": product_results,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def build_verdict(product_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the pass/fail verdict for Experiment 3.

    Criteria
    --------
    1. **Accuracy gate**: >= 2/3 products correctly characterized (>= 70 %
       attribute-channel pairs correct) at BOTH probe levels independently.
       A product is "correctly characterized" only if it passes at both
       sentence and paragraph level.

    2. **Wrong-product control gate**: For all 3 products, the wrong-product
       mean must be strictly lower than the target-product mean at BOTH
       probe levels.

    3. **Probe-level agreement**: Warn (but do not gate) if any product's
       accuracy differs by more than 15 pp between the two levels.
    """
    # Criterion 1: Accuracy at both levels
    products_passing_accuracy = 0
    per_product_accuracy: List[Dict[str, Any]] = []

    for pr in product_results:
        pass_sentence = pr["sentence"]["accuracy"] >= ACCURACY_THRESHOLD
        pass_paragraph = pr["paragraph"]["accuracy"] >= ACCURACY_THRESHOLD
        passes_both = pass_sentence and pass_paragraph

        if passes_both:
            products_passing_accuracy += 1

        per_product_accuracy.append(
            {
                "product_id": pr["product_id"],
                "sentence_accuracy": pr["sentence"]["accuracy"],
                "paragraph_accuracy": pr["paragraph"]["accuracy"],
                "pass_sentence": pass_sentence,
                "pass_paragraph": pass_paragraph,
                "passes_both": passes_both,
            }
        )

    accuracy_gate_pass = products_passing_accuracy >= MIN_PRODUCTS_PASSING

    # Criterion 2: Wrong-product control at both levels
    wrong_product_failures: List[str] = []
    for pr in product_results:
        wpc = pr["wrong_product_control"]
        if not wpc["sentence"].get("pass", False):
            wrong_product_failures.append(
                f"{pr['product_id']} sentence-level"
            )
        if not wpc["paragraph"].get("pass", False):
            wrong_product_failures.append(
                f"{pr['product_id']} paragraph-level"
            )

    wrong_product_gate_pass = len(wrong_product_failures) == 0

    # Criterion 3: Probe-level agreement warnings (not gated)
    probe_level_warnings: List[str] = []
    for pr in product_results:
        if pr["probe_level_warning"]:
            probe_level_warnings.append(
                f"{pr['product_id']}: delta={pr['probe_level_delta'] * 100:.1f}pp"
            )

    # Overall verdict
    overall = "PASS" if (accuracy_gate_pass and wrong_product_gate_pass) else "FAIL"

    notes: List[str] = []
    if not accuracy_gate_pass:
        notes.append(
            f"Accuracy gate FAILED: {products_passing_accuracy}/{len(product_results)} "
            f"products pass at both levels (need {MIN_PRODUCTS_PASSING})."
        )
    if not wrong_product_gate_pass:
        notes.append(
            f"Wrong-product control FAILED for: {', '.join(wrong_product_failures)}."
        )
    if probe_level_warnings:
        notes.append(
            f"Probe-level agreement WARNING: {'; '.join(probe_level_warnings)}."
        )

    return {
        "overall": overall,
        "criteria": {
            "accuracy_gate": {
                "pass": accuracy_gate_pass,
                "products_passing": products_passing_accuracy,
                "min_required": MIN_PRODUCTS_PASSING,
                "per_product": per_product_accuracy,
            },
            "wrong_product_control": {
                "pass": wrong_product_gate_pass,
                "failures": wrong_product_failures,
            },
            "probe_level_agreement": {
                "warnings": probe_level_warnings,
            },
        },
        "notes": notes,
    }


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
    """Run the full Experiment 3 pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Experiment 3: Attribute-Level Drill-Down")
    logger.info("=" * 60)

    # Gate: Experiment 1 must pass
    check_gates()

    # Load Experiment 0 artifacts for probe embedding
    if not GLOBAL_MEAN_PATH.exists():
        logger.error(
            "Global mean not found: %s. Run Experiment 0 first.", GLOBAL_MEAN_PATH
        )
        sys.exit(1)
    if not METRIC_SELECTION_PATH.exists():
        logger.error(
            "Metric selection not found: %s. Run Experiment 0 first.",
            METRIC_SELECTION_PATH,
        )
        sys.exit(1)

    global_mean = np.load(GLOBAL_MEAN_PATH)
    with open(METRIC_SELECTION_PATH) as f:
        metric_selection = json.load(f)

    logger.info(
        "Loaded Exp 0 artifacts: metric=%s layer=%d, global_mean shape=%s",
        metric_selection["aggregation"],
        metric_selection["layer_hdf5_index"],
        global_mean.shape,
    )

    # Load Experiment 1 embeddings
    product_embeddings = load_exp1_embeddings()

    # Get attribute annotations
    annotations = _default_attribute_annotations()

    # Run experiment
    results = run_experiment(
        annotations=annotations,
        product_embeddings=product_embeddings,
        global_mean=global_mean,
        metric_selection=metric_selection,
    )

    # Persist results
    EXP3_DIR.mkdir(parents=True, exist_ok=True)

    results_path = EXP3_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Results written to %s", results_path)

    verdict_path = EXP3_DIR / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(results["verdict"], f, indent=2, default=_json_default)
    logger.info("Verdict written to %s", verdict_path)

    # Save annotations for reproducibility
    annotations_path = EXP3_DIR / "attribute_annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)
    logger.info("Annotations saved to %s", annotations_path)

    # Summary
    verdict = results["verdict"]
    logger.info("=" * 60)
    logger.info("Experiment 3 complete.")
    logger.info("  Overall verdict: %s", verdict["overall"])
    logger.info(
        "  Accuracy gate: %s (%d/%d products pass at both levels)",
        verdict["criteria"]["accuracy_gate"]["pass"],
        verdict["criteria"]["accuracy_gate"]["products_passing"],
        len(results["products"]),
    )
    logger.info(
        "  Wrong-product control: %s",
        verdict["criteria"]["wrong_product_control"]["pass"],
    )
    if verdict["criteria"]["probe_level_agreement"]["warnings"]:
        for w in verdict["criteria"]["probe_level_agreement"]["warnings"]:
            logger.warning("  Probe-level: %s", w)
    if verdict["notes"]:
        for note in verdict["notes"]:
            logger.info("  NOTE: %s", note)


if __name__ == "__main__":
    main()
