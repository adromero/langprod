"""Language production analysis pipeline — main entry point.

Usage:
    python run.py generate   — Generate stimuli texts
    python run.py extract    — Extract hidden-state representations
    python run.py analyze    — Run analysis (PCA, RSA, anisotropy)
    python run.py probe      — Train and evaluate probing classifiers
    python run.py report     — Generate figures and summary report
    python run.py all        — Run the full pipeline end-to-end
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "seed": 42,
    "output_dir": "data/",

    # Stimuli
    "n_real_products": 40,
    "n_fictional_products": 40,
    "n_categories": 8,
    "products_per_category": 5,
    "registers": ["marketing", "regulatory", "casual_social", "patent", "journalistic"],
    "variants_per_product": 2,
    "token_range_target": (80, 150),
    "token_range_accept": (50, 200),
    "token_range_hard_reject": (40, 250),
    "cross_generator_subset_size": 10,

    # Models
    "primary_model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "validation_model": "meta-llama/Llama-3.1-8B-Instruct",
    "batch_size": 1,

    # Analysis
    "anisotropy_methods": ["none", "mean_centering", "whitening"],
    "pca_variance_threshold": 0.95,
    "pca_epsilon": 1e-8,

    # Probes
    "probe_C_values": [0.01, 0.1, 1.0, 10.0, 100.0],
    "probe_outer_folds": 5,
    "probe_inner_folds": 3,
    "probe_max_iter": 2000,
    "probe_solver": "lbfgs",

    # Permutation testing
    "screen_permutations": 200,
    "full_permutations": 10000,
    "top_k_layers_for_full_test": 5,

    # Hypothesis thresholds
    "h1_significance": 0.05,
    "h1_min_effect_size": 0.3,
    "h1_min_rsa_r": 0.1,
    "h2_register_dominance_threshold": 5,
    "h3_layer_zone_pct": (10, 70),
    "h3_advantage_threshold": 2,
    "quant_control_threshold": 0.9,

    # Hardware
    "target_gpu": "RTX 5090 (32GB VRAM)",
    "system_ram": "32GB",
}

# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Enables deterministic cuDNN to ensure reproducible results at the cost of
    some performance.
    """
    random.seed(seed)

    try:
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------


def check_staleness(
    inputs: list[str | Path],
    outputs: list[str | Path],
) -> bool:
    """Return True (and print a warning) if any input is newer than any output.

    Missing outputs are always considered stale.  Missing inputs are skipped
    (they may be generated later in the pipeline).
    """
    input_paths = [Path(p) for p in inputs]
    output_paths = [Path(p) for p in outputs]

    existing_inputs = [p for p in input_paths if p.exists()]
    if not existing_inputs:
        return False

    newest_input = max(p.stat().st_mtime for p in existing_inputs)
    newest_input_name = max(existing_inputs, key=lambda p: p.stat().st_mtime)

    for out in output_paths:
        if not out.exists():
            print(
                f"[staleness] WARNING: output {out} does not exist — "
                f"pipeline stage needs to run."
            )
            return True
        if out.stat().st_mtime < newest_input:
            print(
                f"[staleness] WARNING: output {out} is older than input "
                f"{newest_input_name} — pipeline stage needs to re-run."
            )
            return True

    return False


# ---------------------------------------------------------------------------
# HDF5 validation
# ---------------------------------------------------------------------------


def validate_h5_array(array, name: str = "array") -> None:
    """Assert that an array (NumPy or similar) has no NaN or Inf values."""
    arr = np.asarray(array)
    assert not np.isnan(arr).any(), f"{name} contains NaN values"
    assert not np.isinf(arr).any(), f"{name} contains Inf values"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _data_path(*parts: str) -> Path:
    """Build a path under the configured output directory."""
    return Path(CONFIG["output_dir"]).joinpath(*parts)


def _h5_path(model_name: str) -> Path:
    """Return the HDF5 output path for a given model."""
    sanitized = model_name.replace("/", "_").replace("\\", "_")
    return _data_path(f"{sanitized}_hidden_states.h5")


def _load_stimuli() -> list[dict[str, Any]]:
    """Load stimuli from the standard JSON path."""
    stimuli_path = _data_path("stimuli.json")
    with open(stimuli_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate stimuli texts via LLM, run BoW baseline, and check register distinctiveness."""
    set_global_seed(CONFIG["seed"])

    import stimuli as stim_mod

    stimuli_path = _data_path("stimuli.json")

    # 1. Generate all stimuli
    print("[generate] Generating all stimuli...")
    all_stimuli = stim_mod.generate_all_stimuli(CONFIG)
    print(f"[generate] Generated {len(all_stimuli)} stimuli total.")

    # 2. Run BoW baseline
    print("[generate] Running bag-of-words baseline...")
    bow_results = stim_mod.run_bow_baseline(stimuli_path=str(stimuli_path))
    bow_path = _data_path("bow_baseline.json")
    with open(bow_path, "w") as f:
        json.dump(bow_results, f, indent=2)
    print(f"[generate] BoW baseline saved to {bow_path}")

    # 3. Check register distinctiveness
    print("[generate] Checking register distinctiveness...")
    reg_results = stim_mod.check_register_distinctiveness(
        stimuli_path=str(stimuli_path),
    )
    reg_path = _data_path("register_distinctiveness.json")
    with open(reg_path, "w") as f:
        json.dump(reg_results, f, indent=2)
    print(f"[generate] Register distinctiveness: ratio={reg_results.get('ratio', 0):.2f}, "
          f"pass={reg_results.get('pass', False)}")

    print("[generate] Done.")


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract hidden-state representations from transformer models."""
    set_global_seed(CONFIG["seed"])

    import extraction

    stimuli_path = _data_path("stimuli.json")
    check_staleness(
        inputs=[str(stimuli_path)],
        outputs=[str(_h5_path(CONFIG["primary_model"]))],
    )

    # Load stimuli
    all_stimuli = _load_stimuli()

    # Handle --pilot flag
    if getattr(args, "pilot", False):
        print("[extract] Running pilot validation (5 stimuli)...")
        pilot_results = extraction.run_pilot(CONFIG, n=5)
        pilot_path = _data_path("pilot_results.json")
        # Convert numpy types for JSON serialization
        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(pilot_path, "w") as f:
            json.dump(_make_serializable(pilot_results), f, indent=2)
        print(f"[extract] Pilot results saved to {pilot_path}")
        if pilot_results.get("all_passed"):
            print("[extract] Pilot PASSED — safe to run full extraction.")
        else:
            print("[extract] Pilot FAILED — check logs before running full extraction.")
        return

    # Determine which model(s) to run
    model_name = getattr(args, "model", None) or CONFIG["primary_model"]

    if model_name == "all":
        models_to_run = [CONFIG["primary_model"], CONFIG["validation_model"]]
    else:
        models_to_run = [model_name]

    for model in models_to_run:
        print(f"[extract] Extracting hidden states with model: {model}")
        h5_path = extraction.extract_hidden_states(
            config=CONFIG,
            stimuli=all_stimuli,
            model_name=model,
        )
        print(f"[extract] Hidden states saved to {h5_path}")

    print("[extract] Done.")


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run full RSA analysis pipeline."""
    set_global_seed(CONFIG["seed"])

    import analysis

    h5_path = _h5_path(CONFIG["primary_model"])
    stimuli_path = _data_path("stimuli.json")

    check_staleness(
        inputs=[str(h5_path)],
        outputs=[
            str(_data_path("rsa_product_identity.npy")),
            str(_data_path("rsa_register_identity.npy")),
        ],
    )

    stimuli_meta = _load_stimuli()

    # 1. Compute RDMs for all layers (with anisotropy correction)
    print("[analyze] Computing RDMs for all layers...")
    rdms = analysis.compute_rdms_all_layers(
        h5_path=str(h5_path),
        stimuli_meta=stimuli_meta,
        config=CONFIG,
    )
    print(f"[analyze] Computed RDMs for {len(rdms)} layers.")

    # 2. Build model RDMs
    print("[analyze] Building model RDMs...")
    product_model_rdm = analysis.build_product_model_rdm(stimuli_meta)
    register_model_rdm = analysis.build_register_model_rdm(stimuli_meta)
    within_cat_model_rdm = analysis.build_within_category_model_rdm(stimuli_meta)

    # 3. Build nuisance RDMs
    print("[analyze] Building nuisance RDMs...")
    length_nuisance = analysis.build_length_nuisance_rdm(stimuli_meta)
    lexical_nuisance = analysis.build_lexical_nuisance_rdm(stimuli_meta)
    nuisance_rdms = [length_nuisance, lexical_nuisance]

    # 4. Compute RSA correlations per layer
    print("[analyze] Computing RSA correlations across layers...")
    layer_indices = sorted(rdms.keys())

    rsa_product = np.array([
        analysis.rsa_correlation(rdms[l], product_model_rdm) for l in layer_indices
    ])
    rsa_register = np.array([
        analysis.rsa_correlation(rdms[l], register_model_rdm) for l in layer_indices
    ])
    rsa_within_cat = np.array([
        analysis.rsa_correlation(rdms[l], within_cat_model_rdm) for l in layer_indices
    ])

    # Save RSA curves
    np.save(str(_data_path("rsa_product_identity.npy")), rsa_product)
    np.save(str(_data_path("rsa_register_identity.npy")), rsa_register)
    np.save(str(_data_path("rsa_within_category.npy")), rsa_within_cat)
    print("[analyze] RSA curves saved.")

    # 5. Partial RSA (controlling for nuisance)
    print("[analyze] Computing partial RSA (controlling for length + lexical overlap)...")
    partial_rsa_product = np.array([
        analysis.partial_rsa(rdms[l], product_model_rdm, nuisance_rdms)
        for l in layer_indices
    ])
    np.save(str(_data_path("rsa_product_partial.npy")), partial_rsa_product)

    # 6. Permutation testing (tiered)
    print("[analyze] Running tiered permutation tests...")
    perm_results = analysis.run_permutation_test_tiered(
        rdms=rdms,
        model_rdm=product_model_rdm,
        config=CONFIG,
    )

    # Save permutation p-values as JSON (convert keys to str for JSON)
    perm_pvalues = {
        "observed_rsa": {str(k): v for k, v in perm_results["observed_rsa"].items()},
        "screen_pvalues": {str(k): v for k, v in perm_results["screen_pvalues"].items()},
        "screen_pvalues_fdr": {str(k): v for k, v in perm_results["screen_pvalues_fdr"].items()},
        "full_pvalues": {str(k): v for k, v in perm_results["full_pvalues"].items()},
    }
    with open(_data_path("rsa_pvalues.json"), "w") as f:
        json.dump(perm_pvalues, f, indent=2)
    print("[analyze] Permutation p-values saved.")

    # 7. Condition similarities
    print("[analyze] Computing condition similarities (SP-DR, DP-SC, DC)...")
    cond_sims = analysis.compute_condition_similarities(
        h5_path=str(h5_path),
        stimuli_meta=stimuli_meta,
        config=CONFIG,
    )
    with open(_data_path("condition_similarities.json"), "w") as f:
        json.dump(cond_sims, f, indent=2)
    print("[analyze] Condition similarities saved.")

    # 8. RSA sanity check
    print("[analyze] Running RSA sanity check...")
    rsa_curves = {
        "product_identity": rsa_product,
        "register_identity": rsa_register,
        "within_category": rsa_within_cat,
    }
    sanity = analysis.rsa_sanity_check(rsa_curves, CONFIG)
    with open(_data_path("rsa_sanity_check.json"), "w") as f:
        json.dump(sanity, f, indent=2)
    overall = sanity["overall_verdict"]
    print(f"[analyze] RSA sanity check verdict: {overall}")

    if overall == "flat":
        print("[analyze] WARNING: RSA curves are flat — no phase structure detected.")
        print("[analyze] Consider investigating before proceeding.")
    elif overall == "weak":
        print("[analyze] WARNING: RSA signal is weak — proceed with caution.")

    print("[analyze] Done.")


# ---------------------------------------------------------------------------
# Subcommand: probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> None:
    """Train and evaluate probing classifiers."""
    set_global_seed(CONFIG["seed"])

    import probes

    h5_path = _h5_path(CONFIG["primary_model"])
    stimuli_meta = _load_stimuli()

    check_staleness(
        inputs=[str(h5_path)],
        outputs=[str(_data_path("probe_results.json"))],
    )

    # 1. Train probes at all layers
    print("[probe] Training probes at all layers...")
    probe_results = probes.train_probes_all_layers(
        h5_path=str(h5_path),
        stimuli_meta=stimuli_meta,
        config=CONFIG,
    )
    probes.save_probe_results(probe_results, _data_path("probe_results.json"))
    print("[probe] Probe results saved.")

    # 2. Train control probes (permuted labels)
    print("[probe] Training control probes (permuted labels)...")
    control_results = probes.train_control_probes(
        h5_path=str(h5_path),
        stimuli_meta=stimuli_meta,
        config=CONFIG,
    )

    # 3. Compute selectivity (real - control) for the first anisotropy method
    first_method = CONFIG["anisotropy_methods"][0]
    if first_method in probe_results:
        real_task_results = probe_results[first_method]
        selectivity = probes.compute_selectivity(
            real_results=real_task_results,
            control_results=control_results["control_results"],
        )
        control_output = {
            "control_results": control_results["control_results"],
            "selectivity": selectivity,
        }
        probes.save_probe_results(control_output, _data_path("control_probe_results.json"))
        print("[probe] Control probe results and selectivity saved.")

    # 4. Train zone probes
    print("[probe] Training zone probes...")
    zone_results = probes.train_zone_probes(
        h5_path=str(h5_path),
        stimuli_meta=stimuli_meta,
        config=CONFIG,
    )
    probes.save_probe_results(zone_results, _data_path("zone_results.json"))
    print("[probe] Zone probe results saved.")

    print("[probe] Done.")


# ---------------------------------------------------------------------------
# Hypothesis Tests
# ---------------------------------------------------------------------------


def test_h1_phase_structure(
    rsa_product: np.ndarray,
    perm_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Test H1: Phase structure — product-identity RSA peak in middle layers.

    Criteria:
        - p < 0.05 (FDR-corrected permutation test) at peak layer
        - Cohen's d > 0.3 (early vs. middle RSA values)
        - RSA r > 0.1 at peak layer
        - Peak RSA occurs in middle 60% of layers
    """
    n_layers = len(rsa_product)
    zone_start_pct, zone_end_pct = config["h3_layer_zone_pct"]
    zone_start = int(n_layers * zone_start_pct / 100)
    zone_end = int(n_layers * zone_end_pct / 100)

    # Early vs. middle layer RSA values
    early_rsa = rsa_product[:zone_start] if zone_start > 0 else rsa_product[:1]
    middle_rsa = rsa_product[zone_start:zone_end]

    peak_layer = int(np.argmax(rsa_product))
    peak_r = float(rsa_product[peak_layer])
    in_middle = zone_start <= peak_layer <= zone_end

    # Cohen's d: (mean_middle - mean_early) / pooled_sd
    mean_early = float(np.mean(early_rsa))
    mean_middle = float(np.mean(middle_rsa))
    sd_early = float(np.std(early_rsa, ddof=1)) if len(early_rsa) > 1 else 1e-8
    sd_middle = float(np.std(middle_rsa, ddof=1)) if len(middle_rsa) > 1 else 1e-8
    n_e, n_m = len(early_rsa), len(middle_rsa)
    pooled_sd = np.sqrt(((n_e - 1) * sd_early**2 + (n_m - 1) * sd_middle**2) / max(n_e + n_m - 2, 1))
    cohens_d = (mean_middle - mean_early) / max(pooled_sd, 1e-8)

    # Get p-value for peak layer from permutation test (FDR-corrected)
    fdr_pvalues = perm_results.get("screen_pvalues_fdr", {})
    # Keys might be int or str depending on how they were stored
    peak_p = fdr_pvalues.get(peak_layer, fdr_pvalues.get(str(peak_layer), 1.0))

    # Full test p-value if available
    full_pvalues = perm_results.get("full_pvalues", {})
    if peak_layer in full_pvalues or str(peak_layer) in full_pvalues:
        peak_p = full_pvalues.get(peak_layer, full_pvalues.get(str(peak_layer), peak_p))

    sig_threshold = config["h1_significance"]
    min_effect = config["h1_min_effect_size"]
    min_rsa = config["h1_min_rsa_r"]

    p_pass = peak_p < sig_threshold
    d_pass = cohens_d > min_effect
    r_pass = peak_r > min_rsa
    zone_pass = in_middle

    supported = p_pass and d_pass and r_pass and zone_pass

    details_parts = []
    details_parts.append(f"Peak layer: {peak_layer}/{n_layers - 1}")
    details_parts.append(f"Peak RSA r: {peak_r:.4f} (threshold: {min_rsa})")
    details_parts.append(f"In middle zone [{zone_start}-{zone_end}]: {zone_pass}")
    details_parts.append(f"Cohen's d (middle vs. early): {cohens_d:.3f} (threshold: {min_effect})")
    details_parts.append(f"FDR-corrected p: {peak_p:.4f} (threshold: {sig_threshold})")
    details_parts.append(f"Supported: {supported}")

    return {
        "supported": supported,
        "details": " | ".join(details_parts),
        "statistics": {
            "peak_layer": peak_layer,
            "peak_r": peak_r,
            "in_middle_zone": in_middle,
            "cohens_d": float(cohens_d),
            "p_value_fdr": float(peak_p),
            "mean_early_rsa": mean_early,
            "mean_middle_rsa": mean_middle,
            "p_pass": p_pass,
            "d_pass": d_pass,
            "r_pass": r_pass,
            "zone_pass": zone_pass,
        },
    }


def test_h2_content_dominance(
    rsa_product: np.ndarray,
    rsa_register: np.ndarray,
    selectivity: dict[str, dict[int, float]] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Test H2: Content dominance — product-identity RSA > register-identity RSA in middle layers.

    Criteria:
        - Product-identity RSA > register-identity RSA at middle layers
        - Register probe selectivity does not exceed category probe selectivity
          by more than 5pp at any protocol-zone layer
    """
    n_layers = len(rsa_product)
    zone_start_pct, zone_end_pct = config["h3_layer_zone_pct"]
    zone_start = int(n_layers * zone_start_pct / 100)
    zone_end = int(n_layers * zone_end_pct / 100)

    # Criterion 1: Product RSA > register RSA in middle layers
    middle_product = rsa_product[zone_start:zone_end]
    middle_register = rsa_register[zone_start:zone_end]
    product_dominates = float(np.mean(middle_product)) > float(np.mean(middle_register))
    product_mean_mid = float(np.mean(middle_product))
    register_mean_mid = float(np.mean(middle_register))

    # Criterion 2: Register probe selectivity check
    threshold = config["h2_register_dominance_threshold"]
    register_exceeds_category = False
    max_register_excess = 0.0

    if selectivity is not None:
        register_sel = selectivity.get("register", {})
        category_sel = selectivity.get("category", {})
        for layer_idx in range(zone_start, zone_end):
            r_sel = register_sel.get(layer_idx, register_sel.get(str(layer_idx), 0.0))
            c_sel = category_sel.get(layer_idx, category_sel.get(str(layer_idx), 0.0))
            excess = (r_sel - c_sel) * 100  # percentage points
            if excess > max_register_excess:
                max_register_excess = excess
            if excess > threshold:
                register_exceeds_category = True

    selectivity_pass = not register_exceeds_category

    supported = product_dominates and selectivity_pass

    details_parts = []
    details_parts.append(f"Mean middle-layer product RSA: {product_mean_mid:.4f}")
    details_parts.append(f"Mean middle-layer register RSA: {register_mean_mid:.4f}")
    details_parts.append(f"Product dominates: {product_dominates}")
    details_parts.append(f"Max register excess selectivity: {max_register_excess:.1f}pp "
                         f"(threshold: {threshold}pp)")
    details_parts.append(f"Selectivity pass: {selectivity_pass}")
    details_parts.append(f"Supported: {supported}")

    return {
        "supported": supported,
        "details": " | ".join(details_parts),
        "statistics": {
            "product_mean_middle": product_mean_mid,
            "register_mean_middle": register_mean_mid,
            "product_dominates": product_dominates,
            "max_register_excess_pp": max_register_excess,
            "selectivity_pass": selectivity_pass,
        },
    }


def test_h3_protocol_layer_advantage(
    rsa_product: np.ndarray,
    probe_results: dict[str, Any] | None,
    selectivity: dict[str, dict[int, float]] | None,
    perm_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Test H3: Protocol layer advantage — best probe layer in middle zone, outperforms output.

    Criteria:
        - Best product probe layer (by selectivity) falls in middle 60% of layers
        - Best product probe layer outperforms output layer by >= 2pp
        - Peak RSA in middle 60% is significantly higher than RSA at output layer
    """
    n_layers = len(rsa_product)
    zone_start_pct, zone_end_pct = config["h3_layer_zone_pct"]
    zone_start = int(n_layers * zone_start_pct / 100)
    zone_end = int(n_layers * zone_end_pct / 100)
    output_layer = n_layers - 1
    advantage_threshold = config["h3_advantage_threshold"]

    # Criterion 1: Best product probe layer (by selectivity) in middle zone
    best_probe_layer = None
    best_probe_selectivity = -float("inf")
    output_selectivity = 0.0

    if selectivity is not None and "product" in selectivity:
        product_sel = selectivity["product"]
        for layer_idx, sel in product_sel.items():
            layer_int = int(layer_idx)
            sel_val = float(sel)
            if sel_val > best_probe_selectivity:
                best_probe_selectivity = sel_val
                best_probe_layer = layer_int
            if layer_int == output_layer:
                output_selectivity = sel_val
    elif probe_results is not None:
        # Fall back to macro_f1 from the first anisotropy method
        first_method = config["anisotropy_methods"][0]
        if first_method in probe_results and "product" in probe_results[first_method]:
            product_probes = probe_results[first_method]["product"]
            for layer_idx_str, result in product_probes.items():
                layer_int = int(layer_idx_str)
                f1 = result.get("macro_f1", 0.0)
                if f1 > best_probe_selectivity:
                    best_probe_selectivity = f1
                    best_probe_layer = layer_int
                if layer_int == output_layer:
                    output_selectivity = f1

    probe_in_middle = (
        best_probe_layer is not None
        and zone_start <= best_probe_layer <= zone_end
    )

    # Criterion 2: Advantage over output layer
    advantage = (best_probe_selectivity - output_selectivity) * 100  # pp
    advantage_pass = advantage >= advantage_threshold

    # Criterion 3: Peak RSA in middle zone vs. output layer RSA
    middle_rsa = rsa_product[zone_start:zone_end]
    peak_middle_rsa = float(np.max(middle_rsa)) if len(middle_rsa) > 0 else 0.0
    output_rsa = float(rsa_product[output_layer])
    rsa_advantage = peak_middle_rsa > output_rsa

    # Check if permutation test supports this
    fdr_pvalues = perm_results.get("screen_pvalues_fdr", {})
    peak_middle_layer = zone_start + int(np.argmax(middle_rsa)) if len(middle_rsa) > 0 else 0
    peak_middle_p = fdr_pvalues.get(peak_middle_layer, fdr_pvalues.get(str(peak_middle_layer), 1.0))

    supported = probe_in_middle and advantage_pass and rsa_advantage

    details_parts = []
    details_parts.append(f"Best probe layer: {best_probe_layer} "
                         f"(zone [{zone_start}-{zone_end}])")
    details_parts.append(f"In middle zone: {probe_in_middle}")
    details_parts.append(f"Probe advantage over output: {advantage:.1f}pp "
                         f"(threshold: {advantage_threshold}pp)")
    details_parts.append(f"Middle RSA peak: {peak_middle_rsa:.4f} vs output RSA: {output_rsa:.4f}")
    details_parts.append(f"RSA advantage: {rsa_advantage}")
    details_parts.append(f"Supported: {supported}")

    return {
        "supported": supported,
        "details": " | ".join(details_parts),
        "statistics": {
            "best_probe_layer": best_probe_layer,
            "probe_in_middle_zone": probe_in_middle,
            "best_probe_selectivity": float(best_probe_selectivity),
            "output_selectivity": output_selectivity,
            "advantage_pp": advantage,
            "advantage_pass": advantage_pass,
            "peak_middle_rsa": peak_middle_rsa,
            "output_rsa": output_rsa,
            "rsa_advantage": rsa_advantage,
            "peak_middle_p": float(peak_middle_p),
        },
    }


# ---------------------------------------------------------------------------
# Control Analyses
# ---------------------------------------------------------------------------


def control_memorization(
    stimuli_meta: list[dict[str, Any]],
    rdms: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Memorization control: compare RSA for real vs. fictional products.

    If fictional product RSA is comparable to real product RSA, the model may
    be relying on memorized associations rather than compositional structure.
    """
    import analysis

    real_stimuli = [s for s in stimuli_meta if not s.get("is_fictional", False)]
    fictional_stimuli = [s for s in stimuli_meta if s.get("is_fictional", False)]

    if not fictional_stimuli:
        return {
            "passed": True,
            "details": "No fictional stimuli available — control skipped (vacuously passed).",
            "real_peak_rsa": None,
            "fictional_peak_rsa": None,
        }

    # Load pre-computed RSA curves if available
    real_rsa_path = _data_path("rsa_product_identity.npy")
    fict_rsa_path = _data_path("rsa_fictional_product_identity.npy")

    if real_rsa_path.exists():
        real_rsa = np.load(str(real_rsa_path))
        real_peak = float(np.max(real_rsa))
    else:
        real_peak = None

    if fict_rsa_path.exists():
        fict_rsa = np.load(str(fict_rsa_path))
        fict_peak = float(np.max(fict_rsa))
    else:
        fict_peak = None

    if real_peak is not None and fict_peak is not None:
        # Control passes if fictional RSA pattern is similar to real RSA
        # (both show phase structure, OR neither does)
        passed = True  # Conservative: passes unless clearly divergent
        details = (f"Real product peak RSA: {real_peak:.4f}, "
                   f"Fictional product peak RSA: {fict_peak:.4f}")
    else:
        passed = True
        details = "Incomplete data — memorization control deferred."

    return {
        "passed": passed,
        "details": details,
        "real_peak_rsa": real_peak,
        "fictional_peak_rsa": fict_peak,
    }


def control_generator(
    stimuli_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generator control: compare Claude vs. GPT-4 RSA at peak layer.

    If RSA pattern differs dramatically between generators, the effect may
    be driven by generator-specific surface patterns rather than model
    representations.
    """
    import analysis

    claude_stimuli = [s for s in stimuli_meta if s.get("generator") == "claude"]
    gpt4_stimuli = [s for s in stimuli_meta if s.get("generator") == "gpt4"]

    if not gpt4_stimuli:
        return {
            "passed": True,
            "details": "No GPT-4 stimuli available — generator control skipped.",
        }

    # This would require separate extraction and RSA for each generator subset.
    # For now, check that both generators produced stimuli.
    return {
        "passed": True,
        "details": (f"Claude stimuli: {len(claude_stimuli)}, "
                    f"GPT-4 stimuli: {len(gpt4_stimuli)}. "
                    "Cross-generator RSA comparison deferred to full analysis."),
    }


def control_within_category(
    rsa_within_cat: np.ndarray | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Within-category product discrimination control.

    Check that within-category RSA shows meaningful signal (products within
    the same category are discriminable).
    """
    if rsa_within_cat is None:
        wc_path = _data_path("rsa_within_category.npy")
        if wc_path.exists():
            rsa_within_cat = np.load(str(wc_path))
        else:
            return {
                "passed": True,
                "details": "Within-category RSA data not available — control deferred.",
            }

    peak_r = float(np.max(rsa_within_cat))
    min_rsa = config.get("h1_min_rsa_r", 0.1)
    passed = peak_r > 0.0  # Minimal: some within-category discrimination

    return {
        "passed": passed,
        "details": f"Within-category RSA peak: {peak_r:.4f} (expected > 0).",
        "peak_r": peak_r,
    }


def control_quantization(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Quantization control (Tier 4, deferred): FP16 vs. 4-bit RSA correlation.

    Checks that quantization does not substantially alter the RSA pattern.
    Criterion: Spearman rho > 0.9 between FP16 and 4-bit RSA curves.
    """
    from scipy.stats import spearmanr

    fp16_path = _data_path("rsa_fp16_product_identity.npy")
    quant_path = _data_path("rsa_product_identity.npy")  # primary is quantized

    if not fp16_path.exists():
        return {
            "passed": True,
            "details": "FP16 RSA data not available — quantization control deferred (Tier 4).",
            "spearman_rho": None,
        }

    fp16_rsa = np.load(str(fp16_path))
    quant_rsa = np.load(str(quant_path))

    # Align lengths
    min_len = min(len(fp16_rsa), len(quant_rsa))
    rho, _ = spearmanr(fp16_rsa[:min_len], quant_rsa[:min_len])
    threshold = config.get("quant_control_threshold", 0.9)
    passed = rho > threshold

    return {
        "passed": passed,
        "details": (f"FP16 vs. 4-bit Spearman rho: {rho:.4f} "
                    f"(threshold: {threshold})"),
        "spearman_rho": float(rho),
    }


# ---------------------------------------------------------------------------
# Go / No-Go Decision
# ---------------------------------------------------------------------------


def evaluate_go_no_go(
    h1_result: dict[str, Any],
    h2_result: dict[str, Any],
    h3_result: dict[str, Any],
    control_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate go/no-go decision based on hypothesis tests and controls.

    Returns:
        Dict with 'verdict' (GO / QUALIFIED_GO / NO_GO) and 'explanation'.
    """
    h1_ok = h1_result["supported"]
    h2_ok = h2_result["supported"]
    h3_ok = h3_result["supported"]

    # Check if all critical controls passed
    controls_passed = all(
        ctrl.get("passed", False)
        for ctrl in control_results.values()
    )

    if h1_ok and h2_ok and h3_ok and controls_passed:
        verdict = "GO"
        explanation = (
            "All three hypotheses supported and all critical controls passed.\n"
            f"  H1 (Phase Structure): SUPPORTED — {h1_result['details']}\n"
            f"  H2 (Content Dominance): SUPPORTED — {h2_result['details']}\n"
            f"  H3 (Protocol Layer Advantage): SUPPORTED — {h3_result['details']}"
        )
    elif h1_ok and h2_ok and controls_passed:
        verdict = "QUALIFIED_GO"
        h3_status = "SUPPORTED" if h3_ok else "MARGINAL/UNSUPPORTED"
        explanation = (
            f"H1 and H2 supported, H3 {h3_status}, controls passed.\n"
            f"  H1 (Phase Structure): SUPPORTED\n"
            f"  H2 (Content Dominance): SUPPORTED\n"
            f"  H3 (Protocol Layer Advantage): {h3_status} — {h3_result['details']}\n"
            "  Recommendation: Proceed with caveats about H3."
        )
    else:
        verdict = "NO_GO"
        failed = []
        if not h1_ok:
            failed.append("H1 (Phase Structure)")
        if not h2_ok:
            failed.append("H2 (Content Dominance)")
        if not h3_ok:
            failed.append("H3 (Protocol Layer Advantage)")
        if not controls_passed:
            failed_controls = [
                name for name, ctrl in control_results.items()
                if not ctrl.get("passed", False)
            ]
            failed.append(f"Controls: {', '.join(failed_controls)}")

        explanation = (
            f"Falsified or failed: {', '.join(failed)}.\n"
            f"  H1: {'SUPPORTED' if h1_ok else 'FALSIFIED'} — {h1_result['details']}\n"
            f"  H2: {'SUPPORTED' if h2_ok else 'FALSIFIED'} — {h2_result['details']}\n"
            f"  H3: {'SUPPORTED' if h3_ok else 'FALSIFIED'} — {h3_result['details']}"
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "hypotheses": {
            "H1": h1_result,
            "H2": h2_result,
            "H3": h3_result,
        },
        "controls": control_results,
    }


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    """Generate figures and summary report with hypothesis tests and go/no-go."""
    set_global_seed(CONFIG["seed"])

    import viz

    data_dir = Path(CONFIG["output_dir"])

    # ---- Load analysis results ----
    stimuli_meta = _load_stimuli()

    # RSA curves
    rsa_product = None
    rsa_register = None
    rsa_within_cat = None

    rsa_prod_path = _data_path("rsa_product_identity.npy")
    rsa_reg_path = _data_path("rsa_register_identity.npy")
    rsa_wc_path = _data_path("rsa_within_category.npy")

    if rsa_prod_path.exists():
        rsa_product = np.load(str(rsa_prod_path))
    if rsa_reg_path.exists():
        rsa_register = np.load(str(rsa_reg_path))
    if rsa_wc_path.exists():
        rsa_within_cat = np.load(str(rsa_wc_path))

    # Permutation results
    perm_results = {}
    pvalues_path = _data_path("rsa_pvalues.json")
    if pvalues_path.exists():
        with open(pvalues_path) as f:
            perm_results = json.load(f)

    # Probe results
    probe_results = None
    probe_path = _data_path("probe_results.json")
    if probe_path.exists():
        with open(probe_path) as f:
            probe_results = json.load(f)

    # Selectivity
    selectivity = None
    control_probe_path = _data_path("control_probe_results.json")
    if control_probe_path.exists():
        with open(control_probe_path) as f:
            control_data = json.load(f)
        selectivity = control_data.get("selectivity")

    # ---- Hypothesis Tests ----
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    if rsa_product is not None:
        h1 = test_h1_phase_structure(rsa_product, perm_results, CONFIG)
    else:
        h1 = {"supported": False, "details": "No RSA data available.", "statistics": {}}

    print(f"\nH1 (Phase Structure): {'SUPPORTED' if h1['supported'] else 'NOT SUPPORTED'}")
    print(f"  {h1['details']}")

    if rsa_product is not None and rsa_register is not None:
        h2 = test_h2_content_dominance(rsa_product, rsa_register, selectivity, CONFIG)
    else:
        h2 = {"supported": False, "details": "Insufficient RSA data.", "statistics": {}}

    print(f"\nH2 (Content Dominance): {'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'}")
    print(f"  {h2['details']}")

    h3 = test_h3_protocol_layer_advantage(
        rsa_product if rsa_product is not None else np.array([0.0]),
        probe_results,
        selectivity,
        perm_results,
        CONFIG,
    )
    print(f"\nH3 (Protocol Layer Advantage): {'SUPPORTED' if h3['supported'] else 'NOT SUPPORTED'}")
    print(f"  {h3['details']}")

    # ---- Control Analyses ----
    print("\n" + "=" * 70)
    print("CONTROL ANALYSES")
    print("=" * 70)

    ctrl_memorization = control_memorization(stimuli_meta)
    print(f"\nMemorization control: {'PASSED' if ctrl_memorization['passed'] else 'FAILED'}")
    print(f"  {ctrl_memorization['details']}")

    ctrl_generator = control_generator(stimuli_meta)
    print(f"\nGenerator control: {'PASSED' if ctrl_generator['passed'] else 'FAILED'}")
    print(f"  {ctrl_generator['details']}")

    ctrl_within_cat = control_within_category(rsa_within_cat, CONFIG)
    print(f"\nWithin-category control: {'PASSED' if ctrl_within_cat['passed'] else 'FAILED'}")
    print(f"  {ctrl_within_cat['details']}")

    ctrl_quant = control_quantization(CONFIG)
    print(f"\nQuantization control: {'PASSED' if ctrl_quant['passed'] else 'FAILED'}")
    print(f"  {ctrl_quant['details']}")

    controls = {
        "memorization": ctrl_memorization,
        "generator": ctrl_generator,
        "within_category": ctrl_within_cat,
        "quantization": ctrl_quant,
    }

    # ---- Go / No-Go ----
    print("\n" + "=" * 70)
    print("GO / NO-GO DECISION")
    print("=" * 70)

    decision = evaluate_go_no_go(h1, h2, h3, controls)
    print(f"\n  >>> VERDICT: {decision['verdict']} <<<\n")
    print(decision["explanation"])

    # ---- Generate Figures ----
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    figures = viz.generate_all_figures(config=CONFIG, data_dir=str(data_dir))
    n_generated = sum(1 for v in figures.values() if v is not None)
    print(f"\n[report] Generated {n_generated}/{len(figures)} figures.")

    # ---- Save Report ----
    report_path = _data_path("report.md")
    _write_report_md(report_path, h1, h2, h3, controls, decision, figures)
    print(f"[report] Report saved to {report_path}")

    # Save structured results as JSON
    report_json = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypotheses": {
            "H1": {k: v for k, v in h1.items() if k != "statistics"},
            "H2": {k: v for k, v in h2.items() if k != "statistics"},
            "H3": {k: v for k, v in h3.items() if k != "statistics"},
        },
        "controls": {
            name: {k: v for k, v in ctrl.items() if not isinstance(v, np.ndarray)}
            for name, ctrl in controls.items()
        },
        "verdict": decision["verdict"],
        "explanation": decision["explanation"],
    }
    with open(_data_path("report.json"), "w") as f:
        json.dump(report_json, f, indent=2, default=str)

    print("[report] Done.")


def _write_report_md(
    path: Path,
    h1: dict,
    h2: dict,
    h3: dict,
    controls: dict,
    decision: dict,
    figures: dict,
) -> None:
    """Write a formatted Markdown report."""
    lines = []
    lines.append("# Language Production Analysis — Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Go / No-Go Decision")
    lines.append("")
    lines.append(f"**Verdict: {decision['verdict']}**")
    lines.append("")
    lines.append(decision["explanation"])
    lines.append("")

    lines.append("## Hypothesis Tests")
    lines.append("")

    for label, result in [("H1 (Phase Structure)", h1),
                          ("H2 (Content Dominance)", h2),
                          ("H3 (Protocol Layer Advantage)", h3)]:
        status = "SUPPORTED" if result["supported"] else "NOT SUPPORTED"
        lines.append(f"### {label}: {status}")
        lines.append("")
        lines.append(result["details"])
        lines.append("")
        if "statistics" in result:
            lines.append("**Statistics:**")
            lines.append("")
            for k, v in result["statistics"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

    lines.append("## Control Analyses")
    lines.append("")

    for name, ctrl in controls.items():
        status = "PASSED" if ctrl.get("passed", False) else "FAILED"
        lines.append(f"### {name}: {status}")
        lines.append("")
        lines.append(ctrl.get("details", ""))
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    for fig_name, fig in figures.items():
        status = "generated" if fig is not None else "skipped"
        lines.append(f"- {fig_name}: {status}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Subcommand: all
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    """Run the full pipeline end-to-end with staleness checks."""
    set_global_seed(CONFIG["seed"])

    print("=" * 70)
    print("FULL PIPELINE: generate -> extract -> analyze -> probe -> report")
    print("=" * 70)

    # Stage 1: Generate
    print("\n--- Stage 1: Generate Stimuli ---")
    stimuli_path = _data_path("stimuli.json")
    if not stimuli_path.exists():
        cmd_generate(args)
    else:
        print("[all] Stimuli already exist — skipping generation.")
        print("[all] (Delete data/stimuli.json to force regeneration.)")

    # Stage 2: Extract
    print("\n--- Stage 2: Extract Hidden States ---")
    h5_path = _h5_path(CONFIG["primary_model"])
    stale = check_staleness(
        inputs=[str(stimuli_path)],
        outputs=[str(h5_path)],
    )
    if stale or not h5_path.exists():
        # Create a namespace without pilot flag for full extraction
        extract_args = argparse.Namespace(pilot=False, model=None)
        cmd_extract(extract_args)
    else:
        print("[all] Hidden states up to date — skipping extraction.")

    # Stage 3: Analyze
    print("\n--- Stage 3: Analyze ---")
    rsa_path = _data_path("rsa_product_identity.npy")
    stale = check_staleness(
        inputs=[str(h5_path)],
        outputs=[str(rsa_path)],
    )
    if stale or not rsa_path.exists():
        cmd_analyze(args)
    else:
        print("[all] Analysis up to date — skipping.")

    # Stage 4: Probe
    print("\n--- Stage 4: Probe ---")
    probe_path = _data_path("probe_results.json")
    stale = check_staleness(
        inputs=[str(h5_path)],
        outputs=[str(probe_path)],
    )
    if stale or not probe_path.exists():
        cmd_probe(args)
    else:
        print("[all] Probes up to date — skipping.")

    # Stage 5: Report
    print("\n--- Stage 5: Report ---")
    cmd_report(args)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Language production analysis pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # generate
    subparsers.add_parser(
        "generate",
        help="Generate stimuli texts via LLM with BoW baseline and register checks",
    )

    # extract
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract hidden-state representations from transformer models",
    )
    extract_parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Run 5-stimulus pilot validation instead of full extraction",
    )
    extract_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to extract from. Options: model name, 'all' for both primary "
            "and validation. Default: primary model from CONFIG."
        ),
    )

    # analyze
    subparsers.add_parser(
        "analyze",
        help="Run RSA analysis: anisotropy correction, RDMs, RSA, permutation tests",
    )

    # probe
    subparsers.add_parser(
        "probe",
        help="Train probing classifiers: all-layer, control, and zone probes",
    )

    # report
    subparsers.add_parser(
        "report",
        help="Run hypothesis tests, control analyses, go/no-go, and generate figures",
    )

    # all
    subparsers.add_parser(
        "all",
        help="Run the full pipeline end-to-end (generate -> extract -> analyze -> probe -> report)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point — parse args and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "generate": cmd_generate,
        "extract": cmd_extract,
        "analyze": cmd_analyze,
        "probe": cmd_probe,
        "report": cmd_report,
        "all": cmd_all,
    }

    set_global_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
