"""Extended smoke test: anisotropy correction + multi-layer analysis.

Runs on existing smoke test HDF5 — no new extraction needed.
Tests whether anisotropy correction or different layers reveal
coherence differentiation that raw layer 61 doesn't.

Usage:
    python smoke_test_extended.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cosine

import analysis as ana

SMOKE_DIR = Path("data/coherence/smoke-test")
STIMULI_PATH = SMOKE_DIR / "stimuli.json"
H5_PATH = SMOKE_DIR / "Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4_hidden_states.h5"

# Also compare against the calibration dataset
CALIBRATION_H5 = Path("data/Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4_hidden_states.h5")
CALIBRATION_STIMULI = Path("data/stimuli.json")

LAYERS_TO_TEST = [10, 20, 30, 40, 50, 55, 60, 61, 62, 63]
CORRECTIONS = ["none", "mean_centering", "whitening"]


def load_stimuli(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_embeddings(h5_path: Path, stimuli: list[dict], layer_idx: int) -> dict[str, np.ndarray]:
    """Extract embeddings for all stimuli at a given layer (0-indexed, +1 for embedding offset)."""
    with h5py.File(h5_path, "r") as f:
        hs = f["hidden_states_mean_no_special"]
        ids = [x.decode() if isinstance(x, bytes) else x for x in f["stimulus_ids"][:]]
        sid_to_idx = {sid: i for i, sid in enumerate(ids)}

        embeddings = {}
        for stim in stimuli:
            sid = stim["stimulus_id"]
            if sid in sid_to_idx:
                embeddings[sid] = hs[sid_to_idx[sid], layer_idx + 1, :]
    return embeddings


def compute_gaps(stimuli: list[dict], embeddings: dict[str, np.ndarray]) -> dict:
    """Compute within-product vs between-product similarity gap."""
    within_sims = []
    between_sims = []

    items = [(s["product_id"], s["stimulus_id"]) for s in stimuli if s["stimulus_id"] in embeddings]

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pid_i, sid_i = items[i]
            pid_j, sid_j = items[j]
            sim = 1.0 - cosine(embeddings[sid_i], embeddings[sid_j])
            if pid_i == pid_j:
                within_sims.append(sim)
            else:
                between_sims.append(sim)

    if not within_sims or not between_sims:
        return {"within": 0, "between": 0, "gap": 0}

    return {
        "within": float(np.mean(within_sims)),
        "between": float(np.mean(between_sims)),
        "gap": float(np.mean(within_sims) - np.mean(between_sims)),
    }


def compute_per_product_coherence(stimuli: list[dict], embeddings: dict[str, np.ndarray]) -> dict:
    """Per-product mean within-product similarity."""
    products: dict[str, list[np.ndarray]] = {}
    for s in stimuli:
        if s["stimulus_id"] in embeddings:
            products.setdefault(s["product_id"], []).append(embeddings[s["stimulus_id"]])

    result = {}
    for pid, vecs in products.items():
        sims = []
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                sims.append(1.0 - cosine(vecs[a], vecs[b]))
        result[pid] = float(np.mean(sims)) if sims else 0.0
    return result


def apply_correction(embeddings: dict[str, np.ndarray], method: str) -> dict[str, np.ndarray]:
    """Apply anisotropy correction to a set of embeddings."""
    if method == "none":
        return embeddings

    sids = list(embeddings.keys())
    matrix = np.array([embeddings[sid] for sid in sids])
    corrected = ana.correct_anisotropy(matrix, method=method)
    return {sid: corrected[i] for i, sid in enumerate(sids)}


def main():
    stimuli = load_stimuli(STIMULI_PATH)
    print(f"Loaded {len(stimuli)} smoke test stimuli")

    # =====================================================================
    # Part 1: Multi-layer + anisotropy correction on smoke test data
    # =====================================================================
    print("\n" + "=" * 80)
    print("PART 1: LAYER x CORRECTION SWEEP (smoke test data)")
    print("=" * 80)

    print(f"\n{'Layer':>6} | {'Correction':>15} | {'Within':>8} | {'Between':>8} | {'Gap':>8} | {'Aquaphor':>9} | {'OLLY':>9} | {'Diff':>8}")
    print("-" * 95)

    best_gap = 0
    best_diff = 0
    best_config_gap = ""
    best_config_diff = ""

    for layer in LAYERS_TO_TEST:
        embeddings = get_embeddings(H5_PATH, stimuli, layer)
        if len(embeddings) < 6:
            continue

        for correction in CORRECTIONS:
            corrected = apply_correction(embeddings, correction)
            gaps = compute_gaps(stimuli, corrected)
            per_product = compute_per_product_coherence(stimuli, corrected)

            aq = per_product.get("aquaphor_001", 0)
            ol = per_product.get("olly_001", 0)
            diff = aq - ol  # positive = Aquaphor more coherent (expected)

            print(f"{layer:>6} | {correction:>15} | {gaps['within']:>8.4f} | {gaps['between']:>8.4f} | {gaps['gap']:>8.4f} | {aq:>9.4f} | {ol:>9.4f} | {diff:>+8.4f}")

            if gaps["gap"] > best_gap:
                best_gap = gaps["gap"]
                best_config_gap = f"layer={layer}, correction={correction}"
            if diff > best_diff:
                best_diff = diff
                best_config_diff = f"layer={layer}, correction={correction}"

    print(f"\nBest within/between gap: {best_gap:.4f} at {best_config_gap}")
    print(f"Best Aquaphor-OLLY diff: {best_diff:+.4f} at {best_config_diff}")

    # =====================================================================
    # Part 2: Compare against calibration dataset at best layer
    # =====================================================================
    if CALIBRATION_H5.exists() and CALIBRATION_STIMULI.exists():
        print("\n" + "=" * 80)
        print("PART 2: CALIBRATION REFERENCE (800 stimuli, layer 61)")
        print("=" * 80)

        cal_stimuli = load_stimuli(CALIBRATION_STIMULI)
        # Sample a few products from the same categories (skincare-ish, supplement-ish)
        # to see what calibration gaps look like
        cal_embeddings = get_embeddings(CALIBRATION_H5, cal_stimuli, 61)

        for correction in CORRECTIONS:
            corrected = apply_correction(cal_embeddings, correction)
            gaps = compute_gaps(cal_stimuli, corrected)
            print(f"  {correction:>15}: within={gaps['within']:.4f}, between={gaps['between']:.4f}, gap={gaps['gap']:.4f}")

        # Per-category breakdown
        print(f"\n  Per-product coherence (calibration, layer 61, whitened):")
        cal_embeddings_61 = get_embeddings(CALIBRATION_H5, cal_stimuli, 61)
        corrected_cal = apply_correction(cal_embeddings_61, "whitening")
        cal_coherence = compute_per_product_coherence(cal_stimuli, corrected_cal)

        # Show distribution stats
        scores = list(cal_coherence.values())
        print(f"  N products: {len(scores)}")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std:  {np.std(scores):.4f}")
        print(f"  Min:  {np.min(scores):.4f}")
        print(f"  Max:  {np.max(scores):.4f}")
        print(f"  Range: {np.max(scores) - np.min(scores):.4f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
