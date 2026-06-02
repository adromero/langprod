"""Extended smoke test v2: 6 products, within-category pairs, mean centering.

Tests whether mean-centered representations can differentiate coherence
levels, especially within the same product category.

Usage:
    python smoke_test_v2.py              # extract + analyze
    python smoke_test_v2.py --skip-extract  # analyze only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cosine

import analysis as ana

SMOKE_DIR = Path("data/coherence/smoke-test")
STIMULI_PATH = SMOKE_DIR / "stimuli_extended.json"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

CONFIG = {
    "seed": 42,
    "output_dir": str(SMOKE_DIR),
    "primary_model": MODEL_NAME,
    "batch_size": 1,
}

LAYERS = [10, 20, 30, 40, 50, 55, 60, 61, 63]
CORRECTIONS = ["none", "mean_centering"]


def load_stimuli() -> list[dict]:
    with open(STIMULI_PATH) as f:
        return json.load(f)


def run_extraction(stimuli: list[dict]) -> Path:
    import extraction
    # Use a separate HDF5 file for the extended set
    h5_path = extraction.extract_hidden_states(
        config=CONFIG,
        stimuli=stimuli,
        model_name=MODEL_NAME,
    )
    print(f"[v2] Hidden states saved to {h5_path}")
    return h5_path


def get_h5_path() -> Path:
    sanitized = MODEL_NAME.replace("/", "_").replace("\\", "_")
    return SMOKE_DIR / f"{sanitized}_hidden_states.h5"


def get_embeddings(h5_path: Path, stimuli: list[dict], layer: int) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        hs = f["hidden_states_mean_no_special"]
        ids = [x.decode() if isinstance(x, bytes) else x for x in f["stimulus_ids"][:]]
        sid_to_idx = {sid: i for i, sid in enumerate(ids)}
        embeddings = {}
        for stim in stimuli:
            sid = stim["stimulus_id"]
            if sid in sid_to_idx:
                embeddings[sid] = hs[sid_to_idx[sid], layer + 1, :]
    return embeddings


def apply_correction(embeddings: dict[str, np.ndarray], method: str) -> dict[str, np.ndarray]:
    if method == "none":
        return embeddings
    sids = list(embeddings.keys())
    matrix = np.array([embeddings[sid] for sid in sids])
    corrected = ana.correct_anisotropy(matrix, method=method)
    return {sid: corrected[i] for i, sid in enumerate(sids)}


def compute_product_coherence(stimuli: list[dict], embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean within-product pairwise cosine similarity."""
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


def main():
    parser = argparse.ArgumentParser(description="Extended smoke test v2")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    stimuli = load_stimuli()
    print(f"[v2] Loaded {len(stimuli)} stimuli ({len(set(s['product_id'] for s in stimuli))} products)")

    h5_path = get_h5_path()

    if not args.skip_extract:
        h5_path = run_extraction(stimuli)
    else:
        if not h5_path.exists():
            print(f"[v2] HDF5 not found. Run without --skip-extract first.")
            sys.exit(1)

    # Build product metadata
    product_meta = {}
    for s in stimuli:
        pid = s["product_id"]
        if pid not in product_meta:
            product_meta[pid] = {
                "category": s["category"],
                "expected": s.get("expected_coherence", "unknown"),
            }

    # =====================================================================
    # Layer x Correction sweep with per-product coherence
    # =====================================================================
    print("\n" + "=" * 110)
    print("PER-PRODUCT COHERENCE SCORES")
    print("=" * 110)

    # Product labels
    products_ordered = ["aquaphor_001", "cerave_001", "drunk_elephant_001",
                        "nature_made_001", "olly_001", "bloom_001"]
    labels = {
        "aquaphor_001": "Aquaphor(H)",
        "cerave_001": "CeraVe(H)",
        "drunk_elephant_001": "DrunkEl(L)",
        "nature_made_001": "NatMade(H)",
        "olly_001": "OLLY(L)",
        "bloom_001": "Bloom(L)",
    }

    header = f"{'Layer':>5} {'Corr':>14}"
    for pid in products_ordered:
        header += f" | {labels[pid]:>12}"
    header += " | SK-gap | SP-gap"
    print(header)
    print("-" * len(header))

    best_results = {"skincare_gap": 0, "supplement_gap": 0, "combined": 0,
                    "best_config": ""}

    for layer in LAYERS:
        embeddings = get_embeddings(h5_path, stimuli, layer)
        if len(embeddings) < 18:
            print(f"  Layer {layer}: only {len(embeddings)}/18 embeddings, skipping")
            continue

        for correction in CORRECTIONS:
            corrected = apply_correction(embeddings, correction)
            scores = compute_product_coherence(stimuli, corrected)

            row = f"{layer:>5} {correction:>14}"
            for pid in products_ordered:
                row += f" | {scores.get(pid, 0):>12.4f}"

            # Within-category gaps
            sk_high = np.mean([scores.get("aquaphor_001", 0), scores.get("cerave_001", 0)])
            sk_low = scores.get("drunk_elephant_001", 0)
            sk_gap = sk_high - sk_low

            sp_high = scores.get("nature_made_001", 0)
            sp_low = np.mean([scores.get("olly_001", 0), scores.get("bloom_001", 0)])
            sp_gap = sp_high - sp_low

            row += f" | {sk_gap:>+6.3f} | {sp_gap:>+6.3f}"
            print(row)

            combined = sk_gap + sp_gap
            if combined > best_results["combined"]:
                best_results = {
                    "skincare_gap": float(sk_gap),
                    "supplement_gap": float(sp_gap),
                    "combined": float(combined),
                    "best_config": f"layer={layer}, correction={correction}",
                    "layer": layer,
                    "correction": correction,
                    "scores": {pid: float(scores.get(pid, 0)) for pid in products_ordered},
                }

    # =====================================================================
    # Best result detail
    # =====================================================================
    print(f"\n{'=' * 110}")
    print(f"BEST CONFIGURATION: {best_results['best_config']}")
    print(f"{'=' * 110}")
    print(f"\nSkincare gap (high - low): {best_results['skincare_gap']:+.4f}")
    print(f"  High (Aquaphor, CeraVe avg) vs Low (Drunk Elephant)")
    print(f"Supplement gap (high - low): {best_results['supplement_gap']:+.4f}")
    print(f"  High (Nature Made) vs Low (OLLY, Bloom avg)")
    print(f"Combined gap: {best_results['combined']:+.4f}")

    if best_results["combined"] > 0:
        print(f"\nDIRECTIONAL SIGNAL: high-coherence products score higher than")
        print(f"  low-coherence products in both categories.")
    elif best_results["skincare_gap"] > 0 or best_results["supplement_gap"] > 0:
        print(f"\nPARTIAL SIGNAL: differentiation works in one category but not both.")
    else:
        print(f"\nNO SIGNAL: coherence differentiation not detected.")

    # =====================================================================
    # Pairwise within-category at best config
    # =====================================================================
    layer = best_results.get("layer", 61)
    correction = best_results.get("correction", "mean_centering")
    embeddings = get_embeddings(h5_path, stimuli, layer)
    corrected = apply_correction(embeddings, correction)

    print(f"\n--- Pairwise product similarities at best config (layer={layer}, {correction}) ---")

    # Build product centroids
    centroids = {}
    for pid in products_ordered:
        vecs = [corrected[s["stimulus_id"]] for s in stimuli
                if s["product_id"] == pid and s["stimulus_id"] in corrected]
        if vecs:
            centroids[pid] = np.mean(vecs, axis=0)

    print(f"\n{'':>15}", end="")
    for pid in products_ordered:
        print(f" {labels[pid]:>12}", end="")
    print()

    for pid_i in products_ordered:
        print(f"{labels[pid_i]:>15}", end="")
        for pid_j in products_ordered:
            if pid_i == pid_j:
                print(f" {'---':>12}", end="")
            else:
                sim = 1.0 - cosine(centroids[pid_i], centroids[pid_j])
                print(f" {sim:>12.4f}", end="")
        print()

    # Save results
    results_path = SMOKE_DIR / "smoke_test_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(best_results, f, indent=2)
    print(f"\n[v2] Results saved to {results_path}")


if __name__ == "__main__":
    main()
