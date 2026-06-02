"""Smoke test: do real documents produce a coherence signal?

Extracts hidden states from 6 real documents (2 products x 3 channels),
computes pairwise cosine similarities at layer 61, and checks whether
within-product similarity exceeds between-product similarity.

Usage:
    python smoke_test.py              # extract + analyze
    python smoke_test.py --skip-extract  # analyze only (if HDF5 exists)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cosine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMOKE_DIR = Path("data/coherence/smoke-test")
STIMULI_PATH = SMOKE_DIR / "stimuli.json"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
TARGET_LAYER = 61  # product-identity peak from Protocol Layer Hypothesis

CONFIG = {
    "seed": 42,
    "output_dir": str(SMOKE_DIR),
    "primary_model": MODEL_NAME,
    "batch_size": 1,
}


def load_stimuli() -> list[dict]:
    with open(STIMULI_PATH) as f:
        return json.load(f)


def run_extraction(stimuli: list[dict]) -> Path:
    """Extract hidden states using the existing extraction module."""
    import extraction

    h5_path = extraction.extract_hidden_states(
        config=CONFIG,
        stimuli=stimuli,
        model_name=MODEL_NAME,
    )
    print(f"[smoke] Hidden states saved to {h5_path}")
    return h5_path


def get_h5_path() -> Path:
    """Derive HDF5 path the same way extraction.py does."""
    sanitized = MODEL_NAME.replace("/", "_").replace("\\", "_")
    return SMOKE_DIR / f"{sanitized}_hidden_states.h5"


def analyze(stimuli: list[dict], h5_path: Path) -> dict:
    """Compute within-product vs between-product cosine similarity at target layer."""
    with h5py.File(h5_path, "r") as f:
        # HDF5 format: hidden_states_mean_no_special shape (N, L+1, D)
        # stimulus_ids shape (N,) — maps row index to stimulus_id
        hs = f["hidden_states_mean_no_special"]
        ids = [x.decode() if isinstance(x, bytes) else x for x in f["stimulus_ids"][:]]

        sid_to_idx = {sid: i for i, sid in enumerate(ids)}

        embeddings = {}
        for stim in stimuli:
            sid = stim["stimulus_id"]
            if sid in sid_to_idx:
                # Index TARGET_LAYER+1 because index 0 = embedding layer
                vec = hs[sid_to_idx[sid], TARGET_LAYER + 1, :]
                embeddings[sid] = vec
            else:
                print(f"[smoke] WARNING: {sid} not found in HDF5")

    if len(embeddings) < 6:
        print(f"[smoke] Only {len(embeddings)}/6 stimuli found. Cannot proceed.")
        sys.exit(1)

    # Build product_id -> list of (stimulus_id, embedding)
    products: dict[str, list[tuple[str, np.ndarray]]] = {}
    for stim in stimuli:
        pid = stim["product_id"]
        sid = stim["stimulus_id"]
        if sid in embeddings:
            products.setdefault(pid, []).append((sid, embeddings[sid]))

    # Compute pairwise cosine similarities
    within_sims = []  # same product, different channel
    between_sims = []  # different product

    all_items = [(stim["product_id"], stim["stimulus_id"], embeddings[stim["stimulus_id"]])
                 for stim in stimuli if stim["stimulus_id"] in embeddings]

    for i in range(len(all_items)):
        for j in range(i + 1, len(all_items)):
            pid_i, sid_i, vec_i = all_items[i]
            pid_j, sid_j, vec_j = all_items[j]
            sim = 1.0 - cosine(vec_i, vec_j)  # cosine similarity
            pair_label = f"{sid_i} <-> {sid_j}"

            if pid_i == pid_j:
                within_sims.append((pair_label, sim))
            else:
                between_sims.append((pair_label, sim))

    # Report
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS — Layer", TARGET_LAYER)
    print("=" * 70)

    print(f"\n--- Within-product similarities (same product, different channel) ---")
    for label, sim in within_sims:
        print(f"  {sim:.4f}  {label}")
    mean_within = np.mean([s for _, s in within_sims])
    print(f"  MEAN: {mean_within:.4f}")

    print(f"\n--- Between-product similarities (different products) ---")
    for label, sim in between_sims:
        print(f"  {sim:.4f}  {label}")
    mean_between = np.mean([s for _, s in between_sims])
    print(f"  MEAN: {mean_between:.4f}")

    gap = mean_within - mean_between
    print(f"\n--- Verdict ---")
    print(f"  Within-product mean:  {mean_within:.4f}")
    print(f"  Between-product mean: {mean_between:.4f}")
    print(f"  Gap (within - between): {gap:.4f}")

    if gap > 0:
        print(f"\n  SIGNAL PRESENT: within-product similarity exceeds between-product.")
        print(f"  The directional signal exists in real documents.")
    else:
        print(f"\n  NO SIGNAL: between-product similarity >= within-product.")
        print(f"  The coherence signal does not appear in real documents at this layer.")

    # Per-product coherence (mean of within-product similarities)
    print(f"\n--- Per-product coherence ---")
    for pid, items in products.items():
        sims = []
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                sims.append(1.0 - cosine(items[a][1], items[b][1]))
        print(f"  {pid}: {np.mean(sims):.4f} (from {len(sims)} pairs)")

    results = {
        "layer": TARGET_LAYER,
        "within_product_mean": float(mean_within),
        "between_product_mean": float(mean_between),
        "gap": float(gap),
        "signal_present": bool(gap > 0),
        "within_pairs": {label: float(sim) for label, sim in within_sims},
        "between_pairs": {label: float(sim) for label, sim in between_sims},
    }

    results_path = SMOKE_DIR / "smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[smoke] Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Coherence smoke test")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction (use existing HDF5)")
    args = parser.parse_args()

    stimuli = load_stimuli()
    print(f"[smoke] Loaded {len(stimuli)} stimuli from {STIMULI_PATH}")

    h5_path = get_h5_path()

    if not args.skip_extract:
        h5_path = run_extraction(stimuli)
    else:
        if not h5_path.exists():
            print(f"[smoke] HDF5 not found at {h5_path}. Run without --skip-extract first.")
            sys.exit(1)
        print(f"[smoke] Using existing HDF5: {h5_path}")

    analyze(stimuli, h5_path)


if __name__ == "__main__":
    main()
