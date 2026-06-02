#!/usr/bin/env python3
"""GPU-free reproduction of the RegProd-800 headline results.

Uses ONLY the shipped fp16 residual activations + stimuli.jsonl. No GPU, no 32B
model. Recomputes the three RSA curves (and optionally a linear probe) and checks
them against baselines.json.

    python reproduce.py            # RSA for all 65 layers, compare to baselines
    python reproduce.py --probe    # also run the category linear probe (needs scikit-learn)

Requires: numpy, scipy  (+ scikit-learn only for --probe).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ACT = HERE / "activations" / "hidden_states_residual_fp16.h5"
DSET = "hidden_states_mean_no_special"


def load_aligned():
    """Return (activations [N,L,D] float32, meta list aligned to activation rows)."""
    with h5py.File(ACT, "r") as f:
        acts = f[DSET][:].astype(np.float32)
        sids = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in f["stimulus_ids"][:]]
    by_id = {json.loads(l)["stimulus_id"]: json.loads(l)
             for l in (HERE / "stimuli.jsonl").read_text().splitlines() if l.strip()}
    meta = [by_id[s] for s in sids]  # align metadata to the activation row order
    return acts, meta


def product_rdm(meta):
    n = len(meta)
    r = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if meta[i]["product_id"] == meta[j]["product_id"]:
                d = 0.0
            elif meta[i]["category"] == meta[j]["category"]:
                d = 0.5
            else:
                d = 1.0
            r[i, j] = r[j, i] = d
    return r


def register_rdm(meta):
    n = len(meta)
    r = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0 if meta[i]["register"] == meta[j]["register"] else 1.0
            r[i, j] = r[j, i] = d
    return r


def within_category_rdm(meta):
    n = len(meta)
    r = np.full((n, n), np.nan)
    for i in range(n):
        r[i, i] = 0.0
        for j in range(i + 1, n):
            if meta[i]["category"] == meta[j]["category"]:
                d = 0.0 if meta[i]["product_id"] == meta[j]["product_id"] else 1.0
                r[i, j] = r[j, i] = d
    return r


def rsa(observed_rdm, model_rdm):
    iu = np.triu_indices(observed_rdm.shape[0], k=1)
    obs, mod = observed_rdm[iu], model_rdm[iu]
    valid = ~np.isnan(mod)
    return float(spearmanr(obs[valid], mod[valid]).correlation)


def curve(acts, model_rdm):
    out = np.empty(acts.shape[1])
    for layer in range(acts.shape[1]):
        obs = squareform(pdist(acts[:, layer, :], metric="cosine"))
        out[layer] = rsa(obs, model_rdm)
    return out


def check(name, got_peak_layer, got_peak_r, base):
    exp_layer, exp_r = base["peak_layer"], base["peak_r"]
    dl, dr = abs(got_peak_layer - exp_layer), abs(got_peak_r - exp_r)
    ok = dl <= 1 and dr <= 0.01
    print(f"  {name:18s} peak L{got_peak_layer:>2d} r={got_peak_r:.4f}  "
          f"(baseline L{exp_layer} r={exp_r:.4f}; Δlayer={dl}, Δr={dr:.4f})  "
          f"{'PASS' if ok else 'CHECK'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="also run the category linear probe")
    args = ap.parse_args()

    base = json.loads((HERE / "baselines.json").read_text())
    acts, meta = load_aligned()
    print(f"loaded fp16 activations {acts.shape} aligned to {len(meta)} stimuli\n")

    print("RSA (recomputed from fp16 residual stream):")
    models = {
        "product_identity": product_rdm(meta),
        "register_identity": register_rdm(meta),
        "within_category": within_category_rdm(meta),
    }
    all_ok = True
    for name, mrdm in models.items():
        c = curve(acts, mrdm)
        pl = int(np.nanargmax(c))
        all_ok &= check(name, pl, round(float(np.nanmax(c)), 4), base["rsa"][name])

    if args.probe:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        layer = base["linear_probe_macro_f1"]["category_best_layer"]["layer"]
        y = np.array([m["category"] for m in meta])
        groups = np.array([m["product_id"] for m in meta])
        X = acts[:, layer, :]
        f1s = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), PCA(n_components=200, random_state=42),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            f1s.append(f1_score(y[te], clf.predict(X[te]), average="macro"))
        got = float(np.mean(f1s))
        exp = base["linear_probe_macro_f1"]["category_best_layer"]["macro_f1"]
        ok = abs(got - exp) <= 0.03
        print(f"\nLinear probe (category, layer {layer}, 5-fold GroupKFold, PCA-200):")
        print(f"  macro-F1 {got:.4f}  (baseline {exp:.4f}; Δ={abs(got-exp):.4f})  {'PASS' if ok else 'CHECK'}")
        all_ok &= ok

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS NEED REVIEW'} "
          f"(RSA tol: Δlayer≤1, Δr≤0.01)")


if __name__ == "__main__":
    main()
