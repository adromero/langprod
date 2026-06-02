#!/usr/bin/env python3
"""Minimal, dependency-free loader for the RegProd-800 benchmark.

    from load import load_stimuli, load_tasks, load_baselines, label_vector

    rows = load_stimuli()                 # list[dict], 800 entries
    y    = label_vector(rows, "register") # list[str] aligned to rows

No third-party packages required (stdlib json only). Pandas/sklearn are optional
and only needed if you want to train probes yourself.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_stimuli(path: str | Path | None = None) -> list[dict]:
    """Load the 800 stimuli from stimuli.jsonl as a list of dicts."""
    p = Path(path) if path else HERE / "stimuli.jsonl"
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks() -> dict:
    return json.loads((HERE / "tasks.json").read_text())


def load_baselines() -> dict:
    return json.loads((HERE / "baselines.json").read_text())


def load_activations(as_float32: bool = True):
    """Load the GPU-free reproduction bundle (optional, ~0.5 GB; requires h5py + numpy).

    Returns (acts, stimulus_ids):
      acts          ndarray [800, 65, 5120], residual stream mean-pooled per layer
                    (axis 1: position 0 = embedding, 1..64 = transformer layers)
      stimulus_ids  list[str] aligned to axis 0 of `acts`; join to load_stimuli()
                    by 'stimulus_id'. Raises FileNotFoundError if the bundle was
                    not built (run make_activations_bundle.py first).
    """
    import h5py  # local import so the lightweight core has no heavy deps
    import numpy as np

    path = HERE / "activations" / "hidden_states_residual_fp16.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with: python make_activations_bundle.py"
        )
    with h5py.File(path, "r") as f:
        acts = f["hidden_states_mean_no_special"][:]
        if as_float32:
            acts = acts.astype(np.float32)
        sids = [s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
                for s in f["stimulus_ids"][:]]
    return acts, sids


def label_vector(rows: list[dict], task: str) -> list:
    """Return the label for `task` ('register'|'category'|'product') aligned to rows."""
    field = {"register": "register", "category": "category", "product": "product_id"}[task]
    return [r[field] for r in rows]


if __name__ == "__main__":
    rows = load_stimuli()
    tasks = load_tasks()
    print(f"loaded {len(rows)} stimuli")
    for t in ("register", "category", "product"):
        n = len(set(label_vector(rows, t)))
        print(f"  task {t!r}: {n} classes")
    print(f"first stimulus_id: {rows[0]['stimulus_id']}")
