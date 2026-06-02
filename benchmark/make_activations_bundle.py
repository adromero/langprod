#!/usr/bin/env python3
"""Build the GPU-free reproduction bundle for RegProd-800.

Extracts ONLY the residual-stream hidden states from the full 3.5 GB experiment
HDF5, downcasts float32 -> float16, and writes a compact, self-describing HDF5:

    activations/hidden_states_residual_fp16.h5
        hidden_states_mean_no_special : (800, 65, 5120) float16, gzip
        stimulus_ids                  : (800,) UTF-8  (row alignment key)

This is enough to reproduce the RSA curves and linear probes from the writeup
WITHOUT loading the 32B model or a GPU. The dropped attention_output / mlp_output
arrays are not used by any published result. fp16 is adequate: cosine-distance
RSA is invariant to the ~1e-3 relative error fp16 introduces (verified by
reproduce.py, which lands within rounding of baselines.json).

Run:  python make_activations_bundle.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "data" / "Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4_hidden_states.h5"
OUT_DIR = HERE / "activations"
OUT = OUT_DIR / "hidden_states_residual_fp16.h5"

DSET = "hidden_states_mean_no_special"  # residual stream, (800, 65, 5120)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source activations not found: {SRC}")
    OUT_DIR.mkdir(exist_ok=True)

    with h5py.File(SRC, "r") as f:
        resid = f[DSET][:].astype(np.float16)
        sids = f["stimulus_ids"][:]
    sids = np.array([s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in sids])

    finite = np.isfinite(resid.astype(np.float32)).all()
    with h5py.File(OUT, "w") as out:
        out.create_dataset(
            DSET, data=resid, dtype="float16",
            compression="gzip", compression_opts=4, chunks=(64, resid.shape[1], resid.shape[2]),
        )
        out.create_dataset("stimulus_ids", data=sids.astype(h5py.string_dtype("utf-8")))
        out.attrs["model"] = "Qwen2.5-32B-Instruct-GPTQ-Int4"
        out.attrs["representation"] = "residual stream, mean-pooled over non-special tokens"
        out.attrs["dtype_note"] = "float16 downcast from the float32 extraction"
        out.attrs["layer_convention"] = "axis 1: position 0 = embedding, 1..64 = transformer layers"
        out.attrs["shape"] = resid.shape
        out.attrs["all_finite"] = bool(finite)

    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(HERE)}  shape={resid.shape} dtype=float16  all_finite={finite}  size={mb:.1f} MB")


if __name__ == "__main__":
    main()
