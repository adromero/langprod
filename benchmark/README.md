# RegProd-800 — A Register × Content Probing Benchmark

**RegProd-800** is a controlled set of **800 product descriptions** built to dissociate
*what* a text is about (semantic content) from *how* it is written (linguistic register).
It ships alongside the writeup *"The Protocol Layer Hypothesis: An Experimental Test of
Register-Invariant Semantic Representations in Transformer Middle Layers"* (`../writeup.pdf`)
and is the dataset that experiment was run on.

The design crosses three factors fully:

```
80 products  ×  5 registers  ×  2 paraphrase variants  =  800 descriptions
```

- **80 products**, 10 in each of **8 categories** (oral care, pet food, home cleaning,
  sports nutrition, baby care, coffee/beverage, skincare, smart home).
- Within each category, **5 real** products (e.g. *Colgate Total*) and **5 fictional**
  products (e.g. *AeroMint ProShield*) with invented names and novel feature combinations.
  The fictional half is a **memorization control**: any representational clustering for
  fictional products cannot be attributed to pretraining associations.
- **5 registers**: `marketing`, `regulatory`, `casual_social`, `patent`, `journalistic`.
- **2 paraphrase variants** per (product × register) cell.

Each description is 80–158 words (mean 119.7) and is constrained to convey the same
quantitative core attributes for its product (e.g. `fluoride_ppm: 1450`), so the *content*
is held constant while the *register* varies.

## Files

| File | What it is |
|---|---|
| `stimuli.jsonl` | The 800 labeled descriptions, one JSON object per line. **The dataset.** |
| `tasks.json` | The three classification tasks + recommended probing/CV protocol + RSA model RDMs. |
| `baselines.json` | Reference scores (RSA, linear-probe F1, BoW ceiling, condition similarities) read directly from the experiment's result files. |
| `load.py` | Stdlib loader (`load_stimuli`, `load_tasks`, `load_baselines`, `label_vector`); `load_activations` adds optional h5py/numpy. |
| `build_benchmark.py` | Regenerates `stimuli.jsonl` + `baselines.json` from `../data/` (provenance / reproducibility). |
| `activations/hidden_states_residual_fp16.h5` | **GPU-free reproduction bundle** (~496 MB). fp16 residual-stream activations, `(800, 65, 5120)`, + aligned `stimulus_ids`. |
| `make_activations_bundle.py` | Rebuilds the fp16 bundle from the full `../data/…hidden_states.h5`. |
| `reproduce.py` | Recomputes the RSA curves (and optional category probe) from the fp16 bundle and checks them against `baselines.json`. |

### GPU-free reproduction

The `activations/` bundle lets anyone reproduce the writeup's headline results **without a
GPU or the 32B model** — just `numpy` + `scipy` (+ `scikit-learn` for the probe):

```bash
python reproduce.py          # 3 RSA curves vs baselines    (~80 s, CPU)
python reproduce.py --probe  # + category linear probe       (~90 s, CPU)
```

Verified output (recomputed from the shipped fp16 file):

```
product_identity   peak L61 r=0.3707  (baseline L61 r=0.3707; Δr=0.0000)  PASS
register_identity  peak L47 r=0.6704  (baseline L47 r=0.6704; Δr=0.0000)  PASS
within_category    peak L61 r=0.1975  (baseline L61 r=0.1975; Δr=0.0000)  PASS
category probe (L33, GroupKFold, PCA-200)  macro-F1 0.9924  (baseline 0.9924)  PASS
```

fp16 reproduces the float32 RSA to four decimals — cosine-distance geometry is
insensitive to the downcast.

> **What the bundle drops vs. the source:** only the residual stream is shipped.
> The full `../data/…hidden_states.h5` (3.5 GB) additionally holds per-layer
> `attention_output` and `mlp_output` arrays, which no published result uses. To
> re-extract everything from scratch you need the model + a ~24 GB GPU; the
> extraction code is in `../extraction.py`.

## Schema (`stimuli.jsonl`)

| Field | Type | Description |
|---|---|---|
| `stimulus_id` | str | Unique id, e.g. `oral_care_001_marketing_v0`. |
| `product_id` | str | 80-class label, e.g. `oral_care_001`. |
| `category` | str | 8-class label. |
| `register` | str | 5-class label. |
| `variant` | int | Paraphrase index (0 or 1). |
| `is_fictional` | bool | True for the 400 fictional-product stimuli. |
| `text` | str | The product description (the model input). |
| `token_count` | int | Approx. word/token count (80–158). |
| `core_attributes` | obj | The quantitative attributes the text must convey. |
| `generator` | str | Always `claude`. |
| `generated_at` | str | ISO 8601 timestamp. |

## Tasks

Three classification tasks (full spec + protocol in `tasks.json`):

| Task | Classes | Axis | Reference behavior |
|---|---|---|---|
| `register` | 5 | surface form | Linear probe F1 = **1.000** at every transformer layer (**0.996** already at the embedding layer). |
| `category` | 8 | coarse content | Saturates early — F1 ≈ **0.99 by layer 20**, best layer 33 (0.992). |
| `product` | 80 | fine content | See the GroupKFold note below. |

**RSA** (representational similarity analysis) is the second evaluation mode: correlate the
observed cosine RDM at each layer against the `product_identity`, `register_identity`, and
`within_category` model RDMs defined in `tasks.json`.

## Quickstart

```python
from load import load_stimuli, label_vector, load_baselines

rows = load_stimuli()                       # 800 dicts
texts = [r["text"] for r in rows]
y_reg = label_vector(rows, "register")      # surface-form labels
y_cat = label_vector(rows, "category")      # coarse-content labels

print(load_baselines()["rsa"]["product_identity"])
# {'peak_layer': 61, 'peak_r': 0.3707, 'zone_means': {...}}
```

To embed with any model and reproduce the probes, mean-pool the residual stream over
non-special tokens, PCA-200, then 5-fold GroupKFold by `product_id` (see
`tasks.json → recommended_protocol`).

## Reference results (Qwen2.5-32B, full numbers in `baselines.json`)

- **Register dominates content geometrically at every layer.** Register-identity RSA
  (peak r = **0.670** @ layer 47) is ~2× product-identity RSA (peak r = **0.371** @ layer 61).
- **Fine product identity is a late-layer phenomenon** (peaks at layer 61/64), not a
  middle-layer one — the writeup's headline negative result.
- **Bag-of-words ceiling = 100%** on all three tasks: each product's unique numeric
  attributes make the tasks lexically separable. Neural results must be read against this.

## Known design caveats (read before using)

1. **GroupKFold makes the 80-class `product` task zero-shot.** Holding out whole products
   means test classes are unseen, so probe F1 = 0.000 at every layer *by construction* — it
   measures cross-product generalization, not presence of product identity. For a
   within-product test, re-split with StratifiedKFold over `(product_id, variant)`.
2. **No anisotropy correction** was applied to the reference RSA. Middle layers are highly
   anisotropic; PCA-whitening or mean-centering may change the content/form balance.
3. **BoW ceiling is 100%.** This is a deliberately *easy* lexical separation; the benchmark
   tests representational *geometry*, not classification difficulty.

## Provenance & licensing

- **Generation:** all 800 texts were synthetically generated with the Claude CLI (`claude -p`)
  using register-specific prompt templates preserved in `../stimuli.py`. Random seed 42.
- **Real brand names appear** in the 400 real-product texts (e.g. *Colgate Total*,
  *Blue Buffalo*). These are third-party trademarks used here for research; **confirm
  redistribution terms before publishing the dataset publicly.** The 400 fictional-product
  texts have no such constraint.
- **License: TBD — not yet set.** The intended default is CC BY 4.0 for the labels/structure,
  but the real-brand texts above need a redistribution decision first. Set this before release.

## Citation

> A. Romero (2026). *The Protocol Layer Hypothesis: An Experimental Test of
> Register-Invariant Semantic Representations in Transformer Middle Layers.*
> Dataset: RegProd-800.
