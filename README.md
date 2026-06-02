# langprod — The Protocol Layer Hypothesis

An experimental test of whether transformer **middle layers** encode a *format-agnostic
semantic substrate* — a "protocol layer" where the same product, described in different
linguistic registers, converges in representational geometry.

**Model:** Qwen2.5-32B-Instruct-GPTQ-Int4 (64 layers, hidden dim 5120) ·
**Stimuli:** 800 product descriptions (80 products × 5 registers × 2 variants).

### Headline result: hypothesis **not supported**

Register identity — *how* a description is written — was the dominant organizing
principle of representational geometry at **every** layer (register RSA peak **r = 0.670**
vs. product RSA peak **r = 0.371**). Fine-grained product identity emerged only in **late**
layers (peak at layer 61/64), not the predicted middle. There is no protocol layer for
within-language register variation; form dominates content end-to-end.

## Start here

| If you want to… | Go to |
|---|---|
| **Read the paper** | [`writeup.pdf`](writeup.pdf) (source: [`writeup.md`](writeup.md)) |
| **Use the dataset / benchmark** | [**`benchmark/`**](benchmark/) → [`benchmark/README.md`](benchmark/README.md) |
| **Reproduce results without a GPU** | [`benchmark/reproduce.py`](benchmark/reproduce.py) (CPU-only, ~80 s) |
| **Re-run the full pipeline** | [`run.py`](run.py) (needs the model + a ~24 GB GPU) |

## The benchmark: RegProd-800

The shippable artifact is **[RegProd-800](benchmark/)** — 800 labeled product descriptions
built to dissociate semantic *content* from surface *register*, plus three classification
tasks (register-5, category-8, product-80), reference baselines, and an optional **GPU-free
reproduction bundle** (fp16 residual-stream activations, ~496 MB).

```bash
cd benchmark
python load.py          # sanity-check the loader (800 stimuli, 3 tasks)
python reproduce.py     # recompute the RSA curves from fp16 activations, vs baselines
```

`reproduce.py` regenerates the writeup's headline numbers on CPU to four decimals — see
[`benchmark/README.md`](benchmark/README.md) for the full dataset card, schema, task
protocol, and known caveats (BoW ceiling, GroupKFold zero-shot quirk, anisotropy).

## Repository layout

```
writeup.md / writeup.pdf   The paper (methods, results, hypothesis evaluation)
plan.md                    Pre-registered plan (hypotheses + falsification criteria)

run.py                     Pipeline entry point (subcommands below)
stimuli.py                 Product catalogs, register specs, prompt templates, quality gates
generate_stimuli.py        Stimulus generation runner (via the `claude` CLI)
extraction.py              Hidden-state extraction from the transformer
analysis.py                RDMs, RSA, model RDMs, permutation testing
probes.py                  Linear probing classifiers (all-layer / control / zone)
viz.py                     Figure generation
md_to_pdf.py               Render writeup.md -> writeup.pdf

benchmark/                 ► Shippable RegProd-800 dataset, tasks, baselines, GPU-free repro
coherence/                 Separate "brand-coherence-validation" sub-experiment (not the paper)
data/                      Generated outputs — gitignored (see "Data" below)
tests/                     Test suite
plans/, artifacts/         Planning & design records
```

## Reproduce from scratch

Pipeline stages (each reads/writes under `data/`):

```bash
python run.py generate     # generate 800 stimuli + BoW baseline + register checks
python run.py extract      # extract hidden states  (needs model + GPU; --pilot for a 5-stimulus dry run)
python run.py analyze      # RDMs, RSA, permutation tests
python run.py probe        # all-layer, control, and zone probes
python run.py report       # hypothesis tests, go/no-go, figures
python run.py all          # everything, end-to-end
```

## Data

`data/` is **gitignored** and not shipped. Its largest file is the raw extraction
`data/Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4_hidden_states.h5` (3.5 GB: mean-pooled
residual/attention/MLP activations for all 65 layer positions). You don't need it to use
the benchmark — `benchmark/activations/` ships an fp16 residual-stream subset (~496 MB)
that reproduces every published result on CPU. To regenerate the full file, run
`python run.py extract` with the model available.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime deps; add [dev] for pytest/ruff
```

Built and run on Ubuntu 24.04 (WSL2), Python 3.12, RTX 5090 (32 GB). Extraction requires
GPTQ support (`pip install -e '.[gptq-fallback]'`); the benchmark's `reproduce.py` needs
only `numpy`/`scipy` (+ `scikit-learn` for the probe) and no GPU.

## Citation

> A. Romero (2026). *The Protocol Layer Hypothesis: An Experimental Test of
> Register-Invariant Semantic Representations in Transformer Middle Layers.*
