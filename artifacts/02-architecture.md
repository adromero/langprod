# System Architecture: The Protocol Layer Hypothesis

## 1. Pipeline Overview

This research experiment is a **five-stage data pipeline** with well-defined artifacts at each boundary. Each stage can be re-run independently given its input artifacts, enabling iterative refinement without full re-computation.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Stage 1:       │     │  Stage 2:        │     │  Stage 3:            │
│  Stimulus       │────▶│  Hidden State    │────▶│  Similarity &        │
│  Generation     │     │  Extraction      │     │  Probe Analysis      │
│                 │     │                  │     │                      │
│  Out: JSON      │     │  Out: HDF5       │     │  Out: NPY + CSV      │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌──────────────────────┐
                        │  Stage 5:        │     │  Stage 4:            │
                        │  Reporting &     │◀────│  Statistical         │
                        │  Visualization   │     │  Testing             │
                        │                  │     │                      │
                        │  Out: PDF/PNG    │     │  Out: CSV + JSON     │
                        └──────────────────┘     └──────────────────────┘
```

### Stage Summary

| Stage | Name | Primary Input | Primary Output | Re-runnable Without | Estimated Time |
|-------|------|---------------|----------------|---------------------|----------------|
| 1 | Stimulus Generation | Product definitions (YAML) | `stimuli.json` (~800 descriptions) | Nothing (root stage) | 2-3 hours |
| 2 | Hidden State Extraction | `stimuli.json` + model weights | `hidden_states.h5` (per model) | Stage 1 | 4-8 hours/model |
| 3 | Similarity & Probe Analysis | `hidden_states.h5` | RSA matrices, probe scores (NPY/CSV) | Stages 1-2 |  1-2 hours |
| 4 | Statistical Testing | RSA matrices, probe scores | Test results (CSV/JSON) | Stages 1-3 | <30 min |
| 5 | Reporting & Visualization | All analysis outputs | Figures (PNG), tables (CSV), report | Stages 1-4 | <30 min |

---

## 2. Project Directory Structure

```
protocol-layer-hypothesis/
├── config/
│   ├── experiment.yaml          # Master config: models, layers, seeds, paths
│   ├── products.yaml            # 40 real products with core attributes
│   ├── fictional_products.yaml  # 40 fictional products with core attributes
│   └── registers.yaml           # 5 register definitions with generation prompts
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Config loading and validation (Pydantic)
│   ├── stimulus/
│   │   ├── __init__.py
│   │   ├── generate.py          # Stimulus generation orchestrator
│   │   ├── templates.py         # Per-register prompt templates
│   │   └── validate.py          # Length/semantic validation checks
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extract.py           # Hidden state extraction pipeline
│   │   ├── hooks.py             # Forward hook implementations
│   │   └── models.py            # Model loading (quantized + full-precision)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── rsa.py               # Representational Similarity Analysis
│   │   ├── probes.py            # Linear probe training (40-class, 8-class, 5-class)
│   │   ├── anisotropy.py        # Mean-centering + whitening corrections
│   │   └── decomposition.py     # Attention/MLP/residual decomposition
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── hypothesis_tests.py  # Falsification criteria evaluation
│   │   └── bootstrap.py         # Bootstrap confidence intervals
│   └── visualization/
│       ├── __init__.py
│       ├── phase_curves.py      # H1: per-layer similarity curves
│       ├── probe_curves.py      # H2/H3: per-layer probe accuracy
│       └── rsa_heatmaps.py      # RSA correlation heatmaps
│
├── scripts/
│   ├── run_stage1.py            # CLI entry: stimulus generation
│   ├── run_stage2.py            # CLI entry: hidden state extraction
│   ├── run_stage3.py            # CLI entry: analysis
│   ├── run_stage4.py            # CLI entry: statistical testing
│   ├── run_stage5.py            # CLI entry: visualization
│   └── run_all.py               # Full pipeline orchestrator
│
├── tests/
│   ├── conftest.py              # Shared fixtures (mock models, tiny hidden states)
│   ├── test_config.py
│   ├── test_stimulus_generation.py
│   ├── test_extraction.py
│   ├── test_rsa.py
│   ├── test_probes.py
│   ├── test_anisotropy.py
│   └── test_hypothesis_tests.py
│
├── data/                        # Git-ignored; generated artifacts live here
│   ├── stimuli/
│   │   ├── stimuli_real.json
│   │   ├── stimuli_fictional.json
│   │   └── stimuli_multisource_subset.json
│   ├── hidden_states/
│   │   ├── qwen3.5-27b-4bit/
│   │   │   ├── real.h5
│   │   │   ├── fictional.h5
│   │   │   └── metadata.json
│   │   ├── qwen3.5-27b-fp16/
│   │   │   └── real_subset.h5
│   │   └── llama-3.1-8b/
│   │       ├── real.h5
│   │       └── fictional.h5
│   ├── analysis/
│   │   ├── rsa/
│   │   ├── probes/
│   │   └── decomposition/
│   ├── stats/
│   └── figures/
│
├── notebooks/                   # Exploratory / one-off analyses
│   └── exploration.ipynb
│
├── pyproject.toml
├── .env.example                 # ANTHROPIC_API_KEY=sk-ant-your-key-here
├── .gitignore                   # data/, .env, *.h5, __pycache__
└── README.md
```

---

## 3. Stage 1: Stimulus Generation

### 3.1 Component Design

**Input**: `products.yaml` (40 real products + attributes), `fictional_products.yaml` (40 fictional), `registers.yaml` (5 register specs with prompt templates).

**Output**: Three JSON files:
- `stimuli_real.json`: 400 descriptions (40 products x 5 registers x 2 paraphrases)
- `stimuli_fictional.json`: 400 descriptions (40 fictional x 5 registers x 2 paraphrases)
- `stimuli_multisource_subset.json`: ~150 descriptions (10 products x 5 registers x 3 generators)

### 3.2 Stimulus Schema

```json
{
  "stimulus_id": "real_p001_r1_v1",
  "product_id": "p001",
  "product_name": "Crest Pro-Health Advanced",
  "category": "oral_care",
  "category_id": 0,
  "register": "marketing",
  "register_id": 0,
  "variant": 1,
  "is_fictional": false,
  "generator": "claude",
  "text": "...",
  "token_count": 112,
  "core_attributes": ["fluoride protection", "enamel strengthening", "cavity prevention"],
  "generation_metadata": {
    "model": "claude-sonnet-4-20250514",
    "timestamp": "2026-03-25T10:00:00Z",
    "prompt_hash": "sha256:abc123..."
  }
}
```

### 3.3 Generation Strategy

1. **Primary generation (Claude API)**: All 40 real + 40 fictional products across 5 registers with 2 paraphrase variants each. Total: 800 API calls (can be batched).
2. **Multi-source subset**: 10 products (at least 1 per category) fully crossed with 3 generators (Claude, GPT-4, human). The Claude variants already exist from step 1; GPT-4 variants require OpenAI API calls; human variants are written manually.
3. **Validation**: Automated checks for token count (80-150 target), semantic anchor verification (core attributes mentioned), and register characteristic verification.

### 3.4 Technology Choice

- **Anthropic API** (`anthropic` Python SDK) for Claude generation. Per the project instructions, this is a case where Claude is a *feature of the application* (stimulus generation), but since this is a one-time batch generation step (not a runtime feature), using the direct API with `ANTHROPIC_API_KEY` is appropriate. The Agent SDK's overhead is unnecessary for batch API calls.
- **OpenAI API** for GPT-4 generation of the multi-source subset.
- **Pydantic** for stimulus schema validation.

### 3.5 Cost Estimate

~800 stimuli at ~150 tokens output + ~500 tokens prompt each = ~520K tokens. At Claude Sonnet pricing (~$3/M input, $15/M output): ~$2-3 total. Negligible.

---

## 4. Stage 2: Hidden State Extraction

### 4.1 Component Design

This is the most resource-intensive and architecturally critical stage. It processes each stimulus through the model and captures intermediate representations.

**Input**: `stimuli_*.json` + model weights (downloaded from HuggingFace).

**Output**: HDF5 archives containing per-stimulus, per-layer hidden state vectors.

### 4.2 Why HDF5 Over Alternatives

| Format | Pros | Cons | Verdict |
|--------|------|------|---------|
| **NPY** (one file per stimulus per layer) | Simple, fast read | Thousands of small files; no metadata; filesystem overhead | Rejected |
| **NPY** (single large file) | Simple | Must load entire array into RAM; no partial reads | Rejected |
| **HDF5** | Chunked I/O; partial reads; metadata support; compression; single file per model | Slightly more complex API; h5py dependency | **Selected** |
| **Safetensors** | Fast, safe | Designed for model weights, not research data; no hierarchical structure | Rejected |
| **Zarr** | Cloud-friendly; chunked | Overkill for single-machine use | Rejected |

### 4.3 HDF5 Schema

```
hidden_states.h5
├── metadata (attrs)
│   ├── model_name: "Qwen/Qwen3.5-27B-AWQ"
│   ├── quantization: "4bit-awq"
│   ├── num_layers: 64
│   ├── hidden_dim: 3584
│   ├── extraction_date: "2026-03-25"
│   ├── extraction_seed: 42
│   └── extraction_config_hash: "sha256:..."
│
├── residual_stream/           # Primary: full hidden state at each layer
│   ├── layer_00/              # Shape: (N_stimuli, hidden_dim), float32
│   ├── layer_01/
│   ├── ...
│   └── layer_63/
│
├── attention_output/          # Decomposition: attention sublayer output
│   ├── layer_00/
│   ├── ...
│   └── layer_63/
│
├── mlp_output/                # Decomposition: MLP sublayer output
│   ├── layer_00/
│   ├── ...
│   └── layer_63/
│
├── stimulus_ids/              # String array mapping row index to stimulus_id
└── token_counts/              # Int array: actual token count per stimulus
```

### 4.4 Extraction Architecture: Forward Hooks vs. `output_hidden_states`

**Decision: Forward hooks** (not `output_hidden_states=True`).

Rationale:
1. **Decomposition requirement**: We need attention output, MLP output, and residual stream separately. `output_hidden_states` only returns the residual stream at each layer boundary.
2. **Memory control**: With hooks, we can process and store each layer's output immediately, then discard it from GPU memory. With `output_hidden_states`, all 64 layers are held in memory simultaneously.
3. **Quantization compatibility**: Forward hooks work identically regardless of quantization backend (GPTQ, AWQ, bitsandbytes). `output_hidden_states` behavior under quantization is implementation-dependent.
4. **CPU offloading compatibility**: Hooks fire on whatever device the layer runs on. This avoids the concern raised in the critique about `output_hidden_states` potentially missing layers split across devices.

### 4.5 Hook Implementation Strategy

```python
# Pseudocode for hook registration
class HiddenStateCollector:
    """Registers forward hooks to capture residual stream,
    attention output, and MLP output at each layer."""

    def __init__(self, model, layers_to_capture: list[int]):
        self.data = {}  # layer_idx -> {residual, attn, mlp}
        self.hooks = []

        for idx in layers_to_capture:
            layer = model.model.layers[idx]
            # Hook on the full layer for residual stream
            self.hooks.append(
                layer.register_forward_hook(self._make_residual_hook(idx))
            )
            # Hook on attention sublayer
            self.hooks.append(
                layer.self_attn.register_forward_hook(self._make_attn_hook(idx))
            )
            # Hook on MLP sublayer
            self.hooks.append(
                layer.mlp.register_forward_hook(self._make_mlp_hook(idx))
            )

    def _aggregate(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean pool across non-padding tokens."""
        # Exclude BOS/EOS, apply attention mask
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)
```

### 4.6 Processing Strategy: Batched Single-Stimulus

**Decision: Process stimuli one at a time (batch size 1), but batch the layer storage operations.**

Rationale:
- Stimuli have variable token lengths (80-150). Padding to max length wastes VRAM on the 27B model.
- With batch size 1, each forward pass uses minimal VRAM beyond the model itself.
- The bottleneck is model loading and forward pass compute, not data loading.
- Processing one stimulus at a time simplifies the mean-pooling logic (no padding mask complexity).

However, if profiling shows significant overhead from per-stimulus Python loop overhead, a fallback strategy is to group stimuli by similar token count (e.g., buckets of 80-100, 100-120, 120-150) and batch within each bucket with minimal padding.

### 4.7 VRAM Budget

| Model Config | Model Size in VRAM | KV Cache (seq_len=150, batch=1) | Hooks Overhead | Total VRAM | Fits in 32GB? |
|---|---|---|---|---|---|
| Qwen3.5-27B 4-bit AWQ | ~14 GB | ~0.2 GB | ~0.5 GB (one layer at a time) | **~15 GB** | Yes, comfortably |
| Qwen3.5-27B FP16 (device_map=auto) | ~27 GB GPU + ~27 GB CPU | ~0.2 GB | ~0.5 GB | **~28 GB GPU** (partial offload) | Marginal; needs ~22 GB CPU offload |
| Llama-3.1-8B FP16 | ~16 GB | ~0.1 GB | ~0.3 GB | **~16.5 GB** | Yes, comfortably |

**Critical note on Qwen FP16**: The 27B model at FP16 requires ~54 GB total. With 32 GB VRAM + 32 GB system RAM, the split is feasible but tight. HuggingFace's `device_map="auto"` will place roughly 60% of layers on GPU and 40% on CPU. Forward hooks will fire on whatever device each layer resides on; the hook must move the captured tensor to CPU immediately to avoid fragmentation. Estimated wall-clock slowdown: **5-10x** for the offloaded layers, making the full-precision run roughly 3-5x slower overall than the 4-bit run.

### 4.8 Disk Space Budget

Hidden state dimensions:
- Qwen3.5-27B: 64 layers, hidden_dim = 3584
- Llama-3.1-8B: 32 layers, hidden_dim = 4096

Per-stimulus, per-layer storage: `hidden_dim * 4 bytes` (float32).

| Dataset | Model | Stimuli | Layers | Components | Raw Size | With gzip (est.) |
|---|---|---|---|---|---|---|
| Real | Qwen 4-bit | 400 | 64 | 3 (residual+attn+mlp) | 400 * 64 * 3 * 3584 * 4 = **1.05 GB** | ~0.5 GB |
| Fictional | Qwen 4-bit | 400 | 64 | 3 | **1.05 GB** | ~0.5 GB |
| Real subset | Qwen FP16 | ~100 | 64 | 3 | **0.26 GB** | ~0.13 GB |
| Real | Llama 8B | 400 | 32 | 3 | 400 * 32 * 3 * 4096 * 4 = **0.59 GB** | ~0.3 GB |
| Fictional | Llama 8B | 400 | 32 | 3 | **0.59 GB** | ~0.3 GB |
| Multi-source subset | Qwen 4-bit | ~150 | 64 | 3 | **0.39 GB** | ~0.2 GB |
| **Total** | | | | | **~3.9 GB** | **~2 GB** |

This is very manageable. HDF5 with gzip compression (level 4) will bring total disk usage to approximately 2 GB.

### 4.9 Wall-Clock Time Estimates

Assuming ~0.5 seconds per stimulus for 27B 4-bit (forward pass + hook capture + HDF5 write):

| Run | Stimuli | Time per stimulus | Total |
|---|---|---|---|
| Qwen 4-bit, real | 400 | ~0.5s | ~3.5 min |
| Qwen 4-bit, fictional | 400 | ~0.5s | ~3.5 min |
| Qwen 4-bit, multi-source subset | 150 | ~0.5s | ~1.3 min |
| Qwen FP16, real subset | 100 | ~2.5s (CPU offload) | ~4.2 min |
| Llama 8B FP16, real | 400 | ~0.3s | ~2 min |
| Llama 8B FP16, fictional | 400 | ~0.3s | ~2 min |
| **Total extraction time** | | | **~17 min** |

Plus model loading time (~2-5 min per model), total Stage 2 is approximately **30-45 minutes**. This is much faster than the original estimate of 4-8 hours; the original estimate was likely based on longer sequences or older hardware. The RTX 5090's throughput at batch-size-1 inference on quantized 27B models is very fast for 150-token sequences.

**Caveat**: The Qwen FP16 run with CPU offloading could be significantly slower if the offloaded layers dominate compute. A more conservative estimate for this sub-run is 15-30 minutes. If it exceeds 1 hour, fall back to 8-bit quantization as an intermediate control (fits entirely in 32GB VRAM).

---

## 5. Stage 3: Similarity & Probe Analysis

### 5.1 Component Overview

This stage transforms raw hidden states into the quantitative measurements that test the three hypotheses.

**Input**: HDF5 hidden state archives.

**Output**:
- RSA correlation matrices (NPY)
- Probe accuracy scores per layer (CSV)
- Pairwise distance matrices per layer (NPY, for Stage 4)

### 5.2 Sub-pipeline 3a: Representational Similarity Analysis (Primary)

RSA is the **primary analytical method** (Decision #7). It operates on pairwise distances, giving 19,900+ data points from 200 stimuli (or 79,800+ from 400).

**Procedure at each layer**:
1. Load hidden states for all stimuli at layer L (shape: `N x hidden_dim`).
2. **Anisotropy path A (uncorrected)**: Compute pairwise cosine distances directly.
3. **Anisotropy path B (corrected)**: Mean-center the representations, apply ZCA whitening, then compute pairwise cosine distances.
4. Construct the **observed RDM** (Representational Dissimilarity Matrix): `N x N` symmetric matrix of pairwise distances.
5. Construct the **model RDM** encoding the hypothesis:
   - Same product, different register/variant: distance = 0
   - Same category, different product: distance = 1
   - Different category: distance = 2
6. Compute Spearman rank correlation between observed RDM (upper triangle) and model RDM (upper triangle).
7. Store the correlation coefficient and p-value.

**Output**: Per-layer RSA correlation curve (the core H1 evidence).

### 5.3 Sub-pipeline 3b: Linear Probes (Secondary)

Three probes trained at each layer:
- **40-class product probe** (primary secondary analysis)
- **8-class category probe** (topic-modeling null discriminator)
- **5-class register probe** (H2 test: content vs. format)

**Procedure at each layer**:
1. Load hidden states at layer L.
2. Apply both corrected and uncorrected representations.
3. Train L2-regularized logistic regression with 5-fold stratified CV.
   - Stratification: by product (ensures all variants of a product are in the same fold).
   - **Critical**: Stratify by product, NOT by register. If variants of the same product appear in both train and test, the probe might learn product identity from generation artifacts rather than semantics.
4. Record macro F1, accuracy, and per-class accuracy.

**Cross-validation note**: With 10 variants per product and 5 folds, each fold has 8 training examples and 2 test examples per product class. This is still a low-data regime but substantially better than the original 5-sample design. The 8-class and 5-class probes have 50 and 80 samples per class respectively, which is adequate.

### 5.4 Sub-pipeline 3c: Hidden State Decomposition

For each layer, repeat sub-pipelines 3a and 3b on:
- Attention output only
- MLP output only
- Full residual stream (already done in 3a/3b)

This reveals whether attention or MLP computations drive the protocol-layer effect.

### 5.5 Memory Budget for Analysis

The largest in-memory object is the pairwise distance matrix at a single layer: `400 * 400 * 4 bytes = 0.6 MB`. Even loading all 64 layers simultaneously: `64 * 0.6 MB = 38 MB`. Probe training on `400 x 3584` matrices is trivially small for scikit-learn. **No memory concerns for Stage 3.**

---

## 6. Stage 4: Statistical Testing

### 6.1 Hypothesis Evaluation

Each hypothesis has pre-registered falsification criteria. This stage evaluates them mechanically.

**H1 (Phase Structure)**:
- Test: Is RSA correlation significantly higher at middle layers (25th-75th percentile of stack) vs. early layers (0th-10th percentile)?
- Method: Paired t-test or Wilcoxon signed-rank test on per-layer RSA correlations.
- Falsification: p > 0.05.
- Run on both corrected and uncorrected representations.

**H2 (Content Dominance)**:
- Test: Does register-prediction accuracy exceed category-prediction accuracy by >5pp at any layer in the protocol zone?
- Method: Compare 5-class register probe accuracy vs. 8-class category probe accuracy (both evaluated at the same layers).
- Falsification: Register accuracy > category accuracy + 5pp in protocol zone.

**H3 (Protocol Layer Advantage)**:
- Test: Does the best-performing layer for the 40-class product probe fall in the middle 60% of the stack? Does it outperform the output layer by >= 2pp?
- Method: Identify peak accuracy layer; check its position; compare to output-layer accuracy.
- Falsification: Peak outside middle 60% OR advantage < 2pp.

### 6.2 Quantization Control

- Compute per-layer RSA correlation curves for both Qwen FP16 subset and Qwen 4-bit (same stimuli subset).
- Compute Spearman correlation between the two curves.
- **Pass criterion**: Spearman rho > 0.9 (pre-registered, Decision #12).

### 6.3 Memorization Control

- Run the full RSA + probe analysis on fictional products separately.
- Compare RSA phase-structure curves between real and fictional products.
- **Interpretation**: If fictional products show the same phase structure as real, memorization is not driving the effect.

### 6.4 Generator Control

- For the 10-product multi-source subset: compare RSA correlations and probe accuracies across generators (Claude vs. GPT-4 vs. human).
- Test for generator main effect using ANOVA on pairwise distances.
- **Decision point**: If generator effect is negligible (p > 0.05 and effect size < 0.1), proceed with Claude-only main dataset. If significant, report as caveat.

### 6.5 Bootstrap Confidence Intervals

For all key metrics (RSA correlations, probe accuracies), compute 95% bootstrap CIs (10,000 resamples) to quantify uncertainty.

---

## 7. Stage 5: Reporting & Visualization

### 7.1 Core Figures

1. **Phase Structure Plot (H1)**: RSA correlation (y-axis) vs. layer index (x-axis), with corrected and uncorrected curves. Three conditions: same-product-different-register, same-category-different-product, different-category. Shaded 95% CI bands.

2. **Probe Accuracy Curves (H2)**: Three lines (40-class product, 8-class category, 5-class register) across layers. Shaded protocol zone. Error bars from 5-fold CV.

3. **Zone Comparison Bar Chart (H3)**: Macro F1 for early/middle/late/output zones. Error bars from bootstrap CI.

4. **Decomposition Panel**: Same as figures 1-2 but separately for attention output, MLP output, residual stream.

5. **Quantization Control**: Overlaid RSA curves for FP16 vs. 4-bit. Annotated Spearman rho.

6. **Memorization Control**: Overlaid RSA curves for real vs. fictional products.

7. **RSA Heatmaps**: Full RDM at selected layers (early, middle, late) to visualize clustering structure.

### 7.2 Output Format

- Figures: PNG (300 DPI) for quick inspection, PDF for publication.
- Tables: CSV for machine readability, also rendered in the final report.
- Summary report: Markdown file with embedded figure references and hypothesis verdicts.

---

## 8. Configuration Management

### 8.1 Master Config (`experiment.yaml`)

```yaml
experiment:
  name: "protocol-layer-hypothesis"
  seed: 42
  version: "1.0"

paths:
  data_dir: "data/"
  stimuli_dir: "data/stimuli/"
  hidden_states_dir: "data/hidden_states/"
  analysis_dir: "data/analysis/"
  figures_dir: "data/figures/"

stimuli:
  target_token_range: [80, 150]
  variants_per_product: 2          # paraphrases per register
  registers: ["marketing", "regulatory", "casual", "patent", "journalistic"]
  categories: 8
  products_per_category: 5
  fictional_products_per_category: 5
  multisource_subset_size: 10      # products fully crossed with all generators
  generators: ["claude", "gpt4"]   # human variants added manually

models:
  primary:
    name: "Qwen/Qwen3.5-27B-AWQ"
    quantization: "4bit-awq"
    device_map: "auto"
    torch_dtype: "float16"
  quantization_control:
    name: "Qwen/Qwen3.5-27B"
    quantization: null
    device_map: "auto"            # CPU offloading
    torch_dtype: "float16"
    stimulus_subset: "quantization_control"  # defined in stimuli config
  validation:
    name: "meta-llama/Llama-3.1-8B-Instruct"
    quantization: null
    device_map: "auto"
    torch_dtype: "float16"

extraction:
  pooling: "mean"                  # mean pooling across non-special tokens
  components: ["residual", "attention", "mlp"]
  batch_size: 1
  save_format: "hdf5"
  compression: "gzip"
  compression_level: 4

analysis:
  rsa:
    distance_metric: "cosine"
    model_rdm_levels: [0, 1, 2]   # same-product, same-category, different-category
    correlation_method: "spearman"
  probes:
    classifier: "logistic_regression"
    regularization: "l2"
    cv_folds: 5
    stratify_by: "product"
    metrics: ["accuracy", "macro_f1"]
  anisotropy:
    corrections: ["none", "mean_center_whiten"]
  bootstrap:
    n_resamples: 10000
    ci_level: 0.95

falsification:
  h1_alpha: 0.05
  h2_register_dominance_threshold_pp: 5
  h3_protocol_zone_pct: [20, 80]  # middle 60%
  h3_advantage_threshold_pp: 2
  quantization_spearman_threshold: 0.9
```

### 8.2 Config Validation

Use **Pydantic** to define typed config models. The config is validated at pipeline startup; any missing or invalid field halts execution with a clear error message. This prevents silent misconfigurations from invalidating long extraction runs.

### 8.3 Reproducibility Guarantees

1. **Random seeds**: All random operations (probe CV splits, bootstrap resampling) use seeds derived from the master seed via `numpy.random.SeedSequence`.
2. **Config hashing**: The full config is SHA-256 hashed and stored in every output artifact's metadata. If you re-run with different config, the hash changes, preventing accidental result mixing.
3. **Git tracking**: The config files and all source code are version-controlled. The extraction metadata records the git commit hash at extraction time.
4. **Deterministic model loading**: Pin exact HuggingFace model revision hashes in the config.
5. **Dependency pinning**: `pyproject.toml` with exact version pins for torch, transformers, scikit-learn, numpy, scipy, h5py, matplotlib.

---

## 9. Technology Stack

### 9.1 Core Dependencies

| Package | Purpose | Version Constraint |
|---|---|---|
| `torch` | GPU compute, model inference | >=2.5 (5090 support) |
| `transformers` | Model loading, tokenization | >=4.48 (Qwen3.5 support) |
| `auto-gptq` or `autoawq` | 4-bit quantization support | Latest compatible |
| `h5py` | HDF5 read/write for hidden states | >=3.10 |
| `numpy` | Numerical operations | >=1.26 |
| `scipy` | Statistical tests, spatial distances | >=1.12 |
| `scikit-learn` | Linear probes, cross-validation | >=1.4 |
| `matplotlib` | Visualization | >=3.8 |
| `pydantic` | Config validation | >=2.5 |
| `anthropic` | Claude API for stimulus generation | >=0.40 |
| `openai` | GPT-4 API for multi-source subset | >=1.50 |
| `pyyaml` | Config file loading | >=6.0 |
| `tqdm` | Progress bars | >=4.66 |

### 9.2 Development Dependencies

| Package | Purpose |
|---|---|
| `pytest` | Test framework |
| `pytest-cov` | Coverage reporting |
| `ruff` | Linting and formatting |

### 9.3 Python Version

Python 3.11+ (for `tomllib` support and performance improvements in NumPy/SciPy).

---

## 10. Key Design Trade-offs

### 10.1 Forward Hooks vs. `output_hidden_states`

**Chosen: Forward hooks.**

Forward hooks add implementation complexity (must understand Qwen and Llama layer architectures to register hooks on the correct modules) but provide:
- Component decomposition (attention/MLP/residual) at no extra forward pass cost
- Memory control (capture and offload one layer at a time)
- Robustness across quantization backends and device maps

The alternative (`output_hidden_states=True`) is simpler but only provides the residual stream, holds all layers in memory simultaneously (dangerous at 64 layers x 3584 dims x batch_size), and has uncertain behavior under CPU offloading.

**Risk mitigation**: Build a `test_extraction.py` that loads a tiny model (e.g., a 2-layer random transformer) and verifies hook outputs match `output_hidden_states` outputs. This catches hook registration bugs immediately.

### 10.2 HDF5 vs. NPY Archives

**Chosen: HDF5 with gzip compression.**

HDF5 adds a dependency (`h5py`) and slightly more complex I/O code, but provides:
- Single file per model run (vs. thousands of NPY files)
- Chunked reads (load one layer at a time without reading the whole file)
- Metadata storage (model config, extraction parameters, stimulus IDs)
- Built-in compression (gzip level 4 gives ~2x compression with minimal speed impact)

NPY would be simpler for small experiments but becomes unwieldy at the scale of 64 layers x 3 components x 800 stimuli = 153,600 individual files.

### 10.3 Monolithic Script vs. Modular Pipeline

**Chosen: Modular pipeline with stage scripts.**

Each stage is a standalone script that reads artifacts from the previous stage. This enables:
- Re-running analysis without re-extracting hidden states (the expensive step)
- Parallel development (stimulus generation and extraction code can be developed independently)
- Debugging (inspect intermediate artifacts)
- Incremental execution (run Qwen first, analyze, then add Llama)

The cost is more boilerplate (argument parsing, config loading in each script) and the need to manage artifact paths. Pydantic config management and consistent path conventions minimize this cost.

### 10.4 Batch Size 1 vs. Dynamic Batching

**Chosen: Batch size 1 (with optional bucketed batching fallback).**

For a 27B model processing 150-token sequences, the forward pass dominates wall-clock time regardless of batch size. Python loop overhead per stimulus is ~1ms vs. ~500ms for the forward pass. The simplicity of batch-size-1 processing (no padding, no mask handling in hooks) outweighs the ~0.2% throughput gain from batching.

If profiling reveals otherwise (e.g., on the smaller Llama 8B model where forward passes are faster), the extraction code should support a `batch_size` config parameter with bucketed batching as a fallback.

### 10.5 Mean Pooling vs. Last-Token Representation

**Chosen: Mean pooling (primary), with last-token as optional secondary.**

Mean pooling across non-special tokens is standard for sentence-level representations in probing studies. It averages out position-specific information and produces a single vector per stimulus.

Last-token representation (used in causal LMs for generation) carries more information about the "conclusion" of the sequence but is sensitive to sequence length and end-of-sequence tokens.

The config supports both. Mean pooling is the primary analysis; last-token can be enabled for robustness checks.

---

## 11. Error Handling and Resilience

### 11.1 Checkpointing in Extraction

The HDF5 format supports incremental writes. If extraction crashes mid-run (OOM, power failure), completed stimuli are already persisted. The extraction script should:
1. Check which stimulus IDs already exist in the HDF5 file.
2. Skip already-extracted stimuli.
3. Resume from the first missing stimulus.

This makes the extraction stage idempotent and crash-resilient.

### 11.2 VRAM OOM Handling

If a forward pass triggers CUDA OOM:
1. Log the stimulus ID and token count.
2. Clear CUDA cache (`torch.cuda.empty_cache()`).
3. Skip that stimulus and continue.
4. Report skipped stimuli in the extraction metadata.

This is preferable to crashing the entire multi-hour extraction run for one outlier stimulus.

### 11.3 API Rate Limiting (Stage 1)

Stimulus generation should use exponential backoff with jitter for API rate limits. The `anthropic` SDK handles this natively with `max_retries`. Set `max_retries=5` with a 60-second max backoff.

---

## 12. Testing Strategy

### 12.1 Unit Tests (No GPU, No API Keys)

All tests run without GPU, API keys, or network access per project requirements.

| Test | What It Validates |
|---|---|
| `test_config.py` | Config loading, validation, defaults, error messages |
| `test_stimulus_generation.py` | Prompt template rendering, schema validation, token count checks (mock API calls) |
| `test_extraction.py` | Hook registration on a tiny random model, mean pooling correctness, HDF5 write/read roundtrip |
| `test_rsa.py` | RSA computation on synthetic distance matrices with known structure |
| `test_probes.py` | Probe training/evaluation on synthetic data with known class boundaries |
| `test_anisotropy.py` | Mean-centering correctness, whitening invertibility, effect on known anisotropic data |
| `test_hypothesis_tests.py` | Falsification logic on synthetic results (known pass/fail cases) |

### 12.2 Mock Strategy

- **Model mocks**: Use `torch.nn.TransformerDecoderLayer` (2 layers, hidden_dim=64) instead of real 27B models. Verify hooks capture correct shapes and components.
- **API mocks**: Use `unittest.mock.patch` on `anthropic.Anthropic.messages.create` to return canned stimulus text.
- **HDF5 mocks**: Use `tempfile.TemporaryDirectory` for all file I/O tests; clean up automatically.

### 12.3 Integration Tests (Gated)

Gated behind `RUN_INTEGRATION_TESTS=1`:
- Load actual Qwen 4-bit model, extract hidden states for 5 stimuli, verify shapes and value ranges.
- Call Claude API with one stimulus generation prompt, verify response format.

---

## 13. Execution Plan

### 13.1 Development Order

1. **Config system** (`config.py`, `experiment.yaml`, `test_config.py`) — Foundation for everything.
2. **Stimulus generation** (`stimulus/`, `test_stimulus_generation.py`) — Can run independently.
3. **Hidden state extraction** (`extraction/`, `test_extraction.py`) — Depends on stimulus format.
4. **RSA analysis** (`analysis/rsa.py`, `analysis/anisotropy.py`, `test_rsa.py`, `test_anisotropy.py`) — Depends on HDF5 schema.
5. **Probes** (`analysis/probes.py`, `test_probes.py`) — Depends on HDF5 schema.
6. **Statistical testing** (`stats/`, `test_hypothesis_tests.py`) — Depends on analysis output format.
7. **Visualization** (`visualization/`) — Last; consumes all analysis outputs.
8. **Pipeline orchestrator** (`run_all.py`) — Wires everything together.

### 13.2 Estimated Development Time

| Component | Estimated Hours |
|---|---|
| Config system + project scaffolding | 2 |
| Stimulus generation (templates + API calls + validation) | 4 |
| Hidden state extraction (hooks + HDF5 + checkpointing) | 6 |
| RSA analysis + anisotropy correction | 3 |
| Linear probes | 2 |
| Statistical testing + falsification logic | 2 |
| Visualization | 3 |
| Tests | 4 (concurrent with above) |
| Integration, debugging, iteration | 4 |
| **Total development** | **~30 hours** |

### 13.3 Estimated Execution Time (Once Built)

| Stage | Time |
|---|---|
| Stimulus generation (API calls) | 2-3 hours (mostly API latency) |
| Hidden state extraction (all models) | 45 min - 2 hours |
| Analysis (RSA + probes + decomposition) | 1-2 hours |
| Statistical testing | <30 min |
| Visualization | <30 min |
| **Total execution** | **~5-8 hours** |

---

## 14. Risk Registry

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen FP16 CPU offload fails or is too slow | Medium | Medium | Fall back to 8-bit quantization as intermediate control |
| Hook implementation differs between Qwen and Llama architectures | Medium | Low | Test hook registration on both model families; abstract model-specific layer access |
| 40-class probe still underpowered with 10 variants/product | Medium | Medium | RSA is primary (19,900+ pairwise distances); probes are secondary confirmation |
| HDF5 corruption on extraction crash | Low | High | Write each stimulus atomically; verify checksums on read |
| Anisotropy correction destroys signal | Low | High | Report both corrected and uncorrected; interpret divergence |
| VRAM fragmentation from hooks | Low | Medium | Process hooks to CPU immediately; call empty_cache periodically |
| Generator effect confounds multi-source subset | Medium | Medium | Fully crossed design on subset; explicit statistical test for generator effect |

---

## 15. Open Questions for Implementation

1. **Qwen3.5-27B layer architecture**: The hook implementation needs to know the exact module names (`model.layers[i].self_attn`, `model.layers[i].mlp`). These need to be verified against the actual model architecture at implementation time, as HuggingFace model implementations vary.

2. **AWQ vs. GPTQ**: Both are 4-bit quantization methods. AWQ is generally faster for inference; GPTQ has broader compatibility. The choice depends on which has better HuggingFace integration for Qwen3.5-27B at implementation time. Check `autoawq` and `auto-gptq` compatibility.

3. **Whitening implementation**: ZCA whitening (`W = S^{-1/2}` where `S` is the covariance matrix) can be numerically unstable when eigenvalues are near zero. Regularize with `S + epsilon * I` where `epsilon = 1e-5`. This is a known issue in anisotropy correction literature.

4. **Stratified CV for 40-class probe**: With 10 variants per product, 5-fold CV puts 2 variants in each test fold. Ensure no product-level leakage: all variants of one product must be in the same fold. Use `sklearn.model_selection.GroupKFold` with `groups=product_id`.

5. **Model revision pinning**: At implementation time, pin specific HuggingFace model revision hashes (not just model names) in the config to ensure exact reproducibility.
