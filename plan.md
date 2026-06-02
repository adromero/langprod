# Protocol Layer Hypothesis: Experiment Plan

## Summary

This experiment tests the **Protocol Layer Hypothesis**: that transformer middle layers encode a format-agnostic semantic substrate useful for product classification across text registers. The claim is that middle layers learn an approximate "protocol-level" representation stripped of surface-form serialization -- meaning the same product described as marketing copy, a patent claim, a regulatory filing, a social media post, or a news article should converge in representational geometry at middle layers, while diverging at early (encoding) and late (decoding) layers.

The experiment tests three nested hypotheses: **H1 (Phase Structure)** -- same-product representations converge in middle layers regardless of register; **H2 (Content Dominance)** -- semantic content (product/category identity) is more geometrically prominent than surface register in middle layers; **H3 (Protocol Layer Advantage)** -- middle-layer representations outperform early, late, and output layers for product classification. Primary analysis uses Representational Similarity Analysis (RSA) on pairwise distances across ~800 stimuli (40 real + 40 fictional products x 5 registers x 2 variants). Secondary analysis uses 40-class product, 8-class category, and 5-class register linear probes. Models: Qwen2.5-32B (4-bit GPTQ, `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`) as primary, Llama-3.1-8B as exploratory cross-model comparison only (not confirmatory).

The design includes four experimental controls -- fictional products (memorization control), multi-source generation (generator confound control), full-precision subset (quantization control, Tier 4 deferrable), and anisotropy correction (measurement artifact control) -- plus methodological safeguards: control tasks for probes (Hewitt & Manning 2019), partial RSA for confound control, within-category product discrimination, effect-size criteria alongside significance tests, permutation-based statistical testing, and a BoW baseline leakage check on generated stimuli.

## Problem Statement

Current evidence from RYS layer-duplication experiments (Ng 2026), cross-lingual alignment (Liu & Niehues, ACL 2025), and bell-shaped anisotropy profiles (Tyshchuk et al. 2024) suggests that transformer middle layers learn representations that are more abstract and less tied to surface form than early or late layers. However, this has not been systematically tested for within-language register variation using controlled stimulus sets with known semantic equivalence. This experiment provides that test using product descriptions -- a domain where the same factual content naturally appears across multiple text registers, enabling controlled manipulation of surface form while holding semantic content constant.

The experiment must discriminate the "protocol layer" interpretation (middle layers encode a format-agnostic semantic substrate) from two alternative explanations: (1) **topic modeling** -- middle layers merely encode coarse category membership (oral care, pet food, etc.) which is trivially register-invariant; and (2) **lexical abstraction** -- middle layers map synonym sets to similar representations without any deeper "protocol" structure. The 40-class product probe (vs. 8-class category probe) and within-category discrimination index in RSA are the primary tools for this discrimination.

## Approach

A five-stage pipeline processes stimuli through transformer models and analyzes the representational geometry of hidden states:

1. **Stimulus Generation**: Create ~800 product descriptions (40 real + 40 fictional products x 5 registers x 2 paraphrase variants) using Claude as primary generator, with a 10-product subset crossed with GPT-4 for generator confound testing. After generation, run a BoW baseline (TF-IDF + logistic regression) as a surface-feature leakage check and a quantitative register distinctiveness check (TF-IDF distance between registers, soft gate with warning threshold).
2. **Hidden State Extraction**: Run each stimulus through Qwen2.5-32B 4-bit GPTQ (primary) and Llama-3.1-8B (exploratory), capturing per-layer residual stream, attention output, and MLP output via forward hooks. Mean-pool across tokens (excluding special tokens) to produce one vector per stimulus per layer. Include per-stimulus NaN/Inf check during extraction.
3. **Similarity Analysis**: Compute Representational Dissimilarity Matrices (RDMs) at each layer, correlate with theoretical model RDMs via RSA (primary), and compute per-layer cosine similarity by condition (same-product-different-register, same-category-different-product, different-category). Use tiered permutation testing: 200-permutation screen at all layers, then 10,000-permutation full test at top-5 candidate layers plus the pre-registered peak layer.
4. **Linear Probes**: Train L2-regularized logistic regression probes (40-class product, 8-class category, 5-class register) at each layer with GroupKFold CV (grouped by product_id) and nested CV for regularization tuning. Include control tasks with 1 random permutation per probe task for selectivity measurement.
5. **Statistical Testing & Reporting**: Evaluate pre-registered falsification criteria for H1/H2/H3, run all control analyses (including split-half reliability baseline for fictional-vs-real comparison), generate visualizations, and produce a go/no-go verdict.

Each stage produces well-defined artifacts (JSON, HDF5, NPY, CSV) and can be re-run independently given its inputs. A lightweight staleness warning (timestamp comparison at stage entry points) flags when outputs may be stale relative to their inputs.

## Architecture

### Pipeline Design

```
Stage 1: Stimulus Generation  -->  Stage 2: Hidden State Extraction  -->  Stage 3: RSA + Cosine Analysis
     (JSON output)                    (HDF5 output)                         (NPY/CSV output)
                                                                                    |
                                                                                    v
                                                          Stage 4: Linear Probes  -->  Stage 5: Reporting
                                                              (CSV/JSON output)        (PNG/PDF + Markdown)
```

### Module Structure

The implementation uses 5 core modules plus a single entry point:

```
protocol-layer-hypothesis/
├── pyproject.toml
├── .env.example              # ANTHROPIC_API_KEY=sk-ant-your-key-here, OPENAI_API_KEY=sk-your-key-here
├── .gitignore                # data/, .env, *.h5, __pycache__
├── run.py                    # Single entry point with subcommands (generate, extract, analyze, probe, report, all)
├── stimuli.py                # Stimulus generation, prompts, validation, BoW baseline, register distinctiveness
├── extraction.py             # Model loading, hooks, pooling, extraction pipeline, NaN/Inf checks
├── analysis.py               # RDM computation, RSA, anisotropy correction, cosine conditions, partial RSA
├── probes.py                 # Probe training, control tasks, zone classification, GroupKFold + nested CV
├── viz.py                    # All visualizations (phase plots, probe curves, RSA heatmaps, etc.)
├── tests/
│   ├── test_rdm.py           # RDM computation correctness
│   ├── test_rsa.py           # RSA correlation + partial RSA
│   ├── test_anisotropy.py    # Anisotropy correction (mean centering, whitening)
│   ├── test_pooling.py       # Mean pool and last-token pool on synthetic tensors
│   └── test_groupkfold.py    # GroupKFold stratification (no product leaks across folds)
├── data/                     # Git-ignored generated artifacts
└── notebooks/                # Optional exploratory analysis
```

No Pydantic config system -- use a Python dataclass or dict at the top of `run.py`. No `StageCheckpoint` class -- HDF5 incremental writes handle extraction recovery, and `stimuli.json` saves handle generation recovery. No debug model config -- test code logic with random tensors (shape-correct numpy arrays), test integration with the real model using the 5-stimulus pilot. No separate entry scripts per stage -- `run.py` with subcommands handles everything.

### Configuration

Configuration lives as a Python dataclass or dict in `run.py`:

```python
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
    "primary_model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",  # 64 layers, hidden_dim=5120, 40 attn heads, 8 KV heads (GQA)
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
    "h1_min_effect_size": 0.3,  # Cohen's d
    "h1_min_rsa_r": 0.1,       # RSA correlation at peak
    "h2_register_dominance_threshold": 5,  # percentage points
    "h3_layer_zone_pct": (10, 70),  # middle 60% of layers
    "h3_advantage_threshold": 2,  # percentage points over output layer
    "quant_control_threshold": 0.9,  # Spearman rho for FP16 vs 4-bit

    # Hardware
    "target_gpu": "RTX 5090 (32GB VRAM)",
    "system_ram": "32GB",
}
```

### Data Flow

- **Stimuli**: `data/stimuli.json` -- 800+ stimulus objects with product ID, category, register, text, token count, core attributes, generator metadata.
- **Hidden States**: HDF5 archives per model config in `data/`, containing mean-pooled representations at each layer for residual stream, attention output, and MLP output. Structure: `/hidden_states_mean_no_special` (N x L+1 x D), `/attention_output_mean_no_special` (N x L x D), `/mlp_output_mean_no_special` (N x L x D). Saved incrementally with gzip compression (level 4).
- **Analysis Outputs**: Per-layer RSA correlations, permutation test null distributions, condition similarities, probe F1 scores with bootstrap CIs -- all saved as NPY/JSON in `data/`.

### Storage Budget

| Data | Size |
|------|------|
| Stimuli JSON | < 1 MB |
| Hidden states (all models, all components, gzipped HDF5) | ~6 GB |
| Similarity matrices + probe results | ~1 GB |
| Figures and reports | < 1 GB |
| Model downloads (cached) | ~84 GB |
| **Total experimental data** | **~8 GB** |
| **Total including model cache** | **~92 GB** |

### Staleness Warning

At each stage entry point, check if any input file is newer than the expected output file. If so, print a warning: "Stage N outputs may be stale -- input X was modified after stage N last ran." This is ~10 lines of code, not a framework. Implemented as a simple function:

```python
def check_staleness(inputs: list[Path], outputs: list[Path]) -> None:
    for out in outputs:
        if out.exists():
            out_mtime = out.stat().st_mtime
            for inp in inputs:
                if inp.exists() and inp.stat().st_mtime > out_mtime:
                    print(f"WARNING: {out.name} may be stale -- {inp.name} was modified after it.")
```

## Implementation Plan

### Phase 1: Setup & Model Identity Verification
> Verify the target model, set up minimal project structure, and define the product catalog.

#### Step 1.1: Model Download & Architecture Verification
- **Files**: None (manual verification)
- **Changes**: Download `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` from HuggingFace. Verify architecture matches expected spec: 64 layers, hidden_dim=5120, 40 attention heads, 8 KV heads (GQA), head_dim=128. Confirm module hierarchy for hooks: `model.model.layers[i].self_attn`, `model.model.layers[i].mlp`. Architecture class is `Qwen2ForCausalLM` (standard decoder-only, all 64 layers homogeneous). GPTQ 4-bit symmetric, group_size=128, desc_act=false. Load with native transformers: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", device_map="auto")`. Expected VRAM: ~22-25 GB (leaves 7-10 GB headroom on 32GB RTX 5090).
- **Acceptance Criteria**: Model loads successfully. `model.config.num_hidden_layers == 64`. `model.config.hidden_size == 5120`. Forward pass on a test string returns `output_hidden_states` tuple of length 65 (embedding + 64 layers).
- **Time Budget**: 15 minutes (excluding download time)
- **Risks & Mitigations**: Model download is ~18GB. **Mitigation**: Start download at beginning of evening, work on other tasks while it downloads. Known caveat: `output_hidden_states` may return zeros with multi-GPU `device_map="auto"` (GitHub #36636) -- not an issue on single GPU.

#### Step 1.2: Project Initialization
- **Files**: `pyproject.toml`, `.gitignore`, `.env.example`, `run.py` (skeleton)
- **Changes**: Create pyproject.toml with dependencies: `torch>=2.5` (CUDA 12.6+ for RTX 5090 Blackwell), `transformers>=4.40` (native GPTQ support), `accelerate>=0.28`, `h5py>=3.10`, `numpy>=1.26`, `scipy>=1.12`, `scikit-learn>=1.4`, `matplotlib>=3.8`, `anthropic>=0.40`, `openai>=1.12`, `tqdm>=4.66`, `pandas>=2.2`. Dev deps: `pytest>=8.0`, `pytest-cov>=4.0`, `ruff>=0.3`. Note: `auto-gptq` is NOT a required dependency; native transformers GPTQ support is preferred. Keep `auto-gptq` as an optional fallback only.
- **Acceptance Criteria**: `pip install -e ".[dev]"` succeeds. `python run.py --help` shows subcommands.
- **Time Budget**: 15 minutes
- **Risks & Mitigations**: Torch version must support RTX 5090 Blackwell architecture. **Mitigation**: Verify `nvidia-smi` output and CUDA version on target machine before pinning torch version.

#### Step 1.3: Product Catalog
- **Files**: Top of `stimuli.py` (constants section)
- **Changes**: Define all 40 real products (8 categories x 5 products each) and 40 fictional products as Python data structures. Categories: Oral Care, Pet Food, Home Cleaning, Sports Nutrition, Baby Care, Coffee/Beverage, Skincare, Smart Home. Each product has: id, name, category, is_fictional flag, 3-5 specific quantitative core attributes, and distinguishing features. Define REGISTER_SPECS with voice, tone, structure, vocabulary, example_source for each of the 5 registers (Marketing Copy, Regulatory/Technical, Casual Social, Patent/IP, Journalistic). Define CROSS_GENERATOR_SUBSET_IDS (10 products: 5 real + 5 fictional across 5 categories). Fictional products must have plausible but non-existent names with novel feature combinations.
- **Acceptance Criteria**: 40 real + 40 fictional products defined. No two products in the same category share >1 core attribute. Fictional names pass a plausibility check. Every product has 3-5 specific, quantitative core attributes.
- **Time Budget**: 45 minutes (this is a substantial creative task requiring domain knowledge across 8 categories)
- **Risks & Mitigations**: Fictional product names might accidentally match real products. **Mitigation**: Quick web search during creation to verify novelty.

#### Step 1.4: Seed & I/O Utilities
- **Files**: Top of `run.py` or shared utility section
- **Changes**: Implement `set_global_seed(seed)` for Python random, numpy, torch (CPU+CUDA) with deterministic cuDNN. Implement `check_staleness()` function (timestamp comparison, ~10 lines). Implement basic HDF5 NaN/Inf validation (a few assert statements, not a full module).
- **Acceptance Criteria**: `set_global_seed(42)` produces deterministic numpy random output. Staleness warning prints correctly when inputs are newer than outputs.
- **Time Budget**: 15 minutes

### Phase 2: Stimulus Generation
> Generate ~800 primary stimuli plus ~100 cross-generator stimuli. Validate semantic anchoring and register distinctiveness.

#### Step 2.1: Prompt Templates & Generation
- **Files**: `stimuli.py`
- **Changes**: Implement `build_generation_prompt(product, register, variant_index)` that constructs a prompt instructing the LLM to generate a product description in the target register, grounded in core attributes. Constraints: target 80-150 tokens, all core attributes must be conveyed, no product name as heading, no meta-commentary. Variant 0 vs 1 get different diversity instructions. Include explicit register specs (voice, tone, structure, vocabulary, example source) in each prompt. Implement `generate_all_stimuli(config)` that generates: (a) Primary: 80 products x 5 registers x 2 variants = 800 via Claude Sonnet (using `anthropic.Anthropic()` with `ANTHROPIC_API_KEY`, temperature=0.7, max_tokens=300); (b) Cross-generator: 10-product subset x 5 registers x 2 variants via GPT-4 (whichever variant is current at implementation time, likely gpt-4o). Save to `stimuli.json` after each batch of 20 stimuli for crash recovery. Validate all stimuli: token count within [50, 200] (target [80, 150]), core attribute coverage via keyword extraction and fuzzy matching. Retry failures up to 3 times.
- **Acceptance Criteria**: `data/stimuli.json` contains 800+ primary stimuli and 100+ cross-generator stimuli. Every stimulus has token_count in [50, 200]. Every stimulus passes validation with at most warnings. 5% manual spot-check confirms semantic anchoring and register distinctiveness.
- **Time Budget**: 60 minutes (including prompt engineering iteration and spot-check)
- **Risks & Mitigations**:
  - Register-specific prompts may produce texts that converge in style rather than diverging. **Mitigation**: Include explicit register specs in each prompt; run quantitative register distinctiveness check (Step 2.2).
  - API rate limits. **Mitigation**: Exponential backoff with max_retries=5. ~800 calls is trivial for Anthropic tier.
  - If GPT-4 API is unavailable, skip cross-generator control -- it is Tier 2 and the generator confound can be acknowledged as a limitation.

#### Step 2.2: BoW Baseline & Register Distinctiveness Check
- **Files**: `stimuli.py`
- **Changes**: After stimulus generation, implement two quality gates:
  1. **BoW baseline**: Train a bag-of-words classifier (TF-IDF + logistic regression, using existing scikit-learn) on the same three tasks (product, category, register). If BoW 40-class product classification accuracy is high (>50%), surface features may be driving results and stimuli need revision before proceeding. This is a leakage check from the original experiment design.
  2. **Register distinctiveness (quantitative soft gate)**: Compute mean pairwise TF-IDF cosine distance between registers for each product. If mean inter-register distance is not at least 1.5x mean intra-register distance, print a warning and consider iterating on prompts. This is a soft threshold (warning, not hard gate) because there is no established norm for "how different is different enough." The quantitative measure gives a publishable number and catches shallow register variation early.
- **Acceptance Criteria**: BoW baseline accuracy computed and reported for all three tasks. Register distinctiveness metric computed per product and averaged. If BoW product accuracy > 50% or register distinctiveness below threshold, warning printed with recommendation to revise stimuli.
- **Time Budget**: 30 minutes
- **Risks & Mitigations**: The register distinctiveness threshold is empirically calibrated against the actual data distribution rather than pre-set. If stimuli fail the soft gate, budget additional prompt engineering time (could consume up to 1 hour).

#### Step 2.3: Stimulus Generation Tests
- **Files**: `tests/test_rdm.py` (or a dedicated test, but keeping test count at 5)
- **Changes**: Include basic stimulus validation tests alongside analysis tests. Test that prompt generation produces valid strings with required elements. Test that validation detects missing attributes. Mock all API calls and tokenizer.
- **Acceptance Criteria**: Tests pass. No API or network calls in tests.
- **Time Budget**: Included in test writing (Phase 4)

### Phase 3: Hidden State Extraction
> Extract per-layer representations from all models. Critical VRAM validation gate.

#### Step 3.1: 5-Stimulus Pilot Validation (CRITICAL GATE)
- **Files**: `extraction.py`
- **Changes**: Before any full extraction, implement and run a pilot on 5 real stimuli with the primary model (Qwen 4-bit GPTQ). Verify: (a) model loads correctly via native transformers GPTQ (`transformers>=4.40`, no `auto-gptq` required); (b) `output_hidden_states=True` returns correct layer count and shapes; (c) forward hooks fire on `model.model.layers[i]`, `model.model.layers[i].self_attn`, and `model.model.layers[i].mlp`; (d) no NaN/Inf in captured tensors; (e) wall-clock time per stimulus measured. This replaces the 1-stimulus pilot and the debug model -- testing on the real model with 5 stimuli validates integration directly.
- **Acceptance Criteria**: Pilot produces hidden states with shape (5, L+1, D) for residual stream and (5, L, D) for attention/MLP components. No NaN/Inf. Wall-clock time per stimulus recorded and used to update extraction time estimate.
- **Time Budget**: 15-30 minutes (including time for debugging hook registration if module names differ from expected)
- **Risks & Mitigations**:
  - `output_hidden_states` incompatible with GPTQ kernels. **Mitigation**: Register hooks on `model.model.layers[i]` (DecoderLayer modules) as fallback. These wrapper modules are preserved even under GPTQ quantization.
  - Hooks on `self_attn`/`mlp` fail on quantized layers. **Mitigation**: Fall back to residual-stream-only extraction (defer component decomposition to Tier 3).
  - Model loading fails entirely. **Mitigation**: Try AWQ variant or bitsandbytes 8-bit.
  - Module naming differs from expected. **Mitigation**: Budget 30-60 minutes for debugging; inspect `model.named_modules()` to discover the correct hierarchy.

#### Step 3.2: Model Loading
- **Files**: `extraction.py`
- **Changes**: Implement `load_model_and_tokenizer(model_name, device_map)` that loads the model via native transformers GPTQ integration. Handles: GPTQ-4bit (via `transformers` native loading), FP16 with `device_map="auto"` (CPU offloading via `accelerate`), standard FP16. Sets `pad_token = eos_token` if missing. Returns model in `eval()` mode. Helper functions: `get_layer_count(model)`, `get_hidden_dim(model)`.
- **Acceptance Criteria**: Model loads without error. `model.eval()` applied. Layer count and hidden dim match expected values from Step 1.1.
- **Time Budget**: Included in pilot (Step 3.1)

#### Step 3.3: Forward Hooks & Pooling
- **Files**: `extraction.py`
- **Changes**: Implement `ExtractionHooks` class that registers forward hooks on every transformer layer to capture attention output and MLP output. Hooks move captured tensors to CPU immediately via `.detach().cpu()`. Architecture-agnostic layer/module discovery via inspection of `model.named_modules()`. Implement `mean_pool_no_special(hidden_states, attention_mask, special_ids, input_ids)` that mean-pools excluding BOS/EOS/PAD tokens. Implement `last_token_pool(hidden_states, attention_mask)` as secondary pooling method.
- **Acceptance Criteria**: Hooks fire on every layer. Captured attention and MLP tensors have correct shapes. Mean pooling correctly excludes special tokens. Both pooling methods produce (batch, hidden_dim) output.
- **Time Budget**: Included in pilot (Step 3.1)

#### Step 3.4: Extraction Pipeline
- **Files**: `extraction.py`
- **Changes**: Implement `extract_hidden_states(config, stimuli, model_name)` that processes all stimuli through the specified model at batch_size=1. For each stimulus: tokenize, forward pass with hooks, pool each layer's hidden states, **check for NaN/Inf immediately after mean pooling** (3 lines of code -- if detected, log the stimulus ID, skip it, and continue), save incrementally to HDF5 with gzip compression. Resume support: check which stimulus_ids already exist in HDF5 before processing. GPU memory cleanup after each forward pass (`torch.cuda.empty_cache()`). OOM handling: catch CUDA OOM, log stimulus, clear cache, skip and continue.
- **Acceptance Criteria**: HDF5 output contains all expected stimuli (minus any NaN/Inf skips, which are logged). Shapes correct: (N, L+1, D) for hidden_states, (N, L, D) for attention/MLP. No NaN/Inf in final HDF5. GPU VRAM stays under 30GB during extraction. Resume works (kill and restart produces same output).
- **Time Budget**: 30-60 minutes for Qwen 4-bit extraction of 800 stimuli (realistic estimate including hook capture, CPU transfer, pooling, and gzip HDF5 writes at 2-5 seconds per stimulus). 20-30 minutes for Llama-3.1-8B.
- **Risks & Mitigations**:
  - OOM on Qwen FP16 run (Tier 4, deferred). **Mitigation**: Set 16GB swap file, close all apps, test with 5 stimuli first. If >30 sec/stimulus, reduce subset to 100. If infeasible, use 8-bit quantization as intermediate control or skip.
  - Disk write bottleneck. **Mitigation**: Mean-pool during extraction (not after). Do NOT store raw token-level hidden states. Write incrementally to HDF5.

### Phase 4: Similarity Analysis (RSA)
> Primary analysis: RSA on pairwise distances. Anisotropy correction with numerical stability.

#### Step 4.1: Anisotropy Correction
- **Files**: `analysis.py`
- **Changes**: Implement `correct_anisotropy(representations, method, n_components)` supporting three methods: "none" (passthrough), "mean_centering" (subtract global mean), "whitening" (PCA-whitening via `sklearn.decomposition.PCA(n_components=min(N-1, D), whiten=True)` which handles truncation internally via SVD). Add epsilon=1e-8 regularization. Warn if retained variance < 95%. Monitor condition number per layer, flag layers where it exceeds 1e6. Apply correction independently at each layer. Process layer-by-layer from HDF5 using chunked reads -- never load all layers simultaneously.
- **Acceptance Criteria**: Mean centering produces zero-mean output. Whitening produces unit-variance, uncorrelated components. No NaN/Inf in output. Handles rank-deficient case (N < D) correctly. Whitened output dimensionality = min(N-1, D).
- **Time Budget**: 30 minutes (implementation + verification)
- **Risks & Mitigations**: Whitening amplifies noise dimensions. **Mitigation**: PCA-whitening with truncation retains only components above epsilon. Report both corrected and uncorrected results -- if they diverge, this is itself an informative finding about anisotropy and phase structure.

#### Step 4.2: RDM & RSA Computation
- **Files**: `analysis.py`
- **Changes**: Implement `compute_rdm(representations, metric="cosine")` using scipy's `pdist` + `squareform`. Implement `compute_rdms_all_layers()` for per-layer computation. Build three theoretical model RDMs:
  1. **Product-identity model RDM**: same-product=0, same-category=0.5, different-category=1.0
  2. **Register-identity model RDM**: same-register=0, different-register=1
  3. **Within-category product discrimination model RDM**: only includes within-category pairs, predicting same-product=0, different-product-same-category=1

  Implement `rsa_correlation(observed_rdm, model_rdm)` computing Spearman rank correlation between upper triangles. Implement `partial_rsa(observed_rdm, model_rdm, nuisance_rdms)` controlling for stimulus length difference RDM and lexical overlap RDM (Jaccard distance on token sets) as nuisance regressors. Uses partial Spearman correlation: regress out nuisance RDMs from both observed and model RDM vectors, then correlate residuals.

  Implement `compute_condition_similarities()` that classifies all pairs into SP-DR (same product, different register), DP-SC (different product, same category), and DC (different category), computes mean cosine similarity per condition per layer. Also construct nuisance RDMs for stimulus length difference and lexical overlap.
- **Acceptance Criteria**: RDMs are symmetric with zero diagonal. Model RDMs correctly encode the expected distance structure. RSA on synthetic data with known structure recovers expected correlations. Partial RSA reduces correlation when nuisance variable is confounded. Pair counts match expected combinatorics.
- **Time Budget**: 45 minutes (implementation + synthetic verification)

#### Step 4.3: Tiered Permutation Testing
- **Files**: `analysis.py`
- **Changes**: Implement tiered permutation testing strategy:
  1. **Fast screen**: 200 permutations at all layers to identify candidate significant layers. Each permutation shuffles model RDM labels and recomputes RSA. Time: ~12 minutes for 200 perm x 64 layers.
  2. **Full test**: 10,000 permutations at the top-5 candidate layers (by RSA magnitude from the screen) plus the pre-registered peak layer. Time: ~20 minutes for 10,000 perm x ~6 layers.
  3. **FDR correction**: Apply Benjamini-Hochberg FDR correction across layers for the screening p-values.

  This reduces computation by ~95% compared to full 10,000-permutation testing at all layers while retaining comprehensive layer scanning and rigorous significance testing at key layers.
- **Acceptance Criteria**: Permutation test produces p-values near 0 for strong effects and near 0.5 for null effects on synthetic data. FDR correction applied correctly.
- **Time Budget**: ~35 minutes runtime for both tiers combined per model

#### Step 4.4: Intermediate RSA Sanity Check (DECISION GATE)
- **Files**: `analysis.py` (or manual inspection during `run.py analyze`)
- **Changes**: After the first Qwen RSA results, spend 15 minutes on a three-way triage:
  - **Flat RSA across all layers** (no peak anywhere): STOP. Check stimuli quality (are registers actually distinct? Re-examine BoW baseline). Check extraction (are hidden states non-degenerate? Look for all-zeros or collapsed representations). Check RDM construction (verify pair classification). Do not proceed until diagnosed.
  - **Weak middle-layer peak** (RSA r < 0.05 at peak): Proceed with probes to get additional signal, but flag that the effect may be null or very small. Consider whether partial RSA (controlling for length/lexical overlap) changes the picture.
  - **Clear middle-layer peak** (RSA r > 0.1 at peak): Proceed as planned.
- **Acceptance Criteria**: Triage completed. Decision documented. If flat: root cause identified before proceeding. If weak: acknowledged in notes.
- **Time Budget**: 15 minutes
- **Risks & Mitigations**: The most dangerous failure mode is grinding through probe training and statistical testing on data where a trivial upstream bug would have been caught by eyeballing the RSA curve. This 15-minute check could save hours.

### Phase 5: Linear Probes
> Secondary analysis with control tasks. Cross-validated probe training at every layer.

#### Step 5.1: Probe Training Pipeline
- **Files**: `probes.py`
- **Changes**: Implement `train_probe_at_layer(representations, labels, groups, ...)` that trains L2-regularized logistic regression with 5-fold GroupKFold (groups=product_id, ensuring all variants of one product stay in the same fold) and nested CV for regularization (within each outer fold, tune C over {0.01, 0.1, 1.0, 10.0, 100.0} via 3-fold inner CV). Use `lbfgs` solver, multinomial, max_iter=2000. Record macro-F1, per-fold F1, bootstrap 95% CIs. Implement `train_probes_all_layers()` that runs all three probe tasks (40-class product, 8-class category, 5-class register) at all layers for all anisotropy correction methods.
- **Acceptance Criteria**: 40-class probe achieves above chance (>2.5%) at peak layer. Per-fold results reported. Bootstrap CIs non-degenerate. Stratification verified: no product appears in both train and test of any fold.
- **Time Budget**: ~1.5-4 hours for all probe fits (5 outer folds x 3 inner folds x 5 C values x ~64 layers x 3 tasks x 2-3 anisotropy methods = ~15,000+ fits at ~0.5s each)
- **Risks & Mitigations**:
  - 40-class probe with ~10 samples/class has high variance. **Mitigation**: RSA is primary evidence. Report per-fold variance. Probes are secondary confirmation.
  - Solver convergence warnings at some layers. **Mitigation**: max_iter=2000 with lbfgs solver. Flag layers with convergence warnings.
  - GroupKFold with 40 classes means each test fold has ~8 real products. **Mitigation**: Document this limitation. Report micro-F1 alongside macro-F1.

#### Step 5.2: Control Task Probes (Hewitt & Manning 2019)
- **Files**: `probes.py`
- **Changes**: For each real probe (product, category, register), train a matched control probe where labels are randomly permuted (1 random permutation per probe task) but class sizes are preserved. Compute **selectivity** = real accuracy - control accuracy at each layer. This is the single most important methodological addition -- without it, probe accuracy is uninterpretable given the high dimensionality of representations relative to sample size. Using 1 permutation per task (standard in the original Hewitt & Manning paper) gives 3 additional probe fits per layer, not 15-30.
- **Acceptance Criteria**: Control probes produce non-trivial accuracy (demonstrating probe capacity). Selectivity is positive at layers where real probes perform well. Selectivity is near zero at layers where real probes are at chance.
- **Time Budget**: Included in Step 5.1 runtime (adds ~3 fits per layer)

#### Step 5.3: Zone Classification
- **Files**: `probes.py`
- **Changes**: Implement `compute_zone_boundaries(n_layers)` that divides layers into: early (0-10th percentile), protocol (10th-70th percentile), late (90th-99th percentile), output (final layer). Train probes on mean-pooled zone representations for all three tasks. This provides the H3 test statistic.
- **Acceptance Criteria**: Zone boundaries computed correctly for Qwen and Llama (different layer counts). Zone probes produce F1 scores and CIs.
- **Time Budget**: 15 minutes

### Phase 6: Statistical Testing & Reporting
> Pre-registered hypothesis evaluation, control analyses, visualization, go/no-go verdict.

#### Step 6.1: Hypothesis Tests
- **Files**: `analysis.py` or dedicated section in `run.py report`
- **Changes**: Implement pre-registered falsification criteria:
  - **H1 (Phase Structure)**: RSA correlation significantly higher at middle layers vs. early layers. Criterion: p < 0.05 (permutation test, FDR-corrected) AND Cohen's d > 0.3 AND RSA r > 0.1 at peak layer. Also test: peak product-identity RSA correlation occurs in the middle 60% of layers.
  - **H2 (Content Dominance)**: Product-identity model RDM correlates more strongly with observed RDMs in middle layers than register-identity model RDM. Also: register probe does not exceed category probe selectivity by >5pp at any protocol-zone layer.
  - **H3 (Protocol Layer Advantage)**: Best-performing layer for 40-class probe (by selectivity, not raw accuracy) falls in middle 60% of layer stack AND outperforms output layer by >=2pp. Also: peak RSA correlation in middle 60% is significantly higher than RSA at output layer (permutation test).
  - **Multiple comparison correction**: Benjamini-Hochberg FDR correction across layers for all per-layer tests. Report both corrected and uncorrected results.
- **Acceptance Criteria**: Tests correctly classify synthetic data known to support/falsify each hypothesis. Effect-size criteria applied alongside significance.
- **Time Budget**: 30 minutes

#### Step 6.2: Control Analyses
- **Files**: `analysis.py`
- **Changes**: Implement all control analyses:
  1. **Memorization control**: Compare RSA curves for real vs. fictional products. Pass criterion: use **split-half reliability baseline** -- compute split-half reliability of the real-product RSA curve (split 40 real products into two random subsets of 20, compute RSA on each, correlate). The fictional-vs-real correlation is compared against this empirical ceiling rather than an arbitrary r > 0.7 threshold. This is empirically grounded and the experiment has not been pre-registered yet, so there is no commitment to the old threshold.
  2. **Quantization control** (Tier 4, deferred): Compare Qwen FP16 vs. 4-bit RSA curves. Pre-registered criterion: Spearman rho > 0.9.
  3. **Generator control**: Compare Claude vs. GPT-4 RSA at peak layer for multi-source subset. Test generator main effect with ANOVA on pairwise distances.
  4. **Within-category product discrimination**: Compute RSA with within-category-only model RDM. If protocol-layer hypothesis is correct, within-category discrimination peaks in same middle layers. If only topic modeling, within-category discrimination is flat. This is the key test distinguishing the two interpretations.
- **Acceptance Criteria**: All Tier 1 controls produce quantified results. Fictional vs. real comparison uses split-half baseline. Within-category discrimination computed and plotted.
- **Time Budget**: 45 minutes

#### Step 6.3: Go/No-Go Decision Logic
- **Files**: `run.py report` or `analysis.py`
- **Changes**: Mechanical evaluation:
  - **GO**: All three hypotheses supported + all critical controls passed.
  - **QUALIFIED_GO**: H1+H2 supported + controls passed, H3 marginal. This is the most likely outcome. It means middle layers show the predicted phase structure and content dominance, but the protocol-zone advantage over other layers is not convincingly large. This supports the three-phase model but does not establish the stronger "protocol layer" claim.
  - **NO_GO**: Any hypothesis falsified or critical control failed.
  Each outcome includes a summary explanation with specific numbers.
- **Acceptance Criteria**: Correctly classifies known pass/fail inputs.
- **Time Budget**: 15 minutes

#### Step 6.4: Visualization
- **Files**: `viz.py`
- **Changes**: Generate all figures (PNG 300 DPI + PDF):
  1. Three-condition similarity curves (SP-DR, DP-SC, DC) per layer with 95% CI bands
  2. RSA correlation per layer (product-identity, register-identity, within-category) with significance markers
  3. Probe accuracy/selectivity curves (product, category, register) per layer with chance lines and error bars
  4. Zone comparison bar charts (early, protocol, late, output for each task)
  5. Decomposition panels (attention/MLP/residual RSA curves, if component data available)
  6. Memorization control overlay (real vs. fictional RSA curves)
  7. Full RDM heatmaps at selected layers (early, middle, late)
  8. Register confusion matrices at selected layers
  9. Quantization control overlay (if Tier 4 completed)
- **Acceptance Criteria**: All applicable plots generate without error. Layer indices correct. Legend readable. Chance levels marked.
- **Time Budget**: 1-2 hours (implementation + iteration on readability)

#### Step 6.5: Test Suite
- **Files**: `tests/test_rdm.py`, `tests/test_rsa.py`, `tests/test_anisotropy.py`, `tests/test_pooling.py`, `tests/test_groupkfold.py`
- **Changes**: 5 critical-path test files covering:
  1. **test_rdm.py**: RDM computation produces symmetric matrix with zero diagonal. Model RDM correctly encodes three-level distance structure. Within-category model RDM selects correct pairs.
  2. **test_rsa.py**: RSA correlation recovers known structure from synthetic data. Partial RSA reduces correlation when nuisance is confounded. Permutation test produces expected p-values.
  3. **test_anisotropy.py**: Mean centering produces zero-mean output. Whitening produces unit-variance components. No NaN/Inf for rank-deficient input. Handles N < D case correctly.
  4. **test_pooling.py**: Mean pooling correctly excludes special tokens on synthetic tensors. Last-token pooling selects correct position. Both produce correct output shapes.
  5. **test_groupkfold.py**: GroupKFold with product_id groups: no product appears in both train and test. All variants of one product stay together. Fold sizes are balanced.

  All tests run without GPU, API keys, or network access. Use synthetic tensors and mock data.
- **Acceptance Criteria**: `pytest tests/ -v` passes. No external dependencies in tests.
- **Time Budget**: 1-2 hours (spread across implementation phases, not a single block)

## Risk Mitigations

| Risk | Severity | Mitigation | Applied In |
|------|----------|------------|------------|
| `output_hidden_states` / hooks incompatible with quantized model | CRITICAL | 5-stimulus pilot before any extraction. Fallback: hooks on DecoderLayer. If all fails: switch to bitsandbytes 8-bit. Budget 30-60 min for hook debugging. | Phase 3, Step 3.1 |
| Model identity unknown (layer count, hidden dim, module names) | CRITICAL | Verify exact HuggingFace model ID, architecture, and module hierarchy FIRST (Step 1.1). Update all estimates before writing code. | Phase 1, Step 1.1 |
| Whitening numerical instability (rank deficiency) | HIGH | PCA-whitening with truncation (n_components = min(N-1, D)). Epsilon=1e-8. Monitor condition number per layer. | Phase 4, Step 4.1 |
| RAM exhaustion during analysis | HIGH | Process layer-by-layer from HDF5 chunked reads. Never load all layers simultaneously. Peak RAM ~4GB. | Phase 4, all steps |
| Qwen FP16 OOM or impractically slow | HIGH (Tier 4) | 16GB swap file. Test 5 stimuli. Fallback: 8-bit quant or skip. Run overnight as unattended batch if attempted. | Phase 3, Step 3.4 |
| Extraction takes longer than expected | MEDIUM | Budget 30-60 min (not 7 min). Per-stimulus includes hook capture, CPU transfer, pooling, gzip write at 2-5 sec/stimulus. | Phase 3, Step 3.4 |
| 40-class probe high variance (10 samples/class) | LOW | RSA is primary (19,900+ pairwise distances). Probes are secondary. Report per-fold variance. Supplement with micro-F1. | Phase 5, Step 5.1 |
| Stimulus register variation is shallow | MEDIUM | Quantitative TF-IDF register distinctiveness check (soft gate). BoW baseline leakage check. Budget prompt iteration time. | Phase 2, Step 2.2 |
| Timeline overrun (>20 hours) | MEDIUM | Tier-based scope cutting. Never cut: RSA, probes, fictional control. First cut: FP16 subset. Second: component decomposition. Third: Llama cross-validation. | All |
| Lexical confounds mimic protocol-layer effect in RSA | HIGH | Partial RSA controlling for lexical overlap (Jaccard) and stimulus length. | Phase 4, Step 4.2 |
| Probe results uninterpretable without control tasks | HIGH | Hewitt & Manning control tasks (1 permutation per probe task). Non-negotiable for publishable results. | Phase 5, Step 5.2 |
| NaN/Inf in hidden states from specific stimuli | MEDIUM | Per-stimulus NaN/Inf check in extraction loop, immediately after mean pooling. Skip and log if detected. | Phase 3, Step 3.4 |
| Stale downstream artifacts after re-running a stage | LOW | Timestamp-based staleness warning at each stage entry point (~10 lines of code). | Architecture (staleness check) |
| Results are ambiguous / "qualified GO" | HIGH | Pre-registered criteria with effect-size thresholds. Multiple analytical angles (RSA, probes, within-category discrimination). Budget extra time for nuanced interpretation. | Phase 6, Step 6.3 |

## Execution Schedule

### Evening 1 (4-4.5 hours): Setup + Stimulus Generation + Pilot + Begin Extraction

| Task | Time | Running Total |
|------|------|---------------|
| Model identity verification (search HuggingFace, confirm architecture) | 15 min | 0:15 |
| Project initialization (pyproject.toml, .gitignore, run.py skeleton) | 15 min | 0:30 |
| Product catalog (40 real + 40 fictional products with attributes) | 45 min | 1:15 |
| Seed & I/O utilities (set_global_seed, staleness check) | 15 min | 1:30 |
| Prompt templates + stimulus generation via API | 60 min | 2:30 |
| BoW baseline + register distinctiveness check | 30 min | 3:00 |
| Download primary model (~20 min, can overlap with spot-check) | 20 min | 3:20 |
| 5-stimulus pilot validation (hooks, shapes, NaN check) | 15 min | 3:35 |
| Begin Qwen 4-bit extraction (~30-60 min) | 30-60 min | 4:05-4:35 |

**Key enabler**: The simplified structure (no Pydantic config, no checkpoint system, no debug model, no 15-file test suite) drops scaffolding from "3-4 hours" to "~1.5 hours", making Evening 1 tractable without splitting into two sessions.

**If stimulus generation takes longer than expected** (prompt engineering iteration), extraction can spill into Evening 2 without cascading schedule damage.

### Evening 2 (4 hours): Llama Extraction + Analysis Code + First RSA Results

| Task | Time | Running Total |
|------|------|---------------|
| Llama-3.1-8B extraction (~800 stimuli) | 20-30 min | 0:30 |
| Write and test anisotropy correction (PCA-whitening) | 30 min | 1:00 |
| Write and test RDM + RSA computation | 45 min | 1:45 |
| Run RSA on Qwen 4-bit residual stream (corrected + uncorrected) | 15 min | 2:00 |
| Run tiered permutation testing | 35 min | 2:35 |
| **Intermediate RSA sanity check -- FIRST LOOK AT H1** | 15 min | 2:50 |
| Write condition similarity computation | 30 min | 3:20 |
| Generate initial phase structure plots | 40 min | 4:00 |

### Evening 3 (4 hours): Probes + Cross-Model Comparison

| Task | Time | Running Total |
|------|------|---------------|
| Write probe training pipeline (GroupKFold + nested CV) | 45 min | 0:45 |
| Run linear probes on Qwen 4-bit (all tasks, all layers) | 1.5-2 hr | 2:15-2:45 |
| Run control task probes (Hewitt & Manning) | 30 min | 2:45-3:15 |
| Run RSA on Llama (corrected + uncorrected) | 30 min | 3:15-3:45 |
| Compare Qwen vs. Llama RSA profiles (exploratory only) | 15 min | 3:30-4:00 |
| Write critical-path test suite | remaining | 4:00 |

### Evening 4 (4-5 hours): Deep Analysis + Statistical Testing

| Task | Time | Running Total |
|------|------|---------------|
| Within-category product discrimination analysis | 30 min | 0:30 |
| Multi-source stimulus analysis (generator effects) | 30 min | 1:00 |
| Partial RSA with nuisance regressors (length, lexical overlap) | 30 min | 1:30 |
| Memorization control (fictional vs. real, split-half baseline) | 30 min | 2:00 |
| Hypothesis tests (H1/H2/H3 with effect-size criteria) | 30 min | 2:30 |
| Go/No-Go evaluation | 15 min | 2:45 |
| All visualizations (9 plot types) | 1.5-2 hr | 4:15-4:45 |
| If time: component decomposition extraction (Tier 3) | remaining | -- |

### Evening 5 (3-4 hours, if needed): FP16 Control + Final Report

| Task | Time | Running Total |
|------|------|---------------|
| Start Qwen FP16 subset run (overnight batch, Tier 4) | 30 min setup | 0:30 |
| Analyze FP16 results (Spearman correlation with 4-bit) | 30 min | 1:00 |
| Component-decomposed RSA and probes (if Tier 3 data available) | 1 hr | 2:00 |
| Finalize all plots and tables | 1 hr | 3:00 |
| Write conclusions and archive data | 1 hr | 4:00 |

### Minimum Viable Results (MVR) -- Tiers 1-2, ~12-14 hours across 3 evenings

If time is constrained, the Tier 1-2 deliverables test all three hypotheses with cross-model context:

1. Full 800 stimuli (real + fictional) with BoW baseline and register distinctiveness check
2. Qwen 4-bit extraction (residual stream + hooks for components if pilot succeeds)
3. Llama-3.1-8B extraction (exploratory)
4. RSA analysis (corrected + uncorrected) with tiered permutation testing
5. Linear probes (40-class, 8-class, 5-class) with control tasks
6. Fictional product memorization control with split-half baseline
7. Within-category product discrimination
8. Phase structure plots, probe accuracy/selectivity curves, hypothesis verdicts

### Tier Ordering for Scope Cutting

| Tier | Items | Rationale |
|------|-------|-----------|
| **Tier 1 (Must Have)** | Stimuli, Qwen 4-bit extraction, RSA, probes with control tasks, fictional control, within-category discrimination, BoW baseline | Core hypothesis tests + critical controls |
| **Tier 2 (Should Have)** | Llama extraction + RSA, cross-model comparison (exploratory only) | Adds cross-model context but primary claims rest on Qwen |
| **Tier 3 (Nice to Have)** | Component decomposition (attention/MLP/residual), multi-source generator analysis | Mechanistic depth, not required for hypothesis tests |
| **Tier 4 (Deferrable)** | Qwen FP16 subset run, quantization control (Spearman > 0.9 criterion) | Important control but highest schedule risk; run overnight if attempted |

**Never cut**: RSA analysis, probe analysis, fictional product control, within-category discrimination, BoW baseline. These are the core of the experiment.

## Key Decisions

| # | Decision | Chose | Rationale |
|---|----------|-------|-----------|
| 1 | Primary probe granularity | 40-class individual product (not 8-class category) | 8 categories are lexically separable by BoW -- too easy to distinguish protocol-layer effect from topic clustering |
| 2 | Memorization control | 40 fictional products (invented brands, novel specs) analyzed separately | Real products appear across registers in pretraining data; clustering could reflect memorization rather than protocol-layer processing |
| 3 | Anisotropy handling | Rigorous correction (mean centering + whitening) AND report uncorrected | Middle layers are most anisotropic; raw cosine similarity inflates H1 curve. But anisotropy might BE the signal, so report both. |
| 4 | Quantization control | Spearman rho > 0.9 between FP16 and 4-bit per-layer RSA scores | Pre-registered quantitative threshold prevents subjective "looks similar enough" comparisons |
| 5 | Topic-modeling null hypothesis | Pre-register alongside protocol-layer hypothesis | Topic clustering makes identical predictions for 8-class categories; need explicit discriminating criteria |
| 6 | Stimulus contamination | Diversify generation sources (Claude + GPT-4 on 10-product subset) | Claude-generated stimuli may embed systematic patterns detectable by another transformer |
| 7 | Primary analytical method | RSA primary, 40-class probe secondary | 40-class probe with 5 samples/class is underpowered; RSA operates on 19,900+ pairwise distances with far more statistical power |
| 8 | Stimulus count | ~800 (10 per product: 5 registers x 2 variants) | Belt-and-suspenders: RSA primary + sufficient per-class samples for 40-class probe as secondary confirmation |
| 9 | Anisotropy correction decision | Analyze both corrected and uncorrected | Correction could flatten the effect being measured; need both to interpret |
| 10 | Fictional product analysis | 40 fictional, separate condition (not mixed with real) | Clean memorization control without contaminating main results |
| 11 | Multi-source generation | Fully crossed on 10-product subset (Claude + GPT-4). If generator effect negligible, main dataset uses Claude only | Tests for generator confound without requiring full crossing of all 40 products |
| 12 | Quantization control threshold | Spearman rho > 0.9 | Prevents subjective visual comparison |
| 13 | Target hardware | PowerSpec 5090 system (RTX 5090, 32GB VRAM) | Dedicated GPU system, not WSL desktop |
| 14 | Implementation structure | 5 flat modules + run.py (not 7 subpackages + Pydantic + checkpoint system) | All three red-team critics converged: infrastructure was over-scoped by 6-10 hours. Simplified structure preserves all analytical capability. |
| 15 | Register distinctiveness check | Quantitative soft gate (TF-IDF distance, warning threshold) | Cheap (~30 min), gives publishable number, catches shallow register variation early |
| 16 | Fictional-vs-real threshold | Split-half reliability baseline (replaces arbitrary r > 0.7) | Empirically grounded; experiment not yet pre-registered so no commitment to break |
| 17 | Llama's role | Exploratory cross-model comparison only, not confirmatory | Models differ on too many dimensions (size, architecture, training data, quantization); primary claims rest on Qwen only |

## Hypotheses & Falsification Criteria

### H1: Phase Structure
**Prediction**: Hidden-state representations of product descriptions exhibit the three-phase pattern (encoding -> convergence -> decoding divergence) across text registers.

**Falsification criteria**:
- RSA correlation with product-identity model RDM does NOT show a statistically significant increase from early layers to middle layers (p > 0.05, permutation test with FDR correction)
- OR Cohen's d < 0.3 (effect size too small)
- OR peak RSA r < 0.1 at best middle layer

### H2: Content Dominance
**Prediction**: In the middle layers (protocol zone), semantic content (product/category identity) is more geometrically prominent than surface register.

**Falsification criteria**:
- Register-identity model RDM correlates MORE strongly with observed RDMs than product-identity model RDM at any middle-layer point
- OR register probe selectivity exceeds category probe selectivity by >5pp at any protocol-zone layer

### H3: Protocol Layer Advantage
**Prediction**: Middle-layer representations outperform early, late, and output layers for product classification.

**Falsification criteria**:
- Best-performing layer (by 40-class probe selectivity) falls OUTSIDE the middle 60% of the layer stack
- OR best middle-layer probe does not outperform output-layer probe by >=2pp (selectivity)
- OR peak RSA correlation in middle 60% is NOT significantly higher than RSA at output layer (permutation test)

### Interpretive Notes

- With 19,900+ pairwise distances, even trivial effects are significant. Effect-size criteria (Cohen's d > 0.3, RSA r > 0.1) prevent over-interpreting statistically significant but practically meaningless effects.
- Report both FDR-corrected and uncorrected results. If corrected results are null but uncorrected show a clear pattern, this suggests a diffuse effect across many layers rather than a sharp peak.
- The experiment establishes **correlational geometric facts** about hidden states, not causal mechanisms. It cannot prove the model "uses" these representations during inference. Causal evidence (activation patching) would be a follow-up experiment.
- A **QUALIFIED_GO** (H1+H2 supported, H3 marginal) is the most likely outcome. This would support the three-phase model of transformer processing but not definitively establish the stronger "protocol layer advantage" claim.

## Domain Requirements

### Critical Methodological Requirements (Mandatory -- All Implemented)

1. **Control tasks for all probes** (Hewitt & Manning 2019): Train matched probes with permuted labels. Report selectivity = real accuracy - control accuracy. Without this, probe results are uninterpretable given high-dimensional representations (5120 dims) and small per-class sample sizes. (Step 5.2)

2. **Partial RSA for confound control** (Kriegeskorte & Kievit 2013): Include nuisance model RDMs for stimulus length difference and lexical overlap (Jaccard distance on token sets). For multi-source subset, include generator identity as additional nuisance regressor. (Step 4.2)

3. **Within-category product discrimination index**: The key test distinguishing the protocol-layer hypothesis from topic modeling. Compute RSA with within-category-only model RDM. If discrimination peaks in middle layers, this supports fine-grained semantic identity encoding beyond coarse topic membership. (Step 6.2)

4. **Effect-size criteria alongside significance**: With 19,900+ pairwise distances, even trivial effects are significant. Require Cohen's d > 0.3 for H1 and RSA r > 0.1 at peak layer. (Step 6.1)

5. **GroupKFold stratification**: Use product_id as group for CV splits. All variants of one product must be in the same fold to prevent data leakage. (Step 5.1)

6. **Nested CV for regularization**: Tune C over {0.01, 0.1, 1.0, 10.0, 100.0} via 3-fold inner CV rather than fixing C=1.0. Prevents regularization from being a hidden confound. (Step 5.1)

7. **Tiered permutation testing**: 200-permutation screen at all layers + 10,000-permutation full test at top-5 candidate layers + peak. Benjamini-Hochberg FDR correction across layers. (Step 4.3)

### Recommended Additions (Important but Optional)

8. **Both mean pooling and last-token pooling**: Report convergence/divergence between strategies. Mean pooling primary; last-token as robustness check.

9. **CKA (Centered Kernel Alignment)** as robustness check alongside RSA: Invariant to orthogonal transformation and isotropic scaling, potentially more robust to anisotropy.

10. **Register confusion matrices** at selected layers: Reveals which register pairs are confusable at which processing stages.

11. **Logit lens / tuned lens analysis**: Project each layer's hidden state through the final unembedding matrix. Converging evidence from a different analytical angle. (Follow-up if core results are positive.)

### Reporting Standards

The experiment must report:
- Probe architecture and hyperparameters (including cross-validated C)
- Control task accuracy (selectivity) at each layer
- Per-layer accuracy curves with 95% bootstrap CIs
- Number of probe parameters vs. training examples
- Stratification strategy (GroupKFold with product_id)
- Distance metric used for RSA (cosine)
- Model RDM construction (explicit three-level structure)
- Permutation test details (tiered: 200 screen + 10,000 full, number of layers tested)
- Multiple comparison correction method (Benjamini-Hochberg FDR)
- Both corrected and uncorrected anisotropy results
- All stimuli released (JSON with product attributes)
- Exact model versions and quantization details (HuggingFace repo + revision hash)
- Random seeds for all stochastic processes
- BoW baseline results and register distinctiveness metrics

## Open Questions

See `open-questions.md` for remaining unresolved items that must be addressed at implementation time.

## References

- Ng (2026) -- RYS layer-duplication experiments on Qwen2-72B and Qwen3.5-27B
- Liu & Niehues (ACL 2025) -- Cross-lingual alignment in middle layers
- Tyshchuk et al. (2024) -- Bell-shaped anisotropy profiles in decoder models
- Godey et al. (EACL 2024) -- Intrinsic anisotropy in self-attention
- Hewitt & Manning (2019) -- Control tasks for probing ("A Structural Probe for Finding Syntax")
- Belinkov (2022) -- "Probing Classifiers: Promises, Shortcomings, and Advances"
- Kriegeskorte et al. (2008) -- RSA foundational paper
- Kriegeskorte & Kievit (2013) -- Partial RSA for confound control
- Kornblith et al. (2019) -- CKA (Centered Kernel Alignment)
- Voita & Titov (2020) -- MDL probing
- Gao et al. (2019) -- Representation degeneration problem
- Lim, Aji, & Cohn (2025) -- Language-specific processing in larger models
- Tenney et al. (2019) -- "BERT Rediscovers the Classical NLP Pipeline"
- Meng et al. (2022) -- Activation patching (ROME)
