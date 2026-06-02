# Draft Plan: protocol-layer-hypothesis

## Summary

This experiment tests the **Protocol Layer Hypothesis**: that transformer middle layers encode a format-agnostic semantic substrate useful for product classification across text registers. The claim is that middle layers learn an approximate "protocol-level" representation stripped of surface-form serialization -- meaning the same product described as marketing copy, a patent claim, a regulatory filing, a social media post, or a news article should converge in representational geometry at middle layers, while diverging at early (encoding) and late (decoding) layers.

The experiment tests three nested hypotheses: **H1 (Phase Structure)** -- same-product representations converge in middle layers regardless of register; **H2 (Content Dominance)** -- semantic content (product/category identity) is more geometrically prominent than surface register in middle layers; **H3 (Protocol Layer Advantage)** -- middle-layer representations outperform early, late, and output layers for product classification. Primary analysis uses Representational Similarity Analysis (RSA) on pairwise distances across ~800 stimuli (40 real + 40 fictional products x 5 registers x 2 variants). Secondary analysis uses 40-class product, 8-class category, and 5-class register linear probes. Models: Qwen3.5-27B (4-bit + FP16 subset) and Llama-3.1-8B.

The design includes four experimental controls -- fictional products (memorization control), multi-source generation (generator confound control), full-precision subset (quantization control), and anisotropy correction (measurement artifact control) -- plus methodological safeguards recommended by the domain analysis: control tasks for probes (Hewitt & Manning 2019), partial RSA for confound control, within-category product discrimination, effect-size criteria alongside significance tests, and permutation-based statistical testing.

## Problem Statement

Current evidence from RYS layer-duplication experiments (Ng 2026), cross-lingual alignment (Liu & Niehues, ACL 2025), and bell-shaped anisotropy profiles (Tyshchuk et al. 2024) suggests that transformer middle layers learn representations that are more abstract and less tied to surface form than early or late layers. However, this has not been systematically tested for within-language register variation using controlled stimulus sets with known semantic equivalence. This experiment provides that test using product descriptions -- a domain where the same factual content naturally appears across multiple text registers, enabling controlled manipulation of surface form while holding semantic content constant.

The experiment must discriminate the "protocol layer" interpretation (middle layers encode a format-agnostic semantic substrate) from two alternative explanations: (1) **topic modeling** -- middle layers merely encode coarse category membership (oral care, pet food, etc.) which is trivially register-invariant; and (2) **lexical abstraction** -- middle layers map synonym sets to similar representations without any deeper "protocol" structure. The 40-class product probe (vs. 8-class category probe) and within-category discrimination index in RSA are the primary tools for this discrimination.

## Proposed Approach

A five-stage pipeline processes stimuli through transformer models and analyzes the representational geometry of hidden states:

1. **Stimulus Generation**: Create ~800 product descriptions (40 real + 40 fictional products x 5 registers x 2 paraphrase variants) using Claude as primary generator, with a 10-product subset crossed with GPT-4 for generator confound testing.
2. **Hidden State Extraction**: Run each stimulus through Qwen3.5-27B (4-bit primary, FP16 subset control) and Llama-3.1-8B, capturing per-layer residual stream, attention output, and MLP output via forward hooks. Mean-pool across tokens to produce one vector per stimulus per layer.
3. **Similarity Analysis**: Compute Representational Dissimilarity Matrices (RDMs) at each layer, correlate with theoretical model RDMs via RSA (primary), and compute per-layer cosine similarity by condition (same-product-different-register, same-category-different-product, different-category).
4. **Linear Probes**: Train L2-regularized logistic regression probes (40-class product, 8-class category, 5-class register) at each layer with 5-fold stratified CV. Include control tasks with permuted labels for selectivity measurement.
5. **Statistical Testing & Reporting**: Evaluate pre-registered falsification criteria for H1/H2/H3, run all control analyses, generate visualizations, and produce a go/no-go verdict.

Each stage produces well-defined artifacts (JSON, HDF5, NPY, CSV) and can be re-run independently given its inputs, enabling iterative refinement.

## Architecture

### Pipeline Design

```
Stage 1: Stimulus Generation  -->  Stage 2: Hidden State Extraction  -->  Stage 3: RSA + Probes
     (JSON output)                    (HDF5 output)                         (NPY/CSV output)
                                                                                    |
                                                                                    v
                                                          Stage 4: Statistical Testing  -->  Stage 5: Reporting
                                                              (CSV/JSON output)              (PNG/PDF + Markdown)
```

### Data Flow

- **Stimuli**: `stimuli.json` -- 800+ stimulus objects with product ID, category, register, text, token count, core attributes, generator metadata.
- **Hidden States**: HDF5 archives per model config, containing mean-pooled representations at each layer for residual stream, attention output, and MLP output. Structure: `/hidden_states_mean_no_special` (N x L+1 x D), `/attention_output_mean_no_special` (N x L x D), `/mlp_output_mean_no_special` (N x L x D).
- **Analysis Outputs**: Per-layer RSA correlations, permutation test null distributions, condition similarities, probe F1 scores with bootstrap CIs -- all saved as NPY/JSON.

### Storage Budget

| Data | Size |
|------|------|
| Stimuli JSON | < 1 MB |
| Hidden states (all models, all components, gzipped HDF5) | ~2 GB |
| Similarity matrices + probe results | ~1 GB |
| Figures and reports | < 1 GB |
| Model downloads (cached) | ~84 GB |
| **Total experimental data** | **~4 GB** |
| **Total including model cache** | **~88 GB** |

### Project Directory Structure

```
protocol-layer-hypothesis/
├── pyproject.toml
├── .env.example                    # ANTHROPIC_API_KEY=sk-ant-your-key-here
├── .gitignore                      # data/, .env, *.h5, __pycache__
├── config/
│   ├── default.yaml                # Full experiment config
│   └── debug.yaml                  # Small-scale debug (2 products, 2 registers)
├── src/plh/
│   ├── __init__.py
│   ├── config.py                   # Pydantic config loading + validation
│   ├── constants.py                # 80 products, 8 categories, 5 registers
│   ├── stage1_stimuli/             # Generation, prompts, validation, schema
│   ├── stage2_extraction/          # Model loading, hooks, pooling, extraction pipeline
│   ├── stage3_similarity/          # RDM, RSA, anisotropy, cosine conditions
│   ├── stage4_probes/              # Probe training, zone classification, control tasks
│   ├── stage5_analysis/            # Hypothesis tests, controls, go/no-go
│   ├── visualization/              # Phase plots, probe curves, RSA heatmaps
│   └── utils/                      # Seeds, I/O, checkpointing
├── scripts/                        # CLI entry points per stage + run_all.py
├── tests/                          # Unit tests (no GPU/API required)
├── data/                           # Git-ignored generated artifacts
└── notebooks/                      # Optional exploratory analysis
```

## Implementation Plan

### Phase 1: Foundation & Scaffolding
> Project setup, configuration system, product catalog, and test infrastructure.

#### Step 1.1: Project Initialization
- **Files**: `pyproject.toml`, `.gitignore`, `.env.example`, `src/plh/__init__.py`
- **Description**: Create the project skeleton with all dependencies pinned. Dependencies: torch>=2.2, transformers>=4.40, accelerate>=0.28, auto-gptq>=0.7 (or autoawq>=0.2), h5py>=3.10, numpy>=1.26, scipy>=1.12, scikit-learn>=1.4, matplotlib>=3.8, pydantic>=2.6, anthropic>=0.40, openai>=1.12, pyyaml>=6.0, tqdm>=4.66, pandas>=2.2. Dev deps: pytest>=8.0, pytest-cov>=4.0, ruff>=0.3.
- **Acceptance Criteria**: `pip install -e ".[dev]"` succeeds. `python -c "import plh"` works.
- **Risks**: Torch version must support the actual GPU. **Mitigation**: Check `nvidia-smi` output for GPU model. If RTX 3080 (10GB VRAM) instead of RTX 5090, switch primary model to Qwen2.5-7B or Llama-3.1-8B-only. Torch for RTX 5090 requires >=2.5 with CUDA 12.x.
- **Depends on**: none

#### Step 1.2: Configuration System
- **Files**: `src/plh/config.py`, `config/default.yaml`, `config/debug.yaml`
- **Description**: Implement Pydantic-validated config loading from YAML with CLI override support. Config covers: experiment metadata (name, seed, output_dir), stimuli params, model specs (primary/fp16/validation), extraction params, similarity params, probe params, zone definitions, analysis thresholds. Debug config uses tiny model (Qwen2.5-1.5B), 2 categories, 2 products/category.
- **Acceptance Criteria**: `load_config('config/debug.yaml')` parses without error. All nested Pydantic models validate. Invalid config raises clear error.
- **Risks**: None significant.
- **Depends on**: Step 1.1

#### Step 1.3: Product Catalog
- **Files**: `src/plh/constants.py`
- **Description**: Define all 40 real products (8 categories x 5) and 40 fictional products with Category enum, Register enum, Product dataclass (id, name, category, is_fictional, core_attributes[3-5], distinguishing_features). Define REGISTER_SPECS with voice, tone, structure, vocabulary, example_source for each register. Define CROSS_GENERATOR_SUBSET_IDS (10 products: 5 real, 5 fictional across 5 categories). Fictional products must have plausible but non-existent names with novel feature combinations.
- **Acceptance Criteria**: 40 real + 40 fictional products defined. No two products in the same category share >1 core attribute. Fictional names pass a plausibility check. Every product has 3-5 specific, quantitative core attributes.
- **Risks**: Fictional product names might accidentally match real products. **Mitigation**: Verify with web search during creation.
- **Depends on**: Step 1.1

#### Step 1.4: Utility Modules
- **Files**: `src/plh/utils/seeds.py`, `src/plh/utils/io.py`, `src/plh/utils/checkpoint.py`
- **Description**: (a) `seeds.py`: `set_global_seed()` for Python random, numpy, torch (CPU+CUDA), with deterministic cuDNN. (b) `io.py`: HDF5 validation (check expected keys, NaN/Inf detection). (c) `checkpoint.py`: `StageCheckpoint` class with pickle-based save/load partial results for crash recovery.
- **Acceptance Criteria**: `set_global_seed(42)` produces deterministic numpy random output. HDF5 validator detects NaN in test file. Checkpoint save/load roundtrips correctly.
- **Risks**: None significant.
- **Depends on**: Step 1.1

#### Step 1.5: Test Infrastructure
- **Files**: `tests/conftest.py`, `tests/test_config.py`
- **Description**: Create shared fixtures (tiny_hidden_states: 10x4x8, mock_stimuli: 10 stimuli across 2 categories x 2 products x 2 registers, mock_dataset). Write config loading tests. All tests run without GPU, API keys, or network.
- **Acceptance Criteria**: `pytest tests/test_config.py -v` passes.
- **Risks**: None significant.
- **Depends on**: Steps 1.2, 1.3, 1.4

### Phase 2: Stimulus Generation
> Generate ~800 primary stimuli plus ~200 cross-generator stimuli. Validate semantic anchoring.

#### Step 2.1: Stimulus Schema
- **Files**: `src/plh/stage1_stimuli/schema.py`
- **Description**: Pydantic models for `Stimulus` (stimulus_id, product_id, product_name, category, register, variant, text, token_count, core_attributes_present, is_fictional, generator, generation_model, generation_timestamp) and `StimulusDataset` (version, generation_date, config_hash, stimuli list, metadata dict). Token count validator warns at [60,80] and [150,200], fails outside [40,250].
- **Acceptance Criteria**: Schema validates well-formed stimuli. Rejects token_count outside [40,250]. JSON roundtrip preserves all fields.
- **Risks**: None significant.
- **Depends on**: Step 1.3

#### Step 2.2: Prompt Templates
- **Files**: `src/plh/stage1_stimuli/prompts.py`
- **Description**: `build_generation_prompt(product, register, variant_index)` constructs a prompt that instructs the LLM to generate a product description in the target register, grounded in core attributes. Constraints: 80-150 tokens strict, all core attributes must be conveyed, no product name as heading, no meta-commentary. Variant 0 vs 1 get different diversity instructions. Also implement `build_batch_generation_prompt()` as efficiency fallback.
- **Acceptance Criteria**: Prompts include all core attributes, register specs, and variant instruction. Output is a well-formed string.
- **Risks**: Register-specific prompts may produce texts that converge in style rather than diverging. **Mitigation**: Include explicit register specs (voice, tone, structure, vocabulary, example source) in each prompt.
- **Depends on**: Step 1.3

#### Step 2.3: Semantic Validation
- **Files**: `src/plh/stage1_stimuli/validate.py`
- **Description**: `validate_stimulus(stimulus, product)` checks: (1) token count within target range using the primary model's tokenizer (lazy-loaded); (2) core attribute coverage via keyword extraction and fuzzy matching; (3) register-appropriate formatting. `count_tokens()` uses the target model's tokenizer for accurate counts. `_extract_key_terms()` extracts numbers and multi-word technical phrases from attribute strings.
- **Acceptance Criteria**: Detects missing core attributes in intentionally deficient test stimuli. Token counting matches target model's tokenizer.
- **Risks**: Fuzzy keyword matching may miss paraphrased attributes. **Mitigation**: This is a quality filter, not a hard gate. Manual spot-checking of 5% of stimuli supplements automated validation. Consider optional LLM-as-judge verification for core attribute coverage (Domain Expert recommendation).
- **Depends on**: Steps 2.1, 1.3

#### Step 2.4: Generation Orchestrator
- **Files**: `src/plh/stage1_stimuli/generate.py`, `scripts/run_stage1.py`
- **Description**: `generate_all_stimuli(config)` generates all stimuli: (a) Primary: 80 products x 5 registers x 2 variants = 800 via Claude Sonnet; (b) Cross-generator: 10-product subset x 5 registers x 2 variants x GPT-4 = 100 additional. Uses checkpoint for crash recovery (save every 20 stimuli). Validates all stimuli, retries failures up to 3 times. Uses `anthropic.Anthropic()` with ANTHROPIC_API_KEY, temperature=0.7, max_tokens=300. Script entry: `run_stage1.py --config <path>`.
- **Acceptance Criteria**: `stimuli.json` contains 800+ primary stimuli. Cross-generator subset contains 100+ additional stimuli. Every stimulus has token_count in [60,200] (target [80,150]). Every stimulus passes validation with at most warnings.
- **Risks**: API rate limits (LOW -- ~800 calls is trivial for Anthropic tier). Semantic drift across variants (MEDIUM). **Mitigation**: Rate limits: exponential backoff with max_retries=5. Semantic drift: fact sheet anchoring via core_attributes + post-generation validation. Budget 30-60 min for manual spot-check of 5% sample.
- **Depends on**: Steps 2.1, 2.2, 2.3, 1.4

#### Step 2.5: Stimulus Generation Tests
- **Files**: `tests/test_stage1/test_prompts.py`, `tests/test_stage1/test_schema.py`, `tests/test_stage1/test_validate.py`
- **Description**: Test prompt generation produces valid strings with required elements. Test schema validation catches bad inputs. Test validation detects missing attributes. Mock all API calls and tokenizer.
- **Acceptance Criteria**: `pytest tests/test_stage1/ -v` passes. No API or network calls in tests.
- **Risks**: None.
- **Depends on**: Steps 2.1-2.4

### Phase 3: Hidden State Extraction
> Extract per-layer representations from all models with component decomposition. Critical VRAM validation gate.

#### Step 3.1: VRAM Validation Gate
- **Files**: None (manual + script verification)
- **Description**: Before any extraction, run `nvidia-smi` to verify GPU model and VRAM. If RTX 5090 (32GB): proceed as planned. If RTX 3080 (10GB): switch primary model to Qwen2.5-7B (4-bit, ~4GB) or Llama-3.1-8B-only, drop FP16 subset entirely. Then run a 1-stimulus pilot for the primary model: verify `output_hidden_states=True` returns correct layer count and shapes, verify forward hooks fire on `DecoderLayer`, `self_attn`, and `mlp` modules, verify no NaN/Inf in captured tensors.
- **Acceptance Criteria**: Pilot produces hidden states with shape (1, L+1, D) for residual stream and (1, L, D) for attention/MLP. No NaN/Inf. Wall-clock time per stimulus measured.
- **Risks**: `output_hidden_states` or hooks incompatible with GPTQ/AWQ quantized model (MEDIUM-HIGH risk, CRITICAL impact). **Mitigation**: If `output_hidden_states` fails, register hooks on `model.model.layers[i]` as fallback. If hooks on `self_attn`/`mlp` fail on quantized layers, fall back to residual-stream-only extraction (defer component decomposition to Tier 3). If model loading fails entirely, switch to AWQ variant or bitsandbytes 8-bit.
- **Depends on**: Step 1.1 (for model dependencies)

#### Step 3.2: Model Loading
- **Files**: `src/plh/stage2_extraction/models.py`
- **Description**: `load_model_and_tokenizer(spec)` loads model per ModelSpec config. Handles: GPTQ-4bit (via transformers GPTQ integration), FP16 with device_map="auto" (CPU offloading), standard FP16. Sets pad_token = eos_token if missing. Returns model in eval mode. Helper functions: `get_layer_count()`, `get_hidden_dim()`. **Critical note**: Verify actual HuggingFace model repo names at implementation time. "Qwen3.5-27B" may need to be "Qwen/Qwen2.5-27B-Instruct-GPTQ-Int4" or similar.
- **Acceptance Criteria**: Model loads without error. `model.eval()` applied. Layer count and hidden dim match expected values.
- **Risks**: HuggingFace model name mismatch. **Mitigation**: Search HuggingFace for correct repo name before downloading. Pin exact revision hash in config.
- **Depends on**: Step 1.2

#### Step 3.3: Forward Hooks for Component Decomposition
- **Files**: `src/plh/stage2_extraction/hooks.py`
- **Description**: `ExtractionHooks` class registers forward hooks on every transformer layer to capture attention output and MLP output. Supports both Qwen2 and Llama architectures (both use `model.model.layers[i].self_attn` and `model.model.layers[i].mlp`). Hooks move captured tensors to CPU immediately via `.detach().cpu()`. Handles tuple outputs from attention modules. Architecture-agnostic layer/module discovery via `_get_transformer_layers()`, `_get_attn_module()`, `_get_mlp_module()` with fallbacks.
- **Acceptance Criteria**: Hooks fire on every layer of a test model. Captured attention and MLP tensors have correct shapes. Hooks work correctly on CPU-offloaded layers (mixed device).
- **Risks**: Hooks on individual `Linear` layers inside quantized blocks may not work. **Mitigation**: Register hooks on `DecoderLayer`, `self_attn`, and `mlp` module levels (not on individual Linear layers). These wrapper modules are preserved even under GPTQ/AWQ quantization.
- **Depends on**: Step 3.2

#### Step 3.4: Pooling Strategies
- **Files**: `src/plh/stage2_extraction/pooling.py`
- **Description**: `mean_pool_no_special(hidden_states, attention_mask, special_ids, input_ids)` -- mean pool excluding BOS/EOS/PAD tokens. `last_token_pool(hidden_states, attention_mask)` -- extract last non-padding token's representation. Both handle batched input.
- **Acceptance Criteria**: Mean pooling correctly excludes special tokens (verified on synthetic input). Last-token pooling selects correct position. Both produce (batch, hidden_dim) output.
- **Risks**: Mean pooling sensitive to sequence length (Domain Expert concern). **Mitigation**: Record token count per stimulus. After extraction, check correlation between token count and hidden state norm. Report both pooling strategies; use mean pooling as primary, last-token as robustness check (Domain Expert recommendation #6).
- **Depends on**: Step 1.1

#### Step 3.5: Extraction Pipeline
- **Files**: `src/plh/stage2_extraction/extract.py`, `scripts/run_stage2.py`
- **Description**: `extract_hidden_states(config, stimuli, model_key)` processes all stimuli through the specified model at batch_size=1. For each stimulus: tokenize, forward pass with hooks, pool each layer's hidden states, save incrementally to HDF5. Uses resizable HDF5 datasets with gzip compression (level 4). Checkpoint every 50 stimuli. Resume support: checks which stimulus_ids already exist in HDF5. OOM handling: catch CUDA OOM, log stimulus, clear cache, skip and continue. GPU memory cleanup after each forward pass.
- **Acceptance Criteria**: HDF5 output contains all expected stimuli. Shapes correct: (N, L+1, D) for hidden_states, (N, L, D) for attention/MLP. No NaN/Inf. Checkpoint/resume works (kill and restart produces same output). GPU VRAM stays under 30GB during extraction.
- **Risks**: OOM on Qwen FP16 run (HIGH likelihood, MEDIUM impact). Disk write bottleneck (MEDIUM likelihood, LOW impact). **Mitigation for OOM**: Set 16GB swap file, close all apps, test with 5 stimuli first. If >30 sec/stimulus, reduce subset to 100. If infeasible, use 8-bit quantization as intermediate control or skip (Tier 4 item). **Mitigation for I/O**: Mean-pool during extraction (not after). Do NOT store raw token-level hidden states. Write incrementally to HDF5.
- **Depends on**: Steps 3.2, 3.3, 3.4, 2.4

#### Step 3.6: Extraction Tests
- **Files**: `tests/test_stage2/test_hooks.py`, `tests/test_stage2/test_pooling.py`, `tests/test_stage2/test_extract.py`
- **Description**: Test hooks on a tiny 2-layer random transformer model. Verify hook outputs match `output_hidden_states` for residual stream. Test pooling on synthetic tensors. Test HDF5 save/load roundtrip and checkpoint resume. No GPU required -- use CPU-only tiny model.
- **Acceptance Criteria**: `pytest tests/test_stage2/ -v` passes.
- **Risks**: None.
- **Depends on**: Steps 3.3, 3.4, 3.5

### Phase 4: Similarity Analysis (RSA)
> Primary analysis: RSA on pairwise distances. Anisotropy correction with numerical stability.

#### Step 4.1: Anisotropy Correction
- **Files**: `src/plh/stage3_similarity/anisotropy.py`
- **Description**: `correct_anisotropy(representations, method, n_components)` supports three methods: "none" (passthrough), "mean_centering" (subtract global mean), "whitening" (PCA-whitening via sklearn with truncation). **Critical numerical stability**: With 800 samples in 3584 dimensions, covariance matrix is rank-deficient (rank <= 799). Use `sklearn.decomposition.PCA(n_components=min(N-1, D), whiten=True)` which handles truncation internally. Add epsilon=1e-8 regularization. Warn if retained variance < 95%. Monitor condition number per layer. `correct_anisotropy_all_layers()` applies correction independently at each layer.
- **Acceptance Criteria**: Mean centering produces zero-mean output. Whitening produces unit-variance, uncorrelated components. No NaN/Inf in output. Handles rank-deficient case (N < D) correctly. Whitened output dimensionality = min(N-1, D).
- **Risks**: Whitening amplifies noise dimensions (MEDIUM likelihood, MEDIUM impact). Near-degenerate covariance at some layers. **Mitigation**: PCA-whitening with truncation (retain components with eigenvalue > epsilon). Monitor condition number per layer, flag layers where it exceeds 1e6. Report both corrected and uncorrected results (Decision #9) -- if they diverge, this is itself an informative finding.
- **Depends on**: Step 1.4

#### Step 4.2: Representational Dissimilarity Matrices
- **Files**: `src/plh/stage3_similarity/rdm.py`
- **Description**: `compute_rdm(representations, metric)` computes NxN dissimilarity matrix using scipy's `pdist`+`squareform` with cosine distance. `compute_rdms_all_layers()` applies per-layer. `build_model_rdm(stimulus_ids, product_ids, category_ids)` constructs the theoretical model RDM encoding the hypothesis: same-product=0, same-category=0.5, different-category=1.0. Also build a **register-identity model RDM** (Domain Expert recommendation for H2 RSA test): same-register=0, different-register=1. Build a **within-category product discrimination model RDM** (Domain Expert recommendation #5): only includes within-category pairs, predicting same-product=0, different-product-same-category=1.
- **Acceptance Criteria**: RDMs are symmetric with zero diagonal. Model RDM correctly encodes the three-level distance structure. Within-category model RDM correctly selects only same-category pairs.
- **Risks**: None significant computationally. ~320K pairwise distances per layer at 64 layers takes ~2 minutes total.
- **Depends on**: Step 1.4

#### Step 4.3: RSA Computation
- **Files**: `src/plh/stage3_similarity/rsa.py`
- **Description**: `rsa_correlation(observed_rdm, model_rdm, method)` computes Spearman rank correlation between upper triangles. `rsa_all_layers()` applies per-layer. `rsa_permutation_test(observed_rdm, model_rdm, n_permutations=10000)` shuffles model RDM labels to generate null distribution. Also implement **partial RSA** (Domain Expert critical recommendation #2): `partial_rsa(observed_rdm, model_rdm, nuisance_rdms)` controls for stimulus length difference and lexical overlap (Jaccard distance on token sets) as nuisance regressors. Uses partial Spearman correlation: regress out nuisance RDMs from both observed and model RDM vectors, then correlate residuals.
- **Acceptance Criteria**: RSA on synthetic data with known structure recovers expected correlations. Permutation test produces p-values near 0 for strong effects and near 0.5 for null effects. Partial RSA reduces correlation when nuisance variable is confounded.
- **Risks**: Pairwise distances not independent (shared stimuli). **Mitigation**: Use permutation-based p-values throughout (not parametric). This handles non-independence correctly.
- **Depends on**: Step 4.2

#### Step 4.4: Condition Similarities
- **Files**: `src/plh/stage3_similarity/cosine.py`
- **Description**: `compute_condition_similarities()` classifies all pairs into SP-DR (same product, different register), DP-SC (different product, same category), and DC (different category), then computes mean cosine similarity per condition per layer. Subsample DC pairs to 5000 if count exceeds this (to manage computation). **Nuisance RDM construction**: Also compute stimulus-length-difference RDM and lexical-overlap RDM for use in partial RSA.
- **Acceptance Criteria**: Pair counts match expected combinatorics. Similarity values are in [-1, 1] range.
- **Risks**: None significant.
- **Depends on**: Steps 4.1, 4.2

#### Step 4.5: Similarity Analysis Entry Point & Tests
- **Files**: `scripts/run_stage3.py`, `tests/test_stage3/test_rdm.py`, `tests/test_stage3/test_rsa.py`, `tests/test_stage3/test_anisotropy.py`
- **Description**: Stage 3 script loads hidden states from HDF5, builds metadata index, runs RSA and condition similarities for each anisotropy correction method, saves results as NPY + JSON summaries. Tests verify all analysis components on synthetic data with known structure.
- **Acceptance Criteria**: RSA correlations computed for all layers x correction methods. Permutation test p-values computed at peak layers. All outputs saved. `pytest tests/test_stage3/ -v` passes.
- **Risks**: RAM exhaustion loading hidden states (HIGH likelihood, HIGH impact). **Mitigation**: Process layer-by-layer from HDF5 using chunked reads. Never load all layers simultaneously. Peak RAM for analysis: ~4 GB (one layer at a time + 51MB covariance matrix).
- **Depends on**: Steps 4.1-4.4, 3.5

### Phase 5: Linear Probes
> Secondary analysis with control tasks. Cross-validated probe training at every layer.

#### Step 5.1: Probe Training Pipeline
- **Files**: `src/plh/stage4_probes/train.py`
- **Description**: `train_probe_at_layer(representations, labels, ...)` trains L2-regularized logistic regression with 5-fold stratified CV. **Critical stratification**: Use `GroupKFold` with groups=product_id (not `StratifiedKFold`) to prevent product-level leakage -- all variants of one product must be in the same fold. **Nested CV for regularization** (Domain Expert recommendation #3): Within each outer fold, tune C over {0.01, 0.1, 1.0, 10.0, 100.0} via 3-fold inner CV. Use `lbfgs` solver, multinomial, max_iter=2000. Record macro-F1, per-fold F1, bootstrap 95% CIs. `train_probes_all_layers()` runs all three probe tasks (40-class product, 8-class category, 5-class register) at all layers.
- **Acceptance Criteria**: 40-class probe achieves above chance (>2.5%) at peak layer. Per-fold results reported. Bootstrap CIs non-degenerate. Stratification verified: no product appears in both train and test.
- **Risks**: 40-class probe with ~10 samples/class has high variance (HIGH likelihood, LOW impact since RSA is primary). Solver convergence warnings at some layers (MEDIUM). **Mitigation**: Report per-fold variance. Use RSA as primary evidence. Increase max_iter to 2000, use lbfgs solver.
- **Depends on**: Step 4.1

#### Step 5.2: Control Task Probes (Hewitt & Manning 2019)
- **Files**: `src/plh/stage4_probes/control_tasks.py`
- **Description**: For each real probe (product, category, register), train a matched control probe where labels are randomly permuted but class sizes are preserved. Compute **selectivity** = real accuracy - control accuracy at each layer. This is the single most important methodological addition per the Domain Expert. Without it, probe accuracy is uninterpretable given the high dimensionality of representations (~3584 dims) relative to sample size (~400).
- **Acceptance Criteria**: Control probes produce non-trivial accuracy (demonstrating probe capacity). Selectivity is positive at layers where real probes perform well. Selectivity is near zero at layers where real probes are at chance.
- **Risks**: None computationally. Doubles the number of probe fits (~15,360 total). Still manageable at ~0.5 sec each = ~2 hours.
- **Depends on**: Step 5.1

#### Step 5.3: Zone Classification
- **Files**: `src/plh/stage4_probes/zone_classifier.py`
- **Description**: `compute_zone_boundaries()` divides layers into early (0-10th percentile), protocol (10th-70th percentile), late (90th-99th percentile), output (final layer). `train_zone_probes()` trains probes on mean-pooled zone representations for all three tasks. This provides the H3 test statistic.
- **Acceptance Criteria**: Zone boundaries computed correctly for Qwen (64 layers) and Llama (32 layers). Zone probes produce F1 scores and CIs.
- **Risks**: None significant.
- **Depends on**: Step 5.1

#### Step 5.4: Probe Entry Point & Tests
- **Files**: `scripts/run_stage4.py`, `tests/test_stage4/test_train.py`, `tests/test_stage4/test_evaluate.py`
- **Description**: Stage 4 script loads hidden states, applies anisotropy correction, builds label arrays, trains per-layer and zone probes for all tasks and correction methods. Saves as JSON. Tests verify probe training on synthetic data with known class boundaries.
- **Acceptance Criteria**: `pytest tests/test_stage4/ -v` passes. Per-layer and zone results saved for all tasks.
- **Risks**: None beyond Step 5.1 risks.
- **Depends on**: Steps 5.1-5.3

### Phase 6: Statistical Testing & Reporting
> Pre-registered hypothesis evaluation, control analyses, visualization, go/no-go verdict.

#### Step 6.1: Hypothesis Tests
- **Files**: `src/plh/stage5_analysis/hypothesis_tests.py`
- **Description**: Implement pre-registered falsification criteria with Domain Expert refinements:
  - **H1 (Phase Structure)**: RSA correlation significantly higher at middle layers vs. early layers. Criterion: p < 0.05 (permutation test) AND Cohen's d > 0.3 (effect-size threshold per Domain Expert recommendation #4). Also test in RSA terms: peak product-identity RSA correlation at middle layer.
  - **H2 (Content Dominance)**: Register probe does not exceed category probe by >5pp at any protocol-zone layer. Also test in RSA terms: product-identity model RDM correlates more strongly than register-identity model RDM at middle layers.
  - **H3 (Protocol Layer Advantage)**: Best-performing layer for 40-class probe falls in middle 60% AND outperforms output layer by >=2pp. Also test in RSA terms: peak RSA correlation in middle 60% is significantly higher than RSA at output layer (permutation test).
  - **Multiple comparison correction**: Apply Benjamini-Hochberg FDR correction across layers for all per-layer tests.
- **Acceptance Criteria**: Tests correctly classify synthetic data known to support/falsify each hypothesis. Effect-size criteria applied alongside significance. FDR correction applied.
- **Risks**: Multiple comparisons eliminate marginal effects (MEDIUM). **Mitigation**: Pre-register primary layer selection. Use FDR (not Bonferroni). Report uncorrected results alongside corrected.
- **Depends on**: Steps 4.5, 5.4

#### Step 6.2: Control Analyses
- **Files**: `src/plh/stage5_analysis/controls.py`
- **Description**: (a) **Memorization control**: Compare RSA curves for real vs. fictional products. Pass criterion: Spearman r > 0.7 between curves. (b) **Quantization control**: Compare Qwen FP16 vs. 4-bit RSA curves. Pre-registered criterion: Spearman rho > 0.9 (Decision #12). (c) **Generator control**: Compare Claude vs. GPT-4 RSA at peak layer for multi-source subset. Test generator main effect with ANOVA on pairwise distances. (d) **Within-category product discrimination** (Domain Expert recommendation #5): Compute RSA with within-category-only model RDM. If protocol-layer hypothesis is correct, within-category discrimination peaks in same middle layers. If only topic modeling, within-category discrimination is flat.
- **Acceptance Criteria**: All controls produce quantified results. Fictional vs. real comparison computed. Quantization threshold evaluated.
- **Risks**: Qwen FP16 run may be deferred (Tier 4). **Mitigation**: Quantization control is deferrable. All other controls are Tier 1.
- **Depends on**: Steps 4.5, 5.4

#### Step 6.3: Go/No-Go Decision Logic
- **Files**: `src/plh/stage5_analysis/go_no_go.py`
- **Description**: Mechanical evaluation: GO = all three hypotheses supported + all critical controls passed. QUALIFIED_GO = H1+H2 supported + controls passed, H3 marginal. NO_GO = any hypothesis falsified or critical control failed. Each outcome includes summary explanation.
- **Acceptance Criteria**: Correctly classifies known pass/fail inputs.
- **Risks**: Results may be ambiguous/partially supporting (HIGH likelihood per Risk Assessment). **Mitigation**: Budget extra time for nuanced interpretation. The pre-registered criteria and multiple analytical angles (RSA, probes, decomposition) provide multiple lines of evidence.
- **Depends on**: Steps 6.1, 6.2

#### Step 6.4: Visualization
- **Files**: `src/plh/visualization/phase_plots.py`, `src/plh/visualization/probe_curves.py`, `src/plh/visualization/rsa_heatmaps.py`, `src/plh/visualization/style.py`
- **Description**: Generate core figures: (1) Three-condition similarity curves (SP-DR, DP-SC, DC) per layer with 95% CI bands. (2) RSA correlation per layer with significance markers. (3) Probe accuracy curves (product, category, register) per layer with chance lines and error bars. (4) Zone comparison bar charts. (5) Decomposition panels (attention/MLP/residual). (6) Quantization control overlay. (7) Memorization control overlay. (8) Full RDM heatmaps at selected layers (early, middle, late). (9) Register confusion matrices at selected layers (Domain Expert recommendation #8). Output: PNG (300 DPI) + PDF.
- **Acceptance Criteria**: All plots generate without error. Layer indices correct. Legend readable. Chance levels marked.
- **Risks**: None significant.
- **Depends on**: Steps 4.5, 5.4

#### Step 6.5: Reporting Entry Point & Tests
- **Files**: `scripts/run_stage5.py`, `scripts/run_all.py`, `scripts/validate_data.py`, `tests/test_stage5/test_hypothesis_tests.py`, `tests/test_stage5/test_controls.py`
- **Description**: Stage 5 script loads all analysis outputs, runs hypothesis tests, control analyses, generates all visualizations, produces final JSON report with verdict. `run_all.py` orchestrates the full pipeline sequentially. `validate_data.py` checks inter-stage data integrity.
- **Acceptance Criteria**: Final report JSON generated with verdict, hypothesis results, control results. All figures generated. `pytest tests/test_stage5/ -v` passes. `pytest tests/ -v` (full suite) passes.
- **Risks**: None beyond upstream risks.
- **Depends on**: Steps 6.1-6.4

## Risk Mitigations

| Risk | Severity | Likelihood | Mitigation | Phase/Step |
|------|----------|------------|------------|------------|
| `output_hidden_states` / hooks incompatible with quantized model | CRITICAL | MEDIUM | 1-stimulus pilot before any extraction. Fallback: hooks on `DecoderLayer`. If all fails: switch to bitsandbytes 8-bit. | 3.1 |
| Qwen FP16 OOM or impractically slow | HIGH | HIGH | 16GB swap file. Test with 5 stimuli. Fallback: 8-bit quant or skip (Tier 4 deferral). Run overnight as unattended batch. | 3.5 |
| Whitening numerical instability (rank deficiency) | HIGH | MEDIUM | PCA-whitening with truncation (n_components = min(N-1, D)). Epsilon=1e-8 regularization. Monitor condition number per layer. | 4.1 |
| RAM exhaustion during analysis | HIGH | HIGH | Process layer-by-layer from HDF5. Never load all layers simultaneously. Peak RAM ~4GB. | 4.5 |
| 40-class probe high variance (10 samples/class) | LOW | HIGH | RSA is primary (19,900+ pairwise distances). Probes are secondary confirmation. Report per-fold variance. | 5.1 |
| GPU is RTX 3080 (10GB) not RTX 5090 (32GB) | HIGH | LOW (verify first) | Check `nvidia-smi`. If 3080: use Qwen2.5-7B or Llama-3.1-8B only. | 3.1 |
| Timeline overrun (>20 hours) | MEDIUM | MEDIUM | Tier-based scope cutting: (1) defer FP16 subset, (2) defer component decomposition, (3) defer Llama cross-validation. Never cut: RSA, probes, fictional control. | All |
| Stimulus semantic drift across variants | MEDIUM | MEDIUM | Core attribute fact sheets. Automated validation. 5% manual spot-check. | 2.4 |
| Probe results uninterpretable without control tasks | HIGH | HIGH (if omitted) | Implement Hewitt & Manning control tasks (Step 5.2). Non-negotiable for publishable results. | 5.2 |
| Lexical confounds mimic protocol-layer effect in RSA | HIGH | MEDIUM | Partial RSA controlling for lexical overlap and stimulus length (Step 4.3). | 4.3 |

## Domain Requirements

### Critical Methodological Requirements (Mandatory)

1. **Control tasks for all probes** (Hewitt & Manning 2019): Train matched probes with permuted labels. Report selectivity = real accuracy - control accuracy. Without this, probe results are uninterpretable given high-dimensional representations. Implemented in Step 5.2.

2. **Partial RSA for confound control** (Kriegeskorte & Kievit 2013): Include nuisance model RDMs for stimulus length difference and lexical overlap (Jaccard distance on token sets). For multi-source subset, include generator identity as additional nuisance regressor. Implemented in Step 4.3.

3. **Within-category product discrimination index**: The key test distinguishing the protocol-layer hypothesis from topic modeling. Compute RSA with within-category-only model RDM. If discrimination peaks in middle layers, this supports fine-grained semantic identity encoding beyond coarse topic membership. Implemented in Step 6.2.

4. **Effect-size criteria alongside significance**: With 19,900+ pairwise distances, even trivial effects are significant. Require Cohen's d > 0.3 for H1 and RSA r > 0.1 at peak layer. Implemented in Step 6.1.

5. **GroupKFold stratification**: Use product_id as group for CV splits. All variants of one product must be in the same fold to prevent data leakage. Implemented in Step 5.1.

6. **Nested CV for regularization**: Tune C over {0.01, 0.1, 1.0, 10.0, 100.0} via 3-fold inner CV rather than fixing C=1.0. Implemented in Step 5.1.

### Recommended Additions (Important but Optional)

7. **Both mean pooling and last-token pooling**: Report convergence/divergence between strategies. Mean pooling primary; last-token as robustness check.

8. **Logit lens / tuned lens analysis**: Project each layer's hidden state through the final unembedding matrix to see if middle layers predict product-related tokens more strongly. Converging evidence from a different analytical angle.

9. **CKA (Centered Kernel Alignment)** as robustness check alongside RSA: Invariant to orthogonal transformation and isotropic scaling, potentially more robust to anisotropy.

10. **Register confusion matrices** at selected layers: Reveals which register pairs are confusable at which processing stages.

11. **Intrinsic dimensionality estimate per layer**: Participation ratio or effective rank to characterize the geometry.

### Reporting Standards

Per the domain analysis (Belinkov 2022, RSA literature), the experiment must report:

- Probe architecture and hyperparameters (including cross-validated C)
- Control task accuracy (selectivity) at each layer
- Per-layer accuracy curves with 95% bootstrap CIs
- Number of probe parameters vs. training examples
- Stratification strategy
- Distance metric used for RSA
- Model RDM construction (explicit)
- Permutation test with number of permutations stated
- Multiple comparison correction method
- Both corrected and uncorrected anisotropy results
- All stimuli released (JSON with product attributes)
- Exact model versions and quantization details
- Random seeds for all stochastic processes

## Trade-offs & Decisions

| Decision | Chose | Over | Rationale |
|----------|-------|------|-----------|
| RSA vs. probes as primary | RSA primary, probes secondary | Probes primary | 19,900+ pairwise distances give massive statistical power vs. 5-10 samples/class for probes (Decision #7) |
| Forward hooks vs. `output_hidden_states` | Forward hooks | `output_hidden_states` | Enables component decomposition, memory control, robust across quantization backends (Architect) |
| HDF5 vs. NPY archives | HDF5 with gzip | Individual NPY files | Chunked I/O, metadata support, single file per run, compression (Architect) |
| Batch size 1 vs. dynamic batching | Batch size 1 | Larger batches | Simplicity (no padding/mask complexity), negligible throughput difference for 27B model (Implementer, Risk Analyst) |
| 40-class primary vs. 8-class | 40-class primary, 8-class secondary | 8-class primary | 8 categories trivially separable; 40-class tests fine-grained product identity (Decision #1) |
| Fictional products | 40 fictional, separate analysis | No fictional / mixed analysis | Clean memorization control without contaminating main results (Decisions #2, #10) |
| Fixed C=1.0 vs. nested CV | Nested CV over {0.01..100.0} | Fixed C=1.0 | Domain Expert recommendation; prevents regularization from being a hidden confound |
| Token range 80-150 strict vs. flexible | Flexible 50-200 with length as covariate | Strict 80-150 | Domain Expert recommendation; strict range unnaturally constrains registers. Partial RSA controls for length. |
| Mean pooling vs. last-token | Mean pooling primary, both reported | Last-token only | Mean pooling standard in probing literature; last-token as robustness check (Domain Expert) |

## Specialist Conflicts Resolved

| Conflict | Resolution | Rationale |
|----------|------------|-----------|
| **Architect: forward hooks; Risk Analyst: hooks may fail on quantized layers** | Use hooks with validation gate (Step 3.1). Fallback: `output_hidden_states` for residual-only if hooks fail. | Hooks provide essential component decomposition. 1-stimulus pilot validates before committing. Risk Analyst's fallback path preserved. |
| **Architect: batch_size=4 in default.yaml; Risk Analyst: batch_size=1 recommended** | Default to batch_size=1. Config supports larger batches for profiling. | Risk Analyst's analysis shows batch_size=1 is adequate (7 min for 800 stimuli on Qwen 4-bit). Padding complexity of batching not worth marginal speedup. Updated default.yaml accordingly. |
| **Implementer: C=1.0 fixed; Domain Expert: nested CV for C** | Nested CV. | Domain Expert's recommendation is well-grounded in Belinkov (2022). Minimal computational cost (~2x). Prevents regularization as hidden confound. |
| **Implementer: StratifiedKFold; Architect/Domain Expert: GroupKFold** | GroupKFold with groups=product_id. | StratifiedKFold could split product variants across folds, causing data leakage. GroupKFold ensures all variants of a product stay together. |
| **Architect: 80-150 token strict range; Domain Expert: 50-200 with length covariate** | Allow 50-200, include length as partial RSA nuisance regressor. | Domain Expert correctly notes that enforcing 80-150 unnaturally constrains registers (tweets run 20-50, patents 150-300). Wider range preserves register authenticity; partial RSA controls for length confound. |
| **Implementer: extraction time 30-45 min total; Risk Analyst: 4-8 hours possible** | Budget 1-3 hours, with FP16 run as overnight batch. | Implementer's estimate assumes optimal conditions with RTX 5090. Risk Analyst's estimate includes FP16 CPU offloading. Realistic middle ground: 4-bit and Llama runs fast (~30 min), FP16 run slow (1-6 hours, overnight). |
| **Architecture: 5 stages; Implementation: 6 stages (adds Stage 0)** | 6 implementation groups within 5 pipeline stages. Stage 0 (scaffolding) is development setup, not a pipeline stage. | Both are correct at different levels. The pipeline has 5 runtime stages. Implementation has an additional setup phase. |
| **Architect: permutation test at peak layer only; Domain Expert: permutation test with cluster correction** | Permutation test at all layers with FDR correction, plus peak-layer permutation test with full null distribution. | Domain Expert's recommendation for multiple comparison correction is essential. FDR across layers catches false positives while being less conservative than Bonferroni. |
| **Risk Analyst: Tier 4 deferral of FP16 run; Implementer: FP16 as standard model config** | FP16 is Tier 4 (deferrable). 4-bit primary, Llama-3.1-8B as cross-model validation (Tier 2). | Risk Analyst correctly identifies FP16 as the highest schedule risk. Its value (quantization control) is important but subordinate to the core analysis. Spearman > 0.9 pre-registered threshold can be evaluated later. |

## Open Questions

1. **Exact HuggingFace model IDs**: "Qwen3.5-27B" may not exist as a public model. At implementation time, search for the correct repo name (likely "Qwen/Qwen2.5-27B-Instruct-GPTQ-Int4" or "Qwen/Qwen2.5-Coder-27B-GPTQ-Int4" or similar). Update config accordingly.

2. **AWQ vs. GPTQ**: Both are 4-bit quantization methods. AWQ is generally faster; GPTQ has broader compatibility. The choice depends on which has better transformers integration for the target model at implementation time.

3. **RTX 5090 vs RTX 3080**: The user's machine profile says RTX 3080 (10GB VRAM) but the experiment design assumes RTX 5090 (32GB). The coding agent MUST check `nvidia-smi` before model selection. If 3080: Qwen 27B will NOT fit. Switch to Qwen2.5-7B (4-bit, ~4GB) or run Llama-3.1-8B only.

4. **Human stimuli for cross-generator subset**: Designated as a stretch goal. The pipeline should support it structurally but not block on it. Claude + GPT-4 provide sufficient generator diversity for the control analysis.

5. **Logit lens analysis**: Recommended by Domain Expert as converging evidence but not in the core pipeline. Could be added as a Phase 7 extension if core results are positive.

6. **Causal evidence (activation patching)**: If all hypotheses are supported, the correlational RSA evidence does not establish that the model *uses* these representations. Activation patching (Meng et al. 2022) would provide causal evidence. This is a follow-up experiment, not part of the current scope.

7. **Token range policy**: The resolution allows 50-200 tokens with length as a covariate. The generation prompts still target 80-150 as the preferred range. Should the prompt be updated to explicitly allow wider variance, or should the wider range only serve as a post-hoc acceptance criterion?

## Execution Schedule

Based on Risk Analyst's tiered execution plan:

### Evening 1 (4 hours): Stimulus Generation + Pilot Validation
- Generate all ~800+ stimuli via API
- Quality-check 5% sample manually
- Download and load primary model (Qwen 4-bit)
- **Critical gate**: Run 1-stimulus pilot for hooks and hidden states
- If pilot passes: begin Qwen 4-bit extraction (~15 min)
- Save all hidden states to disk

### Evening 2 (4 hours): Llama Extraction + Analysis Code
- Llama-3.1-8B extraction (~20 min)
- Write and test RSA analysis code (layer-by-layer, memory-mapped)
- Write and test PCA-whitening with regularization
- Run RSA on Qwen 4-bit residual stream (corrected + uncorrected)
- Generate initial phase structure plots -- **first look at H1**

### Evening 3 (4 hours): Probes + Component Decomposition
- Run linear probes on Qwen 4-bit (all three tasks, all layers)
- Run control task probes (Hewitt & Manning)
- Run RSA on Llama (corrected + uncorrected)
- If time: Qwen 4-bit extraction with hooks for component decomposition
- Compare Qwen vs. Llama RSA profiles

### Evening 4 (4-5 hours): Deep Analysis + FP16 Control
- Component-decomposed RSA and probes (if extracted)
- Within-category product discrimination analysis
- Multi-source stimulus analysis (generator effects)
- Start Qwen FP16 subset run (overnight batch)
- Statistical testing, FDR corrections
- Begin writing results

### Evening 5 (3-4 hours, if needed): FP16 Analysis + Final Report
- Analyze FP16 results (Spearman correlation with 4-bit)
- Partial RSA with nuisance regressors
- Finalize all plots and tables
- Write conclusions
- Archive all data and code

### Minimum Viable Results (MVR) -- 8-10 hours
If time is constrained, the Tier 1 deliverables test all three hypotheses with one model:
1. Full 800 stimuli (real + fictional)
2. Qwen 4-bit extraction (residual stream only)
3. RSA analysis (corrected + uncorrected)
4. Linear probes (40-class, 8-class, 5-class) with control tasks
5. Phase structure plots, probe accuracy curves, hypothesis verdicts
