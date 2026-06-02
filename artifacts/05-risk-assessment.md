# Risk Assessment: The Protocol Layer Hypothesis

**Agent**: Risk Analyst
**Date**: 2026-03-25
**Scope**: Execution risks for building and running the experimental pipeline on a single RTX 5090 workstation (32GB VRAM, 32GB system RAM, WSL2), operated by one researcher across ~15-20 hours on 4-5 evenings.

---

## 1. Failure Mode Analysis by Pipeline Stage

### Stage 1: Stimulus Generation (~800 stimuli via Claude API + GPT-4/human subset)

| Failure Mode | Likelihood | Impact | Details |
|---|---|---|---|
| API rate limits throttle generation | LOW | LOW | Anthropic rate limits for Sonnet are generous (~4000 RPM on most tiers). 800 calls is trivial. Even at aggressive parallelism, unlikely to hit limits. |
| API cost overrun | LOW | LOW | ~800 Claude Sonnet calls at ~150 output tokens each. At current Sonnet pricing (~$3/M input, $15/M output), each call costs roughly $0.003. Total: ~$2.50 for Claude. GPT-4 subset (10 products x 5 registers x maybe 3-5 calls) adds ~$1-2. Total stimulus generation cost: under $10. Negligible. |
| Semantic drift across stimuli | MEDIUM | MEDIUM | When generating 10 variants per product (original 5 registers + 5 paraphrases/additional registers), later variants may drift from the core factual anchor. This is a quality issue, not a crash -- but it silently degrades the experimental signal. |
| GPT-4 / human stimuli incompatible with format | LOW | LOW | If the multi-source crossing uses different prompt structures, stimuli may differ in length or structure. Fixable with post-hoc editing. |
| API outage during generation | LOW | LOW | Generation is stateless and resumable. Save after each batch. |

**Net assessment**: Stimulus generation is the lowest-risk stage. The only real concern is quality control -- verifying that all 10 variants per product maintain semantic equivalence. Budget 30-60 minutes for manual spot-checking.

### Stage 2: Hidden State Extraction (3 model configs x ~800 stimuli)

This is the highest-risk stage. Every sub-risk below is real and has bitten practitioners.

| Failure Mode | Likelihood | Impact | Details |
|---|---|---|---|
| **OOM on Qwen3.5-27B 4-bit** | MEDIUM | HIGH | See Section 2 for detailed VRAM budget. The model alone is ~14GB. Hidden states for a single forward pass with `output_hidden_states=True` allocate all 64 layers simultaneously. For a 150-token input on a 3584-dim model with 64 layers, that is 150 x 64 x 3584 x 4 bytes = ~140MB per stimulus. With batch_size=1 this fits. With batch_size>4 it may not, depending on KV cache and activation memory. |
| **`output_hidden_states=True` incompatible with GPTQ/AWQ kernels** | MEDIUM | CRITICAL | This is the single highest-risk technical issue. Quantized models use custom CUDA kernels (e.g., `auto-gptq`, `autoawq`) that replace standard linear layers. Some implementations do not properly propagate hidden states through the quantized forward pass. The HuggingFace integration *should* handle this via the model's `forward()` method, but edge cases exist. **Must be validated in a 1-stimulus pilot before committing to the full run.** |
| **Forward hooks fail with quantized layers** | MEDIUM | HIGH | If using forward hooks (rather than `output_hidden_states`) to capture attention output, MLP output, and residual stream separately, hooks may not fire correctly on quantized `QuantLinear` layers. The hook registration point matters: hooks on the `DecoderLayer` (wrapping the full block) should work; hooks on individual `Linear` layers inside a quantized block may not. |
| **CPU offloading for FP16 Qwen3.5-27B fails or is impractically slow** | HIGH | MEDIUM | 27B params at FP16 = ~54GB. GPU has 32GB, system RAM has 32GB. That is 86GB total, but the OS and framework overhead consume 2-4GB of each. Realistically: 28GB VRAM usable + 28GB RAM usable = 56GB. The model barely fits. Hidden state extraction adds memory pressure. CPU-offloaded layers run 10-50x slower. For 800 stimuli, even at 2 seconds per stimulus on GPU-only, the offloaded layers could push this to 20-60 seconds per stimulus = 4.4-13.3 hours for 800 stimuli. **This may exceed a single evening.** |
| **Hidden state extraction for Llama-3.1-8B works but produces different tensor shapes** | LOW | MEDIUM | Llama-3.1-8B has 32 layers and 4096 hidden_dim vs. Qwen's 64 layers and 3584 hidden_dim. The analysis code must handle variable layer counts and dimensions. Not hard, but a source of index-off-by-one bugs. |
| **WSL2 GPU memory management differs from native Linux** | LOW | MEDIUM | WSL2's GPU memory management has historically had quirks with large allocations. The NVIDIA driver translates CUDA calls through a virtualization layer. In practice, modern WSL2 (6.6+ kernel) handles this well, but peak VRAM usage may be slightly higher than on native Linux due to overhead. |
| **Disk write bottleneck during state saving** | MEDIUM | LOW | Writing 28GB of hidden states per model config to disk takes time. If saving as `.npy` or `.pt` files synchronously after each forward pass, this adds I/O time. If accumulating in RAM first, it adds memory pressure. Use memory-mapped files or incremental writes. |

**Net assessment**: Hidden state extraction is the make-or-break stage. The two critical validations before committing time are: (1) Does `output_hidden_states=True` work with the specific quantized model checkpoint? (2) Do forward hooks fire correctly on `DecoderLayer` modules for component decomposition? Both require a 1-stimulus pilot that should take <15 minutes.

### Stage 3: Similarity Analysis (RSA on pairwise distances)

| Failure Mode | Likelihood | Impact | Details |
|---|---|---|---|
| **RAM exhaustion loading hidden state archives** | HIGH | HIGH | The full hidden state archive for one model config is ~28GB. System RAM is 32GB. Loading the entire archive into RAM for pairwise computation will OOM with OS + Python overhead. **Must use memory-mapped arrays or process layer-by-layer.** |
| **Pairwise distance computation is O(N^2) and slow** | MEDIUM | MEDIUM | 800 stimuli = 320,000 pairwise distances per layer. At 64 layers, that is 20.5M distance computations. With scipy's `pdist` on 3584-dim vectors, each layer takes ~1-2 seconds. Total: ~2 minutes. This is fine. But if doing this for 4 components (full, attn, MLP, residual) x 2 anisotropy variants x 3 model configs = 24 variants, it scales to ~48 minutes. Still manageable. |
| **Whitening is numerically unstable** | MEDIUM | HIGH | Per-layer whitening requires inverting the covariance matrix of 800 samples in 3584 dimensions. With 800 < 3584, the covariance matrix is rank-deficient (rank at most 800). Standard whitening via eigendecomposition will produce zero eigenvalues. **Must use regularized whitening (add small epsilon to diagonal) or PCA-based whitening (project to top-k components first).** Failure mode: NaN or Inf values silently propagate through downstream RSA computation. |
| **Anisotropy correction inflates noise dimensions** | MEDIUM | MEDIUM | As noted in the DA critique: if a few principal components dominate variance, whitening amplifies low-variance (noise) dimensions. This is a methodological risk, but manifests as an execution risk when the whitened representations produce nonsensical RSA scores. Mitigation: use PCA whitening with a variance-explained threshold (e.g., retain components explaining 95% of variance). |
| **Similarity matrix storage exceeds disk budget** | MEDIUM | LOW | 800x800 matrix x float32 = 2.56MB per layer. At 64 layers x 24 variants = ~4GB total. Manageable. The 13GB estimate in the project brief assumed larger matrices; actual size is modest. |

**Net assessment**: The main execution risks are RAM management (solvable with memory mapping) and numerical stability of whitening (solvable with regularization). Both are standard problems with known solutions, but they must be implemented correctly from the start -- not discovered after a full run produces NaN values.

### Stage 4: Linear Probes (40-class, 8-class, 5-class at every layer)

| Failure Mode | Likelihood | Impact | Details |
|---|---|---|---|
| **Scikit-learn memory limits for 3584-dim inputs** | LOW | LOW | LogisticRegression on 800 samples x 3584 features is trivial for sklearn. Memory: ~22MB per matrix. Even running at all 64 layers, this is well within limits. |
| **5-fold CV with 40 classes and small N produces unstable results** | HIGH | MEDIUM | With ~800 stimuli / 80 products (40 real + 40 fictional) / 10 variants each, the 40-class real-product probe has 10 samples per class. 5-fold CV gives 8 train / 2 test per class per fold. This is better than the original 5-samples-per-class design but still few-shot. Expect high variance across folds. **Report per-fold results, not just means.** Use stratified splitting. |
| **Probe training across 64 layers x 4 components x 3 class counts x 2 anisotropy variants takes too long** | MEDIUM | LOW | That is 64 x 4 x 3 x 2 = 1,536 probe configurations, each with 5-fold CV = 7,680 fits. Each fit on 640 samples x 3584 features takes ~0.5 seconds with liblinear. Total: ~64 minutes. Acceptable but not instant. |
| **Solver convergence warnings at certain layers** | MEDIUM | LOW | Logistic regression may not converge in default 100 iterations at layers where representations are poorly conditioned. Increase `max_iter` to 1000. Use `lbfgs` solver for multinomial. |

**Net assessment**: Probe training is computationally cheap and well-understood. The only real risk is interpretability of results with small per-class N, which is a statistical concern already addressed by making RSA primary.

### Stage 5: Reporting and Visualization

| Failure Mode | Likelihood | Impact | Details |
|---|---|---|---|
| **matplotlib memory issues with many large plots** | LOW | LOW | Standard plotting. No risk. |
| **Multiple-comparison corrections reduce significance** | MEDIUM | MEDIUM | Testing H1/H2/H3 across 64 layers with multiple metrics = many comparisons. Bonferroni correction may eliminate marginal effects. Pre-register primary layer selection or use FDR correction (Benjamini-Hochberg). |
| **Results are ambiguous / partially supporting** | HIGH | MEDIUM | The most likely outcome is partial support for some hypotheses. This is not a failure, but it requires more nuanced writing than clear support or falsification. Budget extra time for interpretation. |

**Net assessment**: Low execution risk. Budget time for interpretation.

---

## 2. Resource Risk Assessment

### 2.1 VRAM Budget: Qwen3.5-27B 4-bit + Hidden State Extraction

| Component | VRAM Estimate |
|---|---|
| Model weights (4-bit GPTQ, 27B params) | ~14 GB |
| KV cache (batch=1, seq_len=150, 64 layers, 28 heads x 128 dim) | ~0.07 GB |
| Activations during forward pass | ~0.5 GB |
| Hidden states (all 64 layers, 150 tokens, 3584 dim, float32) | ~0.14 GB per stimulus |
| CUDA context + framework overhead | ~1.5 GB |
| **Total (batch_size=1)** | **~16.2 GB** |
| **Headroom on 32 GB** | **~15.8 GB** |

**Verdict**: Comfortable fit at batch_size=1. Batch_size=4 would add ~0.5GB for hidden states but more for activations/KV cache. Conservative approach: batch_size=1, process sequentially. At ~0.5 seconds per stimulus, 800 stimuli = ~7 minutes. Extremely fast. **This is not a bottleneck.**

Note: The hidden states must be moved to CPU and written to disk immediately after each forward pass. Do not accumulate in VRAM.

### 2.2 VRAM Budget: Qwen3.5-27B FP16 with CPU Offloading

| Component | VRAM + RAM Estimate |
|---|---|
| Model weights (FP16, 27B params) | ~54 GB total |
| -- GPU portion (fill to ~28 GB) | ~28 GB VRAM |
| -- CPU portion (remainder) | ~26 GB RAM |
| KV cache (split across devices) | ~0.14 GB |
| Hidden states | ~0.14 GB |
| OS + Python + framework overhead | ~4 GB RAM |
| **Total RAM needed** | **~30 GB** |
| **Available RAM** | **32 GB** |

**Verdict**: Barely fits. The 2GB margin is dangerously thin. Any memory spike (Python garbage collection, OS background processes) could trigger OOM or swap thrashing. **Recommendation: close all other applications, set a swap file of at least 16GB, and test with 5 stimuli first.** If RAM is insufficient, reduce the GPU allocation to offload more to CPU (trading speed for stability) or skip this control entirely and rely on 8-bit quantization as an intermediate control.

**Speed estimate**: With ~48% of layers on CPU, each forward pass will be dominated by CPU matmul speed. Conservative estimate: 15-30 seconds per stimulus. For a subset of 200 stimuli: 50-100 minutes. For all 800: 3.3-6.6 hours. **This should be run as an overnight batch, not during an interactive evening session.**

### 2.3 VRAM Budget: Llama-3.1-8B FP16

| Component | VRAM Estimate |
|---|---|
| Model weights (FP16, 8B params) | ~16 GB |
| KV cache + activations | ~0.5 GB |
| Hidden states (32 layers, 4096 dim) | ~0.075 GB |
| Framework overhead | ~1.5 GB |
| **Total** | **~18 GB** |
| **Headroom on 32 GB** | **~14 GB** |

**Verdict**: Very comfortable. No issues expected.

### 2.4 System RAM Budget for Analysis Phase

| Data | Size |
|---|---|
| Hidden states, one model, one component | 800 x 64 x 3584 x 4 bytes = ~0.7 GB |
| Hidden states, one model, all 4 components | ~2.8 GB |
| Similarity matrix, one layer | 800 x 800 x 4 bytes = 2.5 MB |
| All similarity matrices, one variant | 64 x 2.5 MB = 160 MB |
| Whitening: covariance matrix per layer | 3584 x 3584 x 4 bytes = 51 MB per layer |
| Whitening: all layers | 64 x 51 MB = 3.3 GB |

**Verdict**: Loading one model's hidden states for one component (~0.7GB) and computing whitening layer-by-layer (51MB peak) is very feasible. The original 28GB estimate assumed loading all 4 components simultaneously -- **do not do this**. Process one component at a time, one layer at a time. Peak RAM for analysis: ~4 GB. No issues.

**Critical insight**: The data sizes in the project brief were worst-case estimates assuming everything loaded simultaneously. With a layer-by-layer, component-by-component processing strategy, RAM is not a constraint.

### 2.5 Disk Budget

| Data | Size |
|---|---|
| Stimuli JSON | < 1 MB |
| Hidden states: Qwen 4-bit, all components | 800 x 64 x 3584 x 4 bytes x 4 components = ~2.8 GB |
| Hidden states: Qwen FP16, all components | Same: ~2.8 GB |
| Hidden states: Llama-3.1-8B, all components | 800 x 32 x 4096 x 4 bytes x 4 components = ~1.6 GB |
| Similarity matrices: all variants | ~4 GB |
| Probe results (accuracy scores) | < 100 MB |
| Plots and reports | < 1 GB |
| Model downloads (cached) | ~14 GB (Qwen 4-bit) + ~16 GB (Llama) + ~54 GB (Qwen FP16) = ~84 GB |
| **Total (excluding model cache)** | **~12 GB** |
| **Total (including model cache)** | **~96 GB** |

**Verdict**: The 100-200GB estimate in the project brief was inflated. Actual experimental data is ~12GB. Model downloads are the bulk of disk usage. If the Qwen FP16 model has already been downloaded for other purposes, this drops significantly. **Ensure at least 100GB free disk space before starting, primarily for model downloads.**

**Important**: The hidden state calculation above uses the *mean-pooled* representation (one vector per layer per stimulus), not the full token-level states. If storing full token-level hidden states (before mean pooling), multiply by ~150 (average token count), yielding ~420GB for Qwen alone. **Mean-pool during extraction, not after. Do not store raw token-level hidden states.**

### 2.6 API Cost Summary

| API | Calls | Estimated Cost |
|---|---|---|
| Claude Sonnet (stimulus generation) | ~800 | ~$2.50 |
| GPT-4 (10-product crossed subset) | ~50 | ~$2.00 |
| **Total** | | **~$4.50** |

**Verdict**: Negligible.

### 2.7 Wall-Clock Time Estimate

| Task | Estimated Time | Notes |
|---|---|---|
| **Stimulus generation** | 2 hours | Including prompt engineering, API calls, quality review |
| **Pilot validation** (hooks, quantization, shapes) | 1 hour | Critical gate before committing to full runs |
| **Hidden states: Qwen 4-bit** (800 stimuli) | 15 minutes | batch_size=1, ~1 sec/stimulus |
| **Hidden states: Qwen FP16 subset** (200 stimuli) | 1.5-3 hours | CPU offloading, 15-30 sec/stimulus. **Run overnight.** |
| **Hidden states: Llama-3.1-8B** (800 stimuli) | 20 minutes | FP16, comfortable fit |
| **Component decomposition hooks** (repeat extraction with hooks) | 45 min - 1.5 hours | Re-run Qwen 4-bit + Llama with forward hooks for attn/MLP/residual |
| **Similarity analysis** (RSA, all variants) | 1 hour | Layer-by-layer processing, 24 variants |
| **Linear probes** (all configurations) | 1.5 hours | 7,680 probe fits |
| **Visualization and reporting** | 3-4 hours | Plots, statistical tests, writeup |
| **Debugging, troubleshooting, re-runs** | 3-5 hours | The unknowable unknown |
| **Total** | **14-20 hours** |

**Verdict**: The 15-20 hour estimate is realistic but tight. The Qwen FP16 run is the schedule risk -- if it takes 6+ hours and must be run overnight, it consumes one of the 4-5 evenings just for setup/monitoring. **If time pressure emerges, the FP16 subset run is the first candidate for deferral** (see Section 5).

---

## 3. Computational Risks

### 3.1 CPU Offloading for FP16 Qwen3.5-27B

**Risk level**: HIGH

HuggingFace's `device_map="auto"` with `accelerate` will split layers across GPU and CPU. Known issues:

1. **Hidden state collection across device boundaries**: `output_hidden_states=True` should work because hidden states are collected at the model level (after each layer's forward, the output is appended to a tuple). The states are tensors on whichever device the layer ran on. They must be moved to a common device before stacking. HuggingFace handles this, but verify with a pilot.

2. **Forward hooks on CPU-offloaded layers**: Hooks fire on the device where the layer executes. If capturing attention/MLP decomposition, hooks on CPU layers return CPU tensors while hooks on GPU layers return CUDA tensors. The extraction code must handle mixed devices.

3. **Speed**: CPU layers dominate runtime. Each CPU matmul for a 3584x3584 weight matrix on a 150-token sequence takes ~10-50ms. With ~30 offloaded layers, each with 2 matmuls (attention + MLP), that is 60 matmuls x 30ms = ~1.8 seconds per layer per stimulus. Total: ~30 offloaded layers x 1.8s = ~54 seconds per stimulus. This is optimistic; memory bandwidth is often the bottleneck, not compute.

**Mitigation**:
- Pilot with 5 stimuli to measure actual wall-clock time.
- If >30 seconds per stimulus, reduce the subset to 100 stimuli (1.5-hour budget).
- If >60 seconds per stimulus, consider 8-bit quantization as an intermediate control instead (fits in 32GB VRAM, runs at GPU speed).
- Run overnight as an unattended batch.

### 3.2 `output_hidden_states=True` Through Quantization

**Risk level**: MEDIUM-HIGH

The `auto-gptq` and `autoawq` libraries replace `nn.Linear` layers with custom `QuantLinear` modules. The `forward()` method of the overall model class (e.g., `Qwen2ForCausalLM`) is responsible for collecting hidden states. This happens at the `DecoderLayer` level, not the `Linear` level. Therefore:

- `output_hidden_states=True` **should work** because it collects the output of each `DecoderLayer`, regardless of what happens inside the layer.
- Forward hooks on `DecoderLayer` **should work** for the same reason.
- Forward hooks on individual `Linear` layers inside quantized blocks **may not work** because these layers are replaced by non-standard modules.

**Validation protocol**:
```python
# Pilot test (run before committing to full extraction)
model = AutoModelForCausalLM.from_pretrained("...", device_map="auto")
inputs = tokenizer("test input", return_tensors="pt").to(model.device)
outputs = model(**inputs, output_hidden_states=True)
assert outputs.hidden_states is not None
assert len(outputs.hidden_states) == model.config.num_hidden_layers + 1
print(f"Hidden states: {len(outputs.hidden_states)} layers, shape: {outputs.hidden_states[0].shape}")
```

If this fails, the fallback is to register forward hooks on `model.model.layers[i]` (the `DecoderLayer` modules) to capture the residual stream output. This is equivalent to `output_hidden_states` and bypasses any quantization-aware code paths.

### 3.3 Numerical Stability of Whitening

**Risk level**: MEDIUM

With 800 samples in 3584 dimensions, the sample covariance matrix is rank-deficient (rank <= 799). Full whitening requires inverting this matrix, which is impossible without regularization.

**Concrete failure scenario**:
1. Compute covariance: `C = (X - X.mean(0)).T @ (X - X.mean(0)) / (N-1)` -- shape (3584, 3584)
2. Eigendecompose: `eigenvalues, eigenvectors = np.linalg.eigh(C)`
3. ~2785 eigenvalues are exactly zero (3584 - 799)
4. Whitening transform: `W = eigenvectors @ diag(1/sqrt(eigenvalues)) @ eigenvectors.T`
5. Division by zero for the 2785 zero eigenvalues --> NaN/Inf

**Required fix**: PCA-whitening with truncation:
```python
# Retain only components with eigenvalue > epsilon
k = min(n_samples - 1, n_features)  # = 799
eigenvalues = eigenvalues[-k:]  # top-k
eigenvectors = eigenvectors[:, -k:]
W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-8)) @ eigenvectors.T
```

This reduces dimensionality from 3584 to 799, which is fine for pairwise distance computation. The epsilon (1e-8) prevents division by near-zero eigenvalues.

**Additional concern**: Some layers (especially very early or very late) may have near-degenerate covariance even within the top-799 components. Monitor the condition number of the covariance matrix at each layer and flag layers where it exceeds 1e6.

### 3.4 Batch Size and Sequence Length Variation

**Risk level**: LOW

Stimuli target 80-150 tokens. With batch_size=1, there is no padding issue. If batching for speed, stimuli must be padded to the longest sequence in the batch, and the mean-pooling mask must exclude pad tokens. This is standard but a common source of bugs.

**Recommendation**: Use batch_size=1 for correctness. The total extraction time is dominated by model loading, not per-stimulus inference. With ~7 minutes for 800 stimuli on Qwen 4-bit and ~20 minutes on Llama, batching saves negligible time while introducing padding complexity.

### 3.5 Memory-Mapped File Access

**Risk level**: LOW

NumPy's `np.memmap` and `np.load(mmap_mode='r')` work well on WSL2. The only concern is if the hidden state files are stored on a Windows NTFS mount (e.g., `/mnt/c/`), which has slower I/O than the native ext4 filesystem inside WSL2. **Store all experimental data on the WSL2 native filesystem** (e.g., `~/`), not on a Windows mount.

---

## 4. Timeline Risk Analysis

### 4.1 Scaling from Original to Current Design

The original protocol estimated 15 hours for 200 stimuli, 1 model, no component decomposition, no anisotropy variants. The current design has:

| Dimension | Original | Current | Scaling Factor |
|---|---|---|---|
| Stimuli | 200 | ~800 (400 real + 400 fictional) | 4x |
| Model configs | 1 (Qwen 4-bit) | 3 (Qwen 4-bit, Qwen FP16 subset, Llama) | 3x |
| Hidden state components | 1 (full) | 4 (full, attn, MLP, residual) | 4x |
| Anisotropy variants | 0 (uncorrected only) | 2 (corrected + uncorrected) | 2x |
| Probe configurations | 8-class only | 40-class + 8-class + 5-class | 3x |
| RSA analysis | Not in original | Primary method | New workstream |

The scaling is multiplicative for some dimensions (stimuli x models for extraction) but additive for others (RSA and probes are independent analysis passes). The effective scaling is roughly:

- **Extraction**: 4x stimuli x 3x models x (1 + overhead for hooks) = ~14x
- **Analysis**: 4x stimuli (quadratic for pairwise: 16x distances) x 4x components x 2x anisotropy = ~128x per-layer computations, but each computation is fast (~seconds)
- **Probes**: 4x stimuli x 4x components x 3x class counts x 2x anisotropy = ~96x probe fits, each taking ~0.5s

**Net scaling on wall-clock time**: Extraction dominates and scales ~14x on paper, but the original estimate was padded. The actual extraction times (see Section 2.7) are modest. The Qwen FP16 run is the exception.

### 4.2 Minimum Viable Results (MVR)

If time runs over, here is the priority ordering from most to least essential:

**Tier 1 -- Must Have (tests all three hypotheses with one model)**:
1. Stimuli generation: full 800 stimuli (real + fictional)
2. Qwen 4-bit extraction: full hidden states (residual stream only, no component decomposition)
3. RSA analysis: corrected and uncorrected, on residual stream
4. Linear probes: 40-class and 8-class, on residual stream
5. Phase structure plots, probe accuracy curves

**Estimated time**: 8-10 hours

**Tier 2 -- Should Have (adds cross-model validation)**:
6. Llama-3.1-8B extraction and analysis (same as above)
7. Comparison of RSA profiles between Qwen and Llama

**Additional time**: 3-4 hours

**Tier 3 -- Nice to Have (adds mechanistic depth)**:
8. Component decomposition (attention, MLP, residual) for Qwen 4-bit
9. Re-run RSA and probes on decomposed components
10. Multi-source stimulus analysis (generator effects)

**Additional time**: 3-4 hours

**Tier 4 -- Deferrable (quantization control)**:
11. Qwen FP16 subset run
12. Spearman correlation between FP16 and 4-bit RSA profiles

**Additional time**: 3-6 hours (mostly unattended)

### 4.3 What to Cut First

If an evening session runs long or a blocker appears:

1. **First cut**: Qwen FP16 subset run. It is a control, not a primary analysis. Run it later as a follow-up if Qwen 4-bit results are interesting.
2. **Second cut**: Component decomposition. Interesting but not required for the three hypotheses. Can be added later without re-running extraction if full hidden states are saved (just add hooks in a second pass).
3. **Third cut**: Llama cross-validation. Reduces confidence in generalizability but does not invalidate the Qwen results.
4. **Do not cut**: RSA analysis, probe analysis, or fictional product control. These are the core of the experiment.

---

## 5. Mitigation Strategies

### 5.1 Critical Path Mitigations (do these first)

| # | Risk | Mitigation | Effort | When |
|---|---|---|---|---|
| M1 | `output_hidden_states` fails with quantized model | Run 1-stimulus pilot immediately after model loads. Verify hidden state count and shapes. | 15 min | Before any extraction |
| M2 | Forward hooks fail on quantized layers | Test hook on `model.model.layers[0]` with 1 stimulus. If fails, fall back to `output_hidden_states` for residual stream only (defer component decomposition). | 15 min | Before any extraction |
| M3 | OOM during Qwen FP16 run | Set up 16GB swap file. Close all applications. Test with 5 stimuli. If OOM, fall back to 8-bit quantization or skip. | 30 min | Before FP16 run |
| M4 | Whitening produces NaN | Implement PCA-whitening with truncation and epsilon from the start. Monitor condition numbers. | 30 min | During analysis code setup |
| M5 | Mean pooling over variable-length sequences introduces noise | Record token count per stimulus. After extraction, check correlation between token count and hidden state norm. If correlated, consider last-token extraction as alternative. | 15 min | After extraction |

### 5.2 Operational Mitigations (build into workflow)

| # | Risk | Mitigation | Details |
|---|---|---|---|
| M6 | Data loss from crash mid-extraction | Save hidden states incrementally (after each stimulus, not at the end). Use `np.save` to individual files or memory-mapped arrays. | Adds ~1 second overhead, prevents losing hours of work. |
| M7 | Disk space surprise | Check free disk space before each major stage. Set a 10GB warning threshold. | `df -h ~/` |
| M8 | Stimulus quality drift | Verify 5% random sample of stimuli manually. Check that all 10 variants of a product share core factual claims. | 30 minutes of reading. |
| M9 | Ambiguous results | Pre-register the analysis plan (write it down before looking at results). Specify what "supports" vs "falsifies" looks like for each hypothesis at each analysis level. | Already partly done in the original protocol. Formalize. |
| M10 | Evening session overruns | Start each session with a clear goal (e.g., "tonight: Qwen 4-bit extraction + RSA"). If behind schedule, cut scope per Section 4.3 rather than rushing. | Discipline, not code. |

### 5.3 Recovery Strategies

| Scenario | Recovery |
|---|---|
| Qwen 4-bit hidden state extraction fails entirely | Switch to Qwen3.5-27B 8-bit (bitsandbytes). Slightly larger (~18GB VRAM), still fits. Less quantization distortion. |
| FP16 Qwen run OOMs even with swap | Skip it. Use 8-bit as intermediate quantization control instead. Alternatively, use a smaller Qwen variant (e.g., Qwen3.5-14B at FP16, ~28GB, fits in VRAM). |
| RSA shows no phase structure (H1 falsified) | This is a valid result. Report it. Check uncorrected (raw) cosine similarity to see if anisotropy correction removed the signal. If raw shows the pattern but corrected does not, this is itself an interesting finding about the relationship between anisotropy and phase structure. |
| All probes show ceiling accuracy at all layers | Task is too easy. This would mean even early layers encode product identity well. Consider: is this because the stimuli are too lexically distinctive? Run the BoW baseline to check. If BoW also achieves high accuracy, the stimuli need revision. |
| Fictional products behave identically to real products | This is the *hoped-for* outcome (supports protocol-layer hypothesis over memorization). Not a failure mode. |
| Fictional products show no phase structure while real products do | Suggests memorization contributes to the effect. An important negative result for the strong form of the hypothesis. |

---

## 6. Risk Summary Matrix

| Risk | Likelihood | Impact | Stage | Mitigation |
|---|---|---|---|---|
| `output_hidden_states` fails with GPTQ/AWQ | MEDIUM | CRITICAL | Extraction | Pilot test (M1) |
| Forward hooks fail on quantized layers | MEDIUM | HIGH | Extraction | Pilot test (M2), fallback to residual-only |
| Qwen FP16 OOM or impractically slow | HIGH | MEDIUM | Extraction | Swap file + pilot (M3), fall back to 8-bit |
| RAM exhaustion during analysis | HIGH | HIGH | Analysis | Process layer-by-layer, memory-map files |
| Whitening numerical instability | MEDIUM | HIGH | Analysis | PCA-whitening with regularization (M4) |
| Timeline overrun beyond 20 hours | MEDIUM | MEDIUM | All | Tier-based scope cutting (Section 4.3) |
| Mean pooling noise from length variation | MEDIUM | MEDIUM | Extraction | Monitor and test last-token alternative (M5) |
| 40-class probe high variance | HIGH | LOW (RSA is primary) | Probes | Report per-fold, use RSA as primary evidence |
| Stimulus semantic drift across variants | MEDIUM | MEDIUM | Generation | Manual spot-check (M8) |
| Disk space exhaustion | LOW | HIGH | All | Monitor free space (M7), mean-pool at extraction time |

---

## 7. Recommended Execution Order

Based on the risk assessment, the optimal execution order is:

### Evening 1: Stimulus Generation + Pilot Validation (4 hours)
1. Generate all ~800 stimuli via Claude API
2. Quality-check 5% sample
3. Download and load Qwen 4-bit model
4. **Run pilot validation**: 1-stimulus test of `output_hidden_states`, forward hooks, shape verification
5. If pilot passes: begin Qwen 4-bit extraction (runs in ~15 min, finish tonight)
6. Save all hidden states to disk

### Evening 2: Llama Extraction + Analysis Code (4 hours)
1. Llama-3.1-8B extraction (20 min)
2. Write and test RSA analysis code (layer-by-layer, memory-mapped)
3. Write and test PCA-whitening with regularization
4. Run RSA on Qwen 4-bit residual stream (corrected + uncorrected)
5. Generate initial phase structure plots -- **first look at H1**

### Evening 3: Probes + Component Decomposition (4 hours)
1. Run linear probes on Qwen 4-bit (40-class, 8-class, 5-class, all layers)
2. Run RSA on Llama (corrected + uncorrected)
3. If time permits: Qwen 4-bit extraction with forward hooks for component decomposition
4. Compare Qwen vs. Llama RSA profiles

### Evening 4: Deep Analysis + FP16 Control (4-5 hours)
1. Component-decomposed RSA and probes (if extracted in Evening 3)
2. Multi-source stimulus analysis (generator effects)
3. Start Qwen FP16 subset run (overnight batch)
4. Statistical testing, multiple-comparison corrections
5. Begin writing results

### Evening 5 (if needed): FP16 Analysis + Final Report (3-4 hours)
1. Analyze FP16 results (Spearman correlation with 4-bit)
2. Finalize all plots and tables
3. Write conclusions
4. Archive all data and code

---

## 8. Top 5 Execution Risks (Ranked)

1. **`output_hidden_states` / forward hooks incompatible with quantized model** -- CRITICAL, must pilot before any real work. If this fails without a workaround, the entire experiment is blocked.

2. **Qwen FP16 run is infeasible on this hardware** -- HIGH likelihood, but MEDIUM impact because it is a control, not a primary analysis. Have a fallback (8-bit quant or smaller model).

3. **Whitening produces garbage due to rank deficiency** -- MEDIUM likelihood but will silently corrupt all downstream RSA results if not caught. Implement PCA-whitening correctly from the start.

4. **Timeline overrun due to scope expansion** -- The design grew 4x in stimuli and added two model configs plus component decomposition. The 15-20 hour estimate is achievable only with disciplined scope management and the tiered cutting strategy.

5. **RAM exhaustion during analysis phase** -- Easily prevented by processing layer-by-layer, but if the analysis code naively loads all hidden states into RAM simultaneously, it will fail. Design the analysis pipeline for streaming/incremental processing from the start.
