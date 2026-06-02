# Feasibility Assessment

## Overall Feasibility: MEDIUM

This plan is technically sound and well-designed, but it rests on a critical model-naming error and several assumptions about library compatibility that must be validated before committing time. The core experiment (RSA + probes on hidden states from a quantized 27B model) is achievable on the specified hardware, but "Qwen3.5-27B" does not exist as a public model -- the plan must use Qwen2.5 or Qwen3 (if released by execution time). The 5-evening schedule is tight but realistic if the researcher follows the tiered deferral strategy and accepts that Tier 3-4 items (component decomposition, FP16 control) will likely slip. The single biggest execution risk is the forward hook + quantization compatibility gate at Step 3.1 -- if this fails, the fallback path works but loses component decomposition entirely.

## Technical Feasibility

### Verified Claims

- **Qwen2.5-27B-Instruct GPTQ-Int4 fits in ~14-16GB VRAM at batch_size=1**: Verified by architecture math. 27B params at 4 bits = ~13.5GB for weights. KV cache at batch=1, seq_len=150 is negligible. Total ~16GB with overhead. Comfortable on 32GB RTX 5090.

- **`output_hidden_states=True` works with GPTQ models in HuggingFace Transformers**: This operates at the `DecoderLayer` level, not the `Linear` level. The GPTQ quantization replaces `nn.Linear` with `QuantLinear` inside each block, but the model's `forward()` method collects hidden states from the output of each `DecoderLayer`. This has been confirmed to work with both `auto-gptq` and the native transformers GPTQ integration (introduced in transformers v4.33). The plan's fallback (manual hooks on `model.model.layers[i]`) is also correct.

- **HDF5 handles incremental writes efficiently**: h5py supports resizable datasets via chunked storage with `maxshape=(None, ...)`. With gzip compression level 4, write overhead is modest. This is a standard pattern for large tensor storage and works as described.

- **PCA-whitening with sklearn is numerically stable for 800 samples in 3584 dimensions**: `sklearn.decomposition.PCA(n_components=min(N-1, D), whiten=True)` internally uses SVD (via LAPACK), which is numerically superior to eigendecomposition of the covariance matrix. With N=800, D=3584, the effective rank is at most 799. PCA will correctly truncate. The epsilon=1e-8 regularization on top of this is belt-and-suspenders but sensible.

- **GroupKFold with nested CV runs in reasonable time**: 5 outer folds x 3 inner folds x 5 C values x 64 layers x 3 tasks x 2 anisotropy methods = ~15,360 probe fits. Each fit: ~800 samples, ~799 dimensions (after PCA-whitening) or 3584 (raw), L2-regularized logistic regression. With `lbfgs` solver, each fit takes 0.2-1.0 seconds. Total: ~1.5-4 hours. This is within the schedule if parallelized across layers (trivially parallelizable) or run sequentially during an evening.

- **Llama-3.1-8B at FP16 fits comfortably in 32GB VRAM**: ~16GB for weights + overhead. No issues.

- **scipy `pdist` handles 800x3584 cosine distance computation**: 319,600 pairwise distances per layer. With scipy's optimized C implementation, this takes ~1-2 seconds per layer. 64 layers = ~2 minutes. Correct.

- **API costs for stimulus generation are negligible**: ~800 Claude Sonnet calls + ~50 GPT-4 calls = under $10. Correct.

### Unverified or Incorrect Claims

- **"Qwen3.5-27B" does not exist as a public HuggingFace model.** As of my knowledge cutoff, the Qwen series includes Qwen2.5 (released late 2024) and Qwen3 (which may or may not have been released by March 2026). There is no "Qwen3.5" series. The plan itself acknowledges this in Open Question #1 and Step 3.2, noting the correct name is likely `Qwen/Qwen2.5-27B-Instruct-GPTQ-Int4`. **Impact**: LOW -- the plan already flags this. At implementation time, search HuggingFace for the correct model ID. If Qwen3 has been released with a 27B variant, that would be an alternative. The architecture (Qwen2/Qwen3 both use standard `DecoderLayer` blocks) is compatible with the hook strategy.

- **auto-gptq>=0.7 compatibility is uncertain.** The `auto-gptq` library has had breaking changes between versions. As of late 2024, HuggingFace Transformers integrated native GPTQ support (via `GPTQConfig` in `transformers>=4.33`), which reduces the dependency on `auto-gptq` as a separate package. For Qwen2.5 GPTQ models hosted on HuggingFace, the native `transformers` GPTQ loader is the recommended path. **Impact**: MEDIUM -- the plan should prefer `transformers`' native GPTQ loading over `auto-gptq` as a standalone library. This simplifies the dependency chain. The plan's `pyproject.toml` should list `auto-gptq` as optional, not required.

- **RTX 5090 + PyTorch compatibility.** The RTX 5090 uses the Blackwell architecture (compute capability 10.0, or possibly 12.0). PyTorch support for Blackwell GPUs requires PyTorch >= 2.5 with CUDA 12.6+. As of early 2025, PyTorch nightly builds supported Blackwell. By March 2026, stable PyTorch should fully support it, but the plan should pin `torch>=2.5` (which it does) and verify CUDA 12.6+ is installed on the target system. **Impact**: LOW if the system has up-to-date drivers and CUDA. The plan correctly identifies this dependency in Step 1.1.

- **Hidden state storage at ~2GB (gzipped HDF5) for all models**: The plan estimates ~2GB for all hidden states. The risk assessment computes ~2.8GB for Qwen 4-bit alone (800 x 64 x 3584 x 4 bytes x 4 components). With gzip compression (typical 2-3x on float32 tensors), this becomes ~1-1.4GB per model config. For 3 model configs with all components: ~3-4.2GB compressed. The plan's ~2GB estimate in the storage budget is too low; ~4GB is more realistic for compressed data. **Impact**: LOW -- disk space is abundant.

- **Qwen2.5-27B hidden dimension is 3584**: This needs verification at implementation time. Qwen2.5-7B uses hidden_dim=3584, but Qwen2.5-27B may use a different dimension. Based on available model cards, Qwen2.5-72B uses 8192 and Qwen2.5-7B uses 3584. The 27B model likely uses an intermediate size. Without web access to confirm the exact config, I cannot verify this claim. If the hidden_dim is larger (e.g., 4096 or 5120), VRAM and storage estimates increase proportionally. **Impact**: MEDIUM -- if hidden_dim is 5120 instead of 3584, VRAM usage increases by ~43% for hidden state storage (still fits in 32GB), and disk usage scales similarly.

- **Qwen2.5-27B has 64 layers**: This also needs verification. Common configurations for 27B-parameter models vary. If the layer count differs, it affects extraction time, probe count, and zone boundary definitions. **Impact**: MEDIUM -- easily adaptable but affects runtime estimates.

### Technology Assessment

| Technology/Tool | Available? | Suitable? | Risk Level | Notes |
|-----------------|-----------|-----------|------------|-------|
| PyTorch >= 2.5 (CUDA 12.6+) | YES (by March 2026) | YES | LOW | Blackwell support should be stable by execution date |
| HuggingFace Transformers >= 4.40 | YES | YES | LOW | Native GPTQ support since 4.33. Actively maintained. |
| auto-gptq >= 0.7 | YES but uncertain maintenance | MAYBE | MEDIUM | Prefer transformers' native GPTQ integration. auto-gptq as fallback only. |
| autoawq >= 0.2 | YES | YES | LOW | Good alternative to GPTQ. Generally faster inference. |
| h5py >= 3.10 | YES | YES | LOW | Stable, mature library. Chunked/resizable datasets well-supported. |
| sklearn PCA (whiten=True) | YES | YES | LOW | SVD-based, handles rank-deficient case correctly. |
| scipy pdist + squareform | YES | YES | LOW | Efficient C implementation. No issues at N=800. |
| Anthropic Python SDK | YES | YES | LOW | Stimulus generation only. Simple usage pattern. |
| OpenAI Python SDK | YES | YES | LOW | Cross-generator subset only. |
| Qwen2.5-27B-Instruct-GPTQ-Int4 | LIKELY YES | YES | MEDIUM | Model name needs verification. HuggingFace likely hosts official GPTQ quants. |
| Llama-3.1-8B | YES | YES | LOW | Well-established, widely used. |
| accelerate (device_map="auto") | YES | YES | LOW | Required for FP16 CPU offloading. Mature. |

## Complexity Analysis

### High-Complexity Steps

- **Step 3.1 (VRAM Validation Gate)**: This is the critical path gate. If `output_hidden_states` or hooks fail with the quantized model, the entire extraction strategy must pivot. The plan correctly identifies this and provides fallback paths, but debugging quantization-related issues can consume an entire evening if the failure mode is subtle (e.g., hooks fire but produce incorrect values, not NaN but slightly wrong due to quantization artifacts).

- **Step 3.3 (Forward Hooks for Component Decomposition)**: Getting hooks to correctly capture attention output vs. MLP output vs. residual stream on quantized models across two different architectures (Qwen2 and Llama) is deceptively complex. The module hierarchy differs between architectures, and the output format of attention modules (tuple vs. tensor) varies. This will require careful architecture-specific code.

- **Step 4.1 (Anisotropy Correction)**: While PCA-whitening is mathematically straightforward, getting it right in practice requires careful handling of per-layer statistics, consistent application across conditions, and monitoring for degenerate layers. The decision to run both corrected and uncorrected (Decision #9) is wise but doubles the analysis pipeline.

- **Step 5.1 (Probe Training with Nested CV + GroupKFold)**: The combination of GroupKFold (product-level groups), nested CV (inner loop for C), and three probe tasks across 64 layers is conceptually simple but implementation-heavy. The main complexity is ensuring correct group assignment -- each product has 10 stimuli (5 registers x 2 variants), and all 10 must stay in the same fold. With 80 products (40 real + 40 fictional) split into 5 folds = 16 products per fold = 160 stimuli per fold. This is a reasonable split but requires careful bookkeeping.

### Hidden Complexity

- **Step 2.4 (Stimulus Generation Orchestrator)**: Appears simple (call API 800 times) but requires: checkpoint/resume logic, retry with backoff, validation loop, cross-generator coordination, and quality assurance. The prompt engineering alone for 5 registers that produce genuinely different surface forms while preserving semantic content is a multi-hour effort. The plan budgets "Evening 1" for this but underestimates the prompt iteration cycles.

- **Step 3.5 (Extraction Pipeline)**: The incremental HDF5 writing with resume support is more complex than it appears. HDF5 files can become corrupted if the process is killed during a write. The checkpoint mechanism needs to handle partial writes gracefully. Additionally, the extraction must track which stimuli have already been processed (for resume) while maintaining correct indexing into the HDF5 dataset.

- **Step 4.3 (Partial RSA)**: Constructing nuisance RDMs (stimulus length difference, lexical overlap via Jaccard distance) and implementing partial Spearman correlation is non-trivial. The partial correlation requires regressing out nuisance variables from both the observed and model RDM vectors (upper triangles), then correlating residuals. This is rarely available as a library function and must be implemented manually.

- **Step 6.1 (Hypothesis Tests with FDR Correction)**: Implementing Benjamini-Hochberg FDR across layers, computing Cohen's d for RSA correlations, and combining permutation-based p-values with effect-size criteria requires careful statistical implementation. The "pre-registered" criteria need to be coded as mechanical decision rules, which is harder than it sounds when results are continuous.

- **Product Catalog (Step 1.3)**: Defining 40 real products with 3-5 quantitative core attributes each, plus 40 fictional products with plausible names and novel feature combinations, is a substantial creative task. The plan treats this as scaffolding but it requires domain knowledge across 8 product categories and careful construction to ensure discriminability within categories.

## Dependency Assessment

### External Dependencies

| Dependency | Maintained? | Stable API? | Risk | Mitigation |
|------------|-----------|-------------|------|------------|
| PyTorch | YES, actively | YES | LOW | Pin version in pyproject.toml. Verify Blackwell support. |
| HuggingFace Transformers | YES, actively | YES (minor breaking changes between major versions) | LOW | Pin >= 4.40. Native GPTQ support is stable. |
| auto-gptq | UNCERTAIN | UNSTABLE (history of breaking changes) | MEDIUM | Prefer transformers native GPTQ. Only use auto-gptq if native loading fails. |
| autoawq | YES, actively (as of late 2024) | MOSTLY STABLE | LOW | Good fallback if GPTQ loading fails. |
| h5py | YES, mature | YES | LOW | No concerns. |
| scikit-learn | YES, actively | YES | LOW | No concerns. |
| scipy | YES, actively | YES | LOW | No concerns. |
| Anthropic SDK | YES, actively | MOSTLY STABLE (breaking changes in v1.0 transition) | LOW | Pin version. Simple usage for generation. |
| OpenAI SDK | YES, actively | YES | LOW | Pin version. Simple usage for generation. |
| Qwen2.5-27B GPTQ model weights | Hosted on HuggingFace (Qwen team) | N/A | MEDIUM | Model may be renamed, re-quantized, or reorganized. Pin exact revision hash. |
| Llama-3.1-8B model weights | Hosted on HuggingFace (Meta) | N/A | LOW | Well-established model. Requires Meta license agreement on HuggingFace. |
| NVIDIA drivers + CUDA | YES | YES | LOW for RTX 5090 by March 2026 | Verify driver >= 560 and CUDA >= 12.6. |

### Critical Dependency Chain

```
nvidia-smi (verify GPU) --> PyTorch + CUDA --> transformers + GPTQ loading --> model download --> 1-stimulus pilot --> full extraction
```

If any link in this chain fails, the experiment cannot proceed. The plan correctly identifies the 1-stimulus pilot as the critical gate.

## Timeline Assessment

### Is 5 evenings realistic?

**Marginally, with aggressive scope management.** The plan's tiered approach is well-designed, but the schedule has almost no slack for unexpected issues. Here is my evening-by-evening assessment:

**Evening 1 (4 hours): Stimulus Generation + Pilot -- TIGHT**
- Project scaffolding (pyproject.toml, config, constants): 1-1.5 hours minimum, not the trivial 30 minutes implied. Setting up 80 products with core attributes is substantial.
- Prompt engineering for 5 registers: 1-1.5 hours of iteration.
- API generation (800 calls): 15-30 minutes of runtime, but monitoring and spot-checking takes longer.
- Model download (~14GB): 10-30 minutes depending on network.
- 1-stimulus pilot: 15 minutes if it works. 1-2 hours if it doesn't.
- **Realistic outcome**: Stimuli generated, pilot validated. Extraction may not begin. The plan's assumption that full Qwen 4-bit extraction (~15 min) happens on Evening 1 is optimistic.

**Evening 2 (4 hours): Extraction + Analysis Code -- FEASIBLE**
- Qwen 4-bit extraction (if not done Evening 1): 15-30 minutes.
- Llama-3.1-8B extraction: 20-30 minutes.
- RSA analysis code: 1.5-2 hours (RDM computation, model RDM construction, Spearman correlation, permutation test).
- PCA-whitening: 30-45 minutes.
- Initial RSA run + plots: 30-45 minutes.
- **Realistic outcome**: Core extraction complete. RSA code written and producing initial results. First look at H1.

**Evening 3 (4 hours): Probes + Deeper Analysis -- FEASIBLE BUT COMPRESSED**
- Probe training pipeline with nested CV: 1.5-2 hours to code.
- Control task probes: 30 minutes additional code.
- Probe execution across all layers: 1.5-4 hours of runtime (can run in background).
- Llama RSA analysis: 30 minutes (reuse code from Evening 2).
- **Realistic outcome**: Probes running or complete. Both models analyzed with RSA. Component decomposition likely deferred.

**Evening 4 (4-5 hours): Deep Analysis -- TIGHT**
- Within-category discrimination analysis: 1 hour.
- Partial RSA (nuisance regressors): 1-1.5 hours (implementation + debugging).
- Multi-source analysis: 30 minutes.
- Statistical testing + FDR: 1-1.5 hours.
- FP16 overnight batch setup: 30 minutes.
- **Realistic outcome**: Core analyses complete. Statistical tests run. FP16 running overnight.

**Evening 5 (3-4 hours): Finalization -- LIKELY NEEDED**
- FP16 analysis (if overnight run succeeded): 1 hour.
- Visualization polish: 1-1.5 hours.
- Results writeup: 1-1.5 hours.
- **Realistic outcome**: Final results and report.

### Where will time actually go?

| Activity | Estimated % of Total Time |
|----------|--------------------------|
| Debugging model loading, hooks, quantization issues | 20-25% |
| Writing analysis code (RSA, probes, statistics) | 25-30% |
| Stimulus generation + quality review | 10-15% |
| Project scaffolding + configuration | 10-15% |
| Extraction runtime (wall-clock waiting) | 5-10% |
| Visualization and reporting | 10-15% |
| Unexpected issues and reruns | 10-15% |

### What's the MVP timeline?

**Minimum Viable Results (Tier 1 only): 10-12 hours across 3 evenings.**

This includes: stimulus generation, Qwen 4-bit extraction (residual stream only), RSA analysis (corrected + uncorrected), linear probes (3 tasks, all layers), control task probes, and hypothesis verdicts. No component decomposition, no FP16 control, no Llama cross-validation.

**Full scope (Tiers 1-4): 18-22 hours across 5-6 evenings.**

The plan estimates 14-20 hours. I estimate 18-22 hours is more realistic, accounting for debugging time that is consistently underestimated in research computing.

## Additional Findings

### Model Name Correction Required

The plan references "Qwen3.5-27B" throughout, which is not a real model series. The correct model is almost certainly from the **Qwen2.5** family. Likely candidates:
- `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` (if a 32B variant exists)
- `Qwen/Qwen2.5-27B-Instruct-GPTQ-Int4` (if specifically 27B)
- `Qwen/QwQ-32B` or `Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4`

If Qwen3 has been released by execution date (March 2026), a Qwen3-27B or similar may exist. The plan must verify exact model IDs on HuggingFace before starting.

### Qwen2.5 Architecture Details (Best Available Knowledge)

Based on available information for the Qwen2.5 family:
- **Qwen2.5-7B**: 28 layers, hidden_dim=3584, 28 attention heads
- **Qwen2.5-14B**: 48 layers, hidden_dim=5120, 40 attention heads
- **Qwen2.5-32B**: 64 layers, hidden_dim=5120, 40 attention heads
- **Qwen2.5-72B**: 80 layers, hidden_dim=8192, 64 attention heads

There may not be a 27B-specific model. If the closest available is 32B, the 4-bit quantized version would be ~16-17GB (fits in 32GB VRAM), but the hidden_dim would be 5120, not 3584 as assumed throughout the plan. This would increase:
- Hidden state storage by ~43%
- PCA-whitening computation time by ~2x (5120^2 covariance matrix)
- Probe training time slightly (more features)

None of these changes break feasibility, but the exact numbers in the plan would need updating.

### auto-gptq vs. Native Transformers GPTQ

The plan lists `auto-gptq>=0.7` as a dependency. As of transformers v4.33+, GPTQ models can be loaded natively without `auto-gptq` if the model was quantized with the GPTQ method and hosted on HuggingFace with the proper config. The native integration uses `optimum` under the hood. **Recommendation**: Try native loading first (`AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")`). Only install `auto-gptq` if native loading fails.

### RTX 5090 Specific Considerations

The RTX 5090 (Blackwell, GB202) has:
- 32GB GDDR7 VRAM (confirmed in the plan)
- PCIe 5.0 x16 interface
- Compute capability 10.0 (or 12.0 -- TBD based on final specs)

Key considerations:
1. **PyTorch Blackwell support**: By March 2026, stable PyTorch should fully support Blackwell. Verify with `torch.cuda.get_device_capability()`.
2. **Flash Attention**: FlashAttention v2+ should support Blackwell. This is relevant for efficient forward passes.
3. **GDDR7 vs. HBM**: The 5090 uses GDDR7, not HBM. Memory bandwidth is high (~1.7 TB/s) but lower than datacenter GPUs. For batch_size=1 inference, this is not a bottleneck.

### WSL2 Considerations

The plan notes this runs on a PowerSpec 5090 system, but the risk assessment mentions WSL2. Decision #13 says the target is a PowerSpec 5090 system, not the WSL desktop. If the target machine runs native Linux (not WSL2), the WSL2-specific concerns in the risk assessment are moot. If it does run WSL2:
- Modern WSL2 (kernel 6.6+) has excellent GPU passthrough
- GPU memory management is transparent -- NVIDIA driver handles the virtualization
- Store data on ext4 (native WSL filesystem), not NTFS mounts
- The `nvidia-smi` output should show the full 32GB VRAM

## Verdict

**This plan is feasible with the following caveats:**

1. **Model name must be corrected.** "Qwen3.5-27B" does not exist. Use Qwen2.5-32B (or whatever the closest available GPTQ-quantized model is at execution time). This may change the hidden_dim and layer count assumptions.

2. **The 1-stimulus pilot (Step 3.1) is absolutely critical.** Do not skip it. If `output_hidden_states` or hooks fail with the specific GPTQ model, the debugging could consume an entire evening. Have the fallback paths (AWQ variant, bitsandbytes 8-bit, hooks on DecoderLayer only) ready to deploy immediately.

3. **The 5-evening schedule is achievable for Tiers 1-2 only.** Tiers 3-4 (component decomposition, FP16 control) will likely require a 6th evening or acceptance of partial results. The plan's own tiered deferral strategy is excellent -- follow it strictly.

4. **Prefer native transformers GPTQ loading over auto-gptq.** This reduces dependency risk and is the officially supported path for HuggingFace-hosted GPTQ models.

**Biggest execution risk:** The forward hook + quantization compatibility gate. If hooks produce incorrect values (not failures, but silently wrong outputs), this could go undetected and invalidate results. The plan should include a cross-validation step: compare hook-captured residual stream outputs against `output_hidden_states` outputs for the same stimulus. They should be identical (or within float32 precision).

**Single most likely cause of failure:** Time. The plan is ambitious for 5 evenings. The most likely failure mode is not a technical crash, but running out of time before completing the full analysis pipeline, resulting in Tier 1 results only. This is an acceptable outcome -- the tiered design ensures that even partial completion yields meaningful results.
