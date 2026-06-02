# Plan Reconciliation

## Executive Summary

The red-team review identified 4 critical, 8 major, and 9 minor findings. After reconciliation: **12 findings are resolved** with specific plan changes, **3 findings are flagged** for user decision, and **6 findings are dismissed** as non-actionable or already addressed. The plan's scientific design is strong and requires no structural changes. The primary intervention is a significant simplification of implementation infrastructure (saving 6-10 hours) and a correction of computational cost estimates for permutation testing. With these changes applied, the experiment fits within 4-5 evenings for Tiers 1-2, which is realistic. The user's 14 prior decisions remain intact -- none are reopened.

## Resolved Findings (Changes Applied to Plan)

### 1. Simplify Implementation Structure (Critical #2)
- **Original Finding**: The plan specifies enterprise-grade infrastructure (7 subpackages, Pydantic config, checkpoint/recovery, 6 entry scripts, 15 test files) adding 6-10 hours of development with zero scientific value.
- **Resolution**: Flatten to 5 core modules: `stimuli.py`, `extraction.py`, `analysis.py`, `probes.py`, `viz.py`. Replace Pydantic config with a Python dataclass or dict. Eliminate the `StageCheckpoint` class -- HDF5 incremental writes handle extraction recovery, and `stimuli.json` saves handle generation recovery. Merge all entry scripts into a single `run.py` with subcommands (or just run modules directly). Reduce test files to 5 covering critical paths: RDM computation, RSA correlation, anisotropy correction, pooling, and GroupKFold stratification.
- **Confidence**: HIGH
- **Rationale**: All three critics converged that the infrastructure is over-scoped. The Scope Auditor's estimate of 6-10 hours saved is credible. The user approved a 3-5 evening timeline; those evenings should be spent on science, not plumbing. The simplified structure preserves every analytical capability.

### 2. Adopt Tiered Permutation Testing (Critical #3)
- **Original Finding**: Full permutation testing (10,000 permutations x 64 layers x multiple RDMs) would take ~10+ hours, exceeding the plan's "1-2 hours" estimate by 5-10x.
- **Resolution**: Two-tier strategy: (a) Fast screen -- 200 permutations at all 64 layers to identify candidate significant layers (~12 minutes). (b) Full test -- 10,000 permutations at the top-5 candidate layers plus the pre-registered peak layer (~20 minutes). For control task probes (Hewitt & Manning), use 1 random permutation per probe task (not 5-10), which is the standard in the original paper and sufficient for computing selectivity. This yields 3 additional probe fits per layer (one per task), not 15-30.
- **Confidence**: HIGH
- **Rationale**: The math is straightforward. 10,000 x 64 layers is computationally prohibitive for a personal evening project. The tiered approach retains comprehensive layer scanning (the Domain Expert's concern) while making full rigor practical (the Architect's concern). The 200-permutation screen has sufficient resolution to rank layers reliably.

### 3. Restructure Evening 1 Schedule (Critical #1)
- **Original Finding**: Evening 1 contains scaffolding, product catalog creation, stimulus generation, model download, pilot validation, and full extraction -- estimated at 5-6 hours by all three critics, not 4.
- **Resolution**: With the simplified implementation (Resolved #1), Evening 1 becomes tractable. Revised schedule:
  - **Evening 1 (4 hours)**: Set up minimal project structure (~30 min). Define product catalog (~45 min). Write and run stimulus generation (~60 min including spot-check). Download primary model (~20 min parallel with spot-check). Run 1-stimulus pilot (~15 min). Begin Qwen 4-bit extraction (~30-60 min). Total: ~3.5-4.5 hours.
  - The key enabler is that scaffolding drops from "3-4 hours" to "~1.5 hours" because there is no Pydantic config, no checkpoint system, no debug model config, and no 15-file test suite to write upfront.
- **Confidence**: HIGH
- **Rationale**: The overloaded Evening 1 was a symptom of over-engineering, not a fundamental content problem. The simplified structure resolves it without splitting into two sessions. If stimulus generation takes longer than expected (prompt engineering iteration), extraction can spill into Evening 2 without cascading schedule damage.

### 4. Budget Realistic Extraction Timing (Major #7)
- **Original Finding**: Plan estimates ~7 minutes for 800-stimulus extraction; realistic estimate is 25-65 minutes including hook capture, CPU transfer, pooling, and HDF5 writes.
- **Resolution**: Budget 30-60 minutes for Qwen 4-bit extraction, 20-30 minutes for Llama 8B. These are already accommodated in the restructured Evening 1 (Resolved #3). Update the plan's Risk Mitigations table to reflect 30-60 min, not 7-15 min.
- **Confidence**: HIGH
- **Rationale**: The 0.5 sec/stimulus estimate only counted forward pass time. Hook capture + CPU transfer + pooling + gzip write adds 1.5-4.5 sec overhead per stimulus. This is a straightforward arithmetic correction.

### 5. Add Intermediate Sanity Check After First RSA Results (Major #1)
- **Original Finding**: The plan is a linear pipeline with no decision gate between Evening 2's first RSA results and proceeding through probes and remaining analyses.
- **Resolution**: After the first Qwen RSA results (Evening 2), spend 15 minutes on a three-way triage:
  - **Flat RSA across all layers**: Stop. Check stimuli quality (are registers actually distinct?), check extraction (are hidden states non-degenerate?), check RDM construction. Do not proceed until diagnosed.
  - **Weak middle-layer peak (RSA r < 0.05)**: Proceed with probes to get additional signal, but flag that the effect may be null or very small. Consider whether partial RSA changes the picture.
  - **Clear middle-layer peak (RSA r > 0.1)**: Proceed as planned.
- **Confidence**: HIGH
- **Rationale**: This costs 15 minutes and could save hours. The most dangerous failure mode is grinding through probe training and statistical testing on data where a trivial upstream bug (e.g., stimuli all being near-identical, or extraction returning zeros) would have been caught by eyeballing the RSA curve.

### 6. Skip Debug Model Configuration (Contradiction Resolution)
- **Original Finding**: The Devil's Advocate says the debug model (Qwen2.5-1.5B) has different architecture and won't catch 27B-specific issues. The Scope Auditor says it is scope creep.
- **Resolution**: Eliminate the debug model config entirely. Test code logic with random tensors (shape-correct numpy arrays). Test integration with the real model using 5 stimuli in the pilot step (expanding the plan's 1-stimulus pilot to 5). This resolves both critics' concerns: no wasted download, and integration testing happens on the actual target.
- **Confidence**: HIGH
- **Rationale**: Both critics are right. The debug model is simultaneously inadequate as a proxy and unnecessary as overhead. Random tensors test code paths; 5-stimulus pilot tests real-model integration.

### 7. Skip Generalized Checkpoint System; Add Staleness Warning (Contradiction Resolution)
- **Original Finding**: Devil's Advocate wants artifact versioning to prevent stale downstream results. Scope Auditor says checkpoint/recovery is over-engineered since longest stage is ~30 minutes.
- **Resolution**: No `StageCheckpoint` class. HDF5 incremental writes handle extraction crashes. Stimulus generation saves to JSON after each batch (trivial). Add a lightweight timestamp check at each stage entry point: if any input file is newer than the expected output file, print a warning ("Stage 3 outputs may be stale -- Stage 2 HDF5 was modified after Stage 3 last ran"). This is ~10 lines of code, not a framework.
- **Confidence**: HIGH
- **Rationale**: The Scope Auditor is correct that crash recovery for 30-minute stages is not worth building. The Devil's Advocate's staleness concern is valid but solved by a simple timestamp comparison, not by an artifact versioning system.

### 8. Add Per-Stimulus NaN/Inf Check During Extraction (Minor)
- **Original Finding**: The plan detects NaN/Inf after extraction (HDF5 validator) but not during. A single NaN stimulus silently corrupts all pairwise distances.
- **Resolution**: Add a NaN/Inf check inside the extraction loop, immediately after mean pooling. If detected, log the stimulus ID, skip it, and continue. This is 3 lines of code in the extraction function.
- **Confidence**: HIGH
- **Rationale**: Trivial to implement, high payoff. Catching NaN at extraction time rather than post-hoc saves debugging time.

### 9. Prefer Native Transformers GPTQ Over auto-gptq (Minor)
- **Original Finding**: `auto-gptq` has had breaking changes; native transformers GPTQ loading (since v4.33) is preferred.
- **Resolution**: Remove `auto-gptq` from required dependencies. Use `transformers>=4.40` native GPTQ support (already a dependency). Keep `auto-gptq` as an optional fallback dependency only, not installed by default.
- **Confidence**: HIGH
- **Rationale**: The Feasibility Checker is correct. Native transformers GPTQ reduces dependency surface and breaking-change risk. The plan already requires transformers>=4.40 which includes this support.

### 10. Resolve RTX 5090 vs RTX 3080 Ambiguity (Minor)
- **Original Finding**: Open Question #3 flags that the user profile says RTX 3080 but the experiment assumes RTX 5090.
- **Resolution**: Decision #13 in the decision log explicitly states "Target hardware: PowerSpec 5090 system (RTX 5090, 32GB VRAM), not WSL desktop." This is already resolved. Remove Open Question #3 from the plan. The experiment runs on the PowerSpec machine, not the WSL desktop.
- **Confidence**: HIGH
- **Rationale**: The user already decided this (Decision #13). The red-team review flagged it because the plan still carried it as an open question.

### 11. Add BoW Baseline Leakage Check (Major #8)
- **Original Finding**: The original idea included a BoW baseline check that was dropped from the draft plan.
- **Resolution**: Add a lightweight BoW baseline after stimulus generation: train a bag-of-words classifier (TF-IDF + logistic regression) on the same three tasks (product, category, register). If BoW product classification accuracy is high (>50% for 40-class), surface features may be driving results and stimuli need revision. This takes ~30 minutes to implement and run, using existing scikit-learn infrastructure. Place it at the end of Stage 1 as a stimulus quality gate.
- **Confidence**: HIGH
- **Rationale**: This was part of the original experimental design for good reason. The 40-class probe switch (Decision #1) and within-category discrimination partially address the concern, but the BoW baseline is a direct, easy test. Its absence from the draft plan appears to be an oversight.

### 12. Correct Storage Estimate (Minor)
- **Original Finding**: Plan estimates ~2GB for hidden states; Feasibility Checker computes ~4GB compressed.
- **Resolution**: Update storage budget table: hidden states from ~2 GB to ~4 GB. Total experimental data from ~4 GB to ~6 GB. Total including model cache unchanged (~88 GB, dominated by model downloads).
- **Confidence**: HIGH
- **Rationale**: Arithmetic correction. Does not affect feasibility (both fit comfortably on any modern disk) but the plan should have accurate numbers.

## Flagged for User Decision

### 1. Quantitative Register Distinctiveness Check (Major #3)
- **Original Finding**: Modern LLMs tend toward "helpful assistant" style regardless of register prompts. If register variation is shallow (vocabulary swaps rather than genuine structural differences), the experiment tests a weaker hypothesis. The plan includes manual spot-checking but no quantitative distinctiveness measure.
- **Options**:
  - **Option A: Add a quantitative register distinctiveness gate.** After stimulus generation, compute mean pairwise TF-IDF cosine distance between registers for each product. Define a minimum threshold (e.g., mean inter-register distance > 2x mean intra-register distance). If stimuli fail, iterate on prompts before proceeding. Cost: ~30-45 minutes of implementation + potential prompt iteration time.
  - **Option B: Rely on manual spot-check + BoW baseline.** The 5% manual spot-check catches obvious failures. The BoW register classifier (from Resolved #11) serves as a post-hoc distinctiveness measure: if BoW can classify registers well, they are at least lexically distinct. No explicit gate; trust prompt engineering.
- **Arbiter Recommendation**: Option A, but with a soft threshold (warning, not hard gate). The quantitative check is cheap and provides a number to report in the paper. However, the threshold should be empirically calibrated against the actual data distribution rather than pre-set, since there is no established norm for "how different is different enough" across registers.
- **Why This Needs User Input**: This is a trade-off between experimental rigor (catching shallow stimuli early) and schedule pressure (prompt iteration could consume an evening if the threshold is set too aggressively). The user's appetite for prompt-engineering iteration time determines which option is practical.

### 2. Fictional-vs-Real Control Threshold Calibration (Major #4)
- **Original Finding**: The Spearman r > 0.7 threshold for comparing fictional vs. real product RSA curves has no empirical basis and could be too lax or too strict.
- **Options**:
  - **Option A: Replace with split-half reliability baseline.** Compute split-half reliability of the real-product RSA curve (split 40 real products into two random subsets of 20, compute RSA on each, correlate). Use this as the empirical ceiling. The fictional-vs-real correlation should be compared against this baseline rather than an arbitrary absolute threshold. Cost: ~15 minutes of additional computation, trivially implemented.
  - **Option B: Keep r > 0.7 but report split-half alongside.** The pre-registered threshold stands for the formal test, but split-half reliability is reported as context for interpretation. This avoids changing a pre-registered criterion mid-stream.
- **Arbiter Recommendation**: Option A. The experiment has not been pre-registered anywhere yet (it is still in planning), so there is no commitment to r > 0.7. The split-half baseline is a strictly better criterion because it is empirically grounded. The implementation cost is negligible.
- **Why This Needs User Input**: This changes a pre-registered analysis criterion. Even though the experiment is not yet formally pre-registered, the user may prefer to keep the explicit threshold for simplicity and decide the split-half question after seeing data.

### 3. Cross-Model Validation Interpretation Framework (Major #5)
- **Original Finding**: If Qwen shows a protocol-layer effect but Llama does not, the plan provides no guidance. The two models differ in size, architecture, training data, and quantization -- making disagreement uninterpretable.
- **Options**:
  - **Option A: Pre-register specific agreement/divergence criteria.** Define: "agreement" = both show peak RSA in middle 60% of layers AND both show H1 effect at p < 0.05. "Partial agreement" = both show middle-layer peak but only one reaches significance. "Divergence" = one shows middle-layer peak, other shows flat or different pattern. Pre-register that divergence is attributed to "architecture/scale-dependent" rather than "null hypothesis" and requires follow-up with size-matched models.
  - **Option B: Keep Llama as exploratory.** Explicitly label the Llama comparison as exploratory (not confirmatory). Report whatever is found without pre-registered interpretation criteria. The primary claims rest entirely on Qwen.
- **Arbiter Recommendation**: Option B. Qwen and Llama differ on too many dimensions for any meaningful confirmatory comparison. Pre-registering interpretation criteria for a confounded comparison creates a false sense of rigor. Honest labeling as "exploratory cross-model comparison" is more scientifically appropriate. If both models agree, that is noteworthy but not confirmatory. If they disagree, the confounds make it uninterpretable regardless of pre-registration.
- **Why This Needs User Input**: This affects how the paper frames its claims. If the user intends the paper to make claims about "transformer middle layers in general" (architecture-general), then Llama needs a stronger role. If the claims are about "this model's representational geometry," then Llama-as-exploratory is fine.

## Dismissed Findings

### 1. No Artifact Versioning Strategy (Minor)
- **Original Finding**: If Stage 2 is re-run after fixing a bug, old HDF5 files are overwritten and Stage 3 results become stale.
- **Reason for Dismissal**: Addressed by Resolved #7 (staleness warning via timestamp check). A full versioning system is inappropriate for a solo research experiment run 5-10 times. The researcher knows when they re-ran a stage.

### 2. No GPT-4 Variant Specified (Minor)
- **Original Finding**: The plan does not specify which GPT-4 model or provide a fallback.
- **Reason for Dismissal**: This is a trivial implementation-time decision, not a planning concern. Use whatever GPT-4 variant is current at implementation time (likely gpt-4o). If the OpenAI API is unavailable, skip the cross-generator control -- it is not in Tier 1 and the generator confound can be acknowledged as a limitation.

### 3. PCA-Whitening Timing Unverified (Minor)
- **Original Finding**: PCA on 800 x 3584 (or 5120) at 64 layers could be slower than assumed.
- **Reason for Dismissal**: PCA on an 800 x 5120 matrix takes <1 second with sklearn on modern hardware. At 64 layers, that is ~1 minute total. This is not a timing risk even if the hidden dimension is larger than assumed.

### 4. Mean Pooling vs. Last-Token Divergence Not Checked Early (Minor)
- **Original Finding**: If pooling methods produce divergent RSA results, there is no early diagnostic.
- **Reason for Dismissal**: The plan already stores both pooling methods and reports both. The intermediate sanity check (Resolved #5) will reveal gross divergence. Adding a formal early pooling comparison gate adds complexity for a concern that is naturally addressed by the existing dual-reporting design.

### 5. 9 Visualization Types Exceed Original Scope (Minor)
- **Original Finding**: The original idea called for 3 visualization types; the plan specifies 9.
- **Reason for Dismissal**: The user approved scope expansions throughout the decision log (RSA, fictional products, multi-source, etc.). Additional visualizations are a consequence of the expanded analytical design. With the simplified implementation, visualization code is one module (`viz.py`) and each plot is a single function. The marginal effort for additional plot types is 10-15 minutes each, and they serve the publishability goal. The implementation simplification (Resolved #1) frees far more time than the extra plots consume.

### 6. Partial RSA Length Correction Assumes Linearity (Major #6)
- **Original Finding**: Partial RSA assumes the length confound is linear. Short texts may have qualitatively different representation properties than long texts.
- **Reason for Dismissal**: This is a valid theoretical concern but not actionable at the planning stage. The plan already (a) uses partial RSA (which is the standard approach), (b) targets 80-150 tokens (limiting the variance), and (c) reports both corrected and uncorrected results. The suggested diagnostic (plot residual RSA vs. length) is a good practice but does not require a plan change -- it is a standard post-hoc check any competent researcher would perform when reviewing residuals. Adding it as a formal plan step risks scope creep for a concern rated MEDIUM confidence.

## Emergent Issues

### Schedule Sensitivity to Model Identity Resolution
Three findings (Critical #4 model identity, Major #7 extraction timing, Minor storage estimate) all trace to the same root cause: the plan's numerical estimates are built on an unverified model architecture. The model identity issue (Critical #4) is not just its own finding -- it is a load-bearing assumption that propagates through every timing and storage estimate in the plan. **Resolution**: Model identity verification must be the literal first task of Evening 1, before any scaffolding. The first 15 minutes should be: search HuggingFace for the exact Qwen GPTQ model, confirm layer count and hidden dimension, and update the project's configuration constants. All downstream estimates become accurate once this is anchored.

### Simplified Implementation Unlocks the Schedule
The most important insight from the review is that Critical findings #1, #2, and #7 are deeply interdependent. Evening 1 is overloaded (Critical #1) *because* the implementation is over-engineered (Critical #2), and extraction timing is underestimated (Major #7). Resolving #2 automatically resolves #1 and absorbs #7. This is not three separate problems requiring three separate fixes -- it is one architectural decision (simplify) that cascades through the entire schedule.

## Reconciled Plan Changes Summary

Ordered by implementation sequence:

1. **Flatten project structure** to 5 modules (`stimuli.py`, `extraction.py`, `analysis.py`, `probes.py`, `viz.py`) plus a single `run.py` entry point. Eliminate Pydantic config, checkpoint class, debug model config, and per-stage scripts.
2. **Reduce test infrastructure** to 5 critical-path test files (RDM computation, RSA correlation, anisotropy correction, pooling, GroupKFold stratification). Tests still run without GPU/API/network.
3. **Verify model identity first** -- search HuggingFace for exact Qwen GPTQ model, confirm architecture, update all numerical estimates before writing any code.
4. **Expand 1-stimulus pilot to 5-stimulus pilot** for integration validation on the real model. No separate debug model.
5. **Add per-stimulus NaN/Inf check** in the extraction loop (3 lines of code).
6. **Remove `auto-gptq` from required dependencies**; rely on native transformers GPTQ support.
7. **Update extraction time budget** from 7-15 minutes to 30-60 minutes for Qwen 4-bit.
8. **Add BoW baseline** after stimulus generation as a surface-feature leakage check.
9. **Implement tiered permutation testing**: 200-permutation screen at all layers, 10,000-permutation full test at top-5 + peak layer. Control task probes use 1 random permutation each.
10. **Add intermediate sanity check** after first RSA results (15-minute triage protocol).
11. **Add lightweight staleness warning** (timestamp comparison at stage entry points, ~10 lines).
12. **Update storage estimate** from ~2GB to ~4GB for hidden states.
13. **Remove Open Question #3** (RTX 5090 vs 3080) -- already resolved by Decision #13.
14. **Revised Evening 1 schedule**: Model ID verification (15 min) -> minimal scaffolding (30 min) -> product catalog (45 min) -> stimulus generation + spot-check (60 min) -> model download (20 min, parallel) -> 5-stimulus pilot (15 min) -> begin extraction (30-60 min).

## Remaining Open Questions

1. **Register distinctiveness threshold** (Flagged #1): Should there be a quantitative gate on register variation, and if so, how aggressive should the threshold be?
2. **Fictional-vs-real threshold methodology** (Flagged #2): Replace r > 0.7 with split-half reliability baseline, or keep the explicit threshold?
3. **Llama's role in the paper** (Flagged #3): Exploratory cross-model comparison or confirmatory with pre-registered criteria?
4. **Exact Qwen model ID**: Still unknown. Must be resolved at implementation time (see Emergent Issues). The most likely candidate is `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` (64 layers, hidden_dim=5120), which would increase VRAM and storage estimates by ~43% over the plan's assumed 3584 dimensions. If hidden_dim is 5120, confirm that 4-bit quantized model still fits in 32GB VRAM with extraction hooks active.
5. **Token range prompt policy** (Open Question #7 from draft plan, untouched): Should generation prompts explicitly allow wider variance or target 80-150 with wider post-hoc acceptance? This interacts with Flagged #1 (register distinctiveness) since constraining length may flatten register differences.
