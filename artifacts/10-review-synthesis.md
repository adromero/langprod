# Red-Team Review Synthesis

## Summary

Three red-team agents reviewed the draft plan for the Protocol Layer Hypothesis experiment. The review identified **4 critical findings, 8 major findings, and 9 minor findings**. The plan is scientifically thorough and methodologically sound, but suffers from three systemic issues: (1) the schedule is overloaded, particularly Evening 1, with realistic time estimates exceeding the plan's estimates by 50-100%; (2) the implementation is architecturally over-engineered for a solo research experiment, adding 6-10 hours of development time for infrastructure that contributes no scientific value; and (3) several computational cost estimates are off by an order of magnitude, creating hidden schedule risks that compound across phases. No critic found fundamental scientific flaws -- the analytical design (RSA primary, probes secondary, control tasks, partial RSA, within-category discrimination) is well-constructed and appropriate for publishable results.

## Critical Findings

### 1. Evening 1 Is Overloaded and Will Slip
- **Severity**: CRITICAL
- **Confidence**: HIGH
- **Source(s)**: Devil's Advocate (Risk Scenario #5), Scope Auditor (Effort Assessment), Feasibility Checker (Timeline Assessment)
- **Plan Section**: Execution Schedule, Evening 1
- **Issue**: Evening 1 contains: (a) full project scaffolding (5 steps, ~10-15 files of non-trivial code including 80 product definitions with quantitative attributes), (b) stimulus generation via API (~800 calls + spot-checking), (c) model download (~14GB), (d) 1-stimulus pilot validation, and (e) beginning full extraction. All three critics independently concluded this is 5-6 hours of work, not 4. The Scope Auditor estimates scaffolding alone at 3-4 hours with the current over-engineered structure; the Feasibility Checker estimates prompt engineering for 5 registers at 1-1.5 hours of iteration; the Devil's Advocate notes that product catalog creation (80 products with 3-5 quantitative attributes each) is a substantial creative task treated as scaffolding.
- **Impact**: If Evening 1 slips, every subsequent evening is compressed. The plan's tiered scope-cutting begins activating earlier than intended, potentially losing Tier 2 items (Llama cross-validation) that are important for the paper's claims.
- **Recommendation**: Either (a) split Evening 1 into two sessions (scaffolding + stimuli on one evening, model validation + extraction on the next), (b) simplify the scaffolding dramatically (see Critical Finding #2), or (c) acknowledge the schedule is 6-7 evenings, not 5. The Feasibility Checker's MVP timeline of 10-12 hours across 3 evenings (Tier 1 only) is a more honest baseline.

### 2. Implementation Is Over-Engineered by 6-10 Hours
- **Severity**: CRITICAL
- **Confidence**: HIGH
- **Source(s)**: Scope Auditor (Over-Engineering Findings, primary), Devil's Advocate (Alternative Perspectives, supporting)
- **Plan Section**: Phase 1 (Steps 1.1-1.5), project directory structure, all entry point scripts
- **Issue**: The plan specifies an enterprise-grade project structure (7 subpackages, 6 entry point scripts, Pydantic config system with YAML + CLI overrides, checkpoint/crash recovery system, HDF5 validation module, debug model config, inter-stage validation script, `run_all.py` orchestrator, 15 test files across 6 directories) for a solo research experiment that will be run perhaps 5-10 times. The Scope Auditor estimates this adds 6-10 hours of pure infrastructure development. The original idea could be implemented in ~1,350 lines across 5 files while preserving all scientific rigor.
- **Impact**: The engineering overhead transforms a "3-4 evenings of research" plan into a "5+ evenings of software development that happens to do research" plan. The infrastructure contributes zero scientific value and will never be reused.
- **Recommendation**: Adopt the Scope Auditor's simplified structure: (1) eliminate Pydantic config -- use a Python dict or dataclass; (2) eliminate the checkpoint system -- HDF5 incremental writes suffice; (3) reduce tests to critical-path-only (RDM computation, RSA correlation, anisotropy correction, pooling, GroupKFold stratification); (4) merge entry point scripts into one; (5) flatten to 4-5 modules: `stimuli.py`, `extraction.py`, `analysis.py`, `probes.py`, `viz.py`. This saves 6-10 hours while preserving all analytical rigor.

### 3. Permutation Test Computational Cost Is Underestimated by 5-10x
- **Severity**: CRITICAL
- **Confidence**: HIGH
- **Source(s)**: Devil's Advocate (Challenged Assumption #6, primary), Feasibility Checker (supporting via probe training estimate)
- **Plan Section**: Step 4.3 (RSA Computation), Step 4.5 (Stage 3 time estimate), Execution Schedule Evening 2
- **Issue**: The plan estimates Stage 3 (RSA + similarity analysis) at "1-2 hours." But the resolved conflict (Specialist Conflicts table, last row) calls for "permutation test at all layers with FDR correction." This means: 10,000 permutations x 64 layers x 2 correction methods x 3 model RDMs = 3,840,000 RSA correlations. Each requires Spearman correlation on ~320,000 pairs (~10ms each), totaling ~10.7 hours for Qwen alone. The plan's own Architect recommended peak-layer-only permutation testing, while the Domain Expert recommended all-layers testing; the plan adopted the Domain Expert's recommendation without updating the time estimate. Additionally, the Devil's Advocate notes that control task probe permutations (if following Hewitt & Manning's recommended 5-10 random permutations per probe) could multiply probe training time by 5-10x, pushing it to ~10 hours.
- **Impact**: If full permutation testing runs at all layers, it alone consumes an entire evening session. Combined with probe control permutations, the computational work could exceed available time by 2-3x.
- **Recommendation**: Adopt the Devil's Advocate's tiered permutation strategy: (a) run a fast 100-permutation screen at all layers to identify candidate significant layers, (b) run full 10,000-permutation tests only at the top-5 candidate layers plus the pre-registered peak layer. This reduces computation by ~95% while retaining comprehensive layer scanning and rigorous significance testing at key layers. For control task probes, specify exactly how many permutations (1, 5, or 10) and adjust the schedule accordingly.

### 4. Model Identity Unresolved -- Architecture Assumptions May Be Wrong
- **Severity**: CRITICAL
- **Confidence**: HIGH
- **Source(s)**: Devil's Advocate (Challenged Assumption #1), Feasibility Checker (Unverified Claims, Additional Findings)
- **Plan Section**: Steps 3.1-3.5 (entire extraction phase), zone boundary definitions (Step 5.3)
- **Issue**: "Qwen3.5-27B" does not exist as a public model. The plan assumes 64 layers and 3584 hidden dimensions, but the Feasibility Checker's architecture analysis suggests the closest available model (Qwen2.5-32B) has 64 layers and hidden_dim=5120, not 3584. If hidden_dim is 5120, VRAM usage increases ~43%, storage increases proportionally, PCA-whitening takes ~2x longer, and probe training is affected. The Devil's Advocate adds that GPTQ-quantized models from different providers may have different module naming, breaking forward hooks. The plan's own Open Question #1 flags this but provides no resolution pathway.
- **Impact**: Every numeric estimate in the plan (VRAM budget, extraction timing, storage, PCA time, probe training time) depends on the model's actual architecture. Getting this wrong does not break feasibility but cascades through all timing estimates. If hooks break due to unexpected module naming, debugging could consume an entire evening.
- **Recommendation**: Before any implementation, verify the exact HuggingFace model ID, layer count, hidden dimension, and module hierarchy. Update all numeric estimates. Budget 30-60 minutes for hook debugging on the actual model. The plan's 1-stimulus pilot (Step 3.1) is the right mechanism but needs explicit time allocated for debugging.

## Major Findings

### 1. No Intermediate Decision Gates After First Results
- **Severity**: MAJOR
- **Confidence**: HIGH
- **Source(s)**: Devil's Advocate (Gap #2, Risk Scenario #6)
- **Plan Section**: Execution Schedule Evening 2, Step 6.3 (Go/No-Go)
- **Issue**: The plan is a linear pipeline with a single go/no-go at the end. If H1 is clearly falsified after the first Qwen RSA results (Evening 2), there is no protocol for whether to proceed with probes, investigate stimulus quality, check for extraction bugs, or declare early termination. The most likely outcome (per the Risk Assessment) is a "qualified GO" that is difficult to interpret, but the plan does not discuss how to frame this.
- **Recommendation**: Add a lightweight sanity check after the first RSA results. Define: (a) if RSA correlation is flat across all layers, check stimuli and extraction before proceeding; (b) if middle-layer peak exists but is weak, proceed with probes; (c) if strong peak exists, proceed as planned. This costs 15 minutes and could save hours of wasted work.

### 2. GroupKFold + 40-Class Probe Creates Noisy Fold-to-Fold Metrics
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Devil's Advocate (Challenged Assumption #3)
- **Plan Section**: Step 5.1 (Probe Training Pipeline)
- **Issue**: With 80 products in 5 folds, each test fold has 16 products. For the 40-class real-product probe, each test fold has only ~8 real products. Macro-F1 is computed over only the classes present in the test set, which varies across folds. The probe's learned decision boundaries for the ~32 unseen classes per fold are never validated in that fold. This makes aggregate F1 potentially misleading and fold-to-fold comparison noisy.
- **Recommendation**: Document this limitation explicitly. Consider reporting micro-F1 alongside macro-F1. The Devil's Advocate also suggests supplementing with k-NN classification (k=1 or k=3), which makes no linear separability assumptions and directly tests whether same-product representations are nearest neighbors -- a stronger geometric claim for the "protocol layer" framing.

### 3. Stimulus Register Distinctiveness May Be Shallow
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Devil's Advocate (Challenged Assumption #5), Feasibility Checker (Hidden Complexity -- Step 2.4)
- **Plan Section**: Step 2.2 (Prompt Templates), Step 2.4 (Generation Orchestrator)
- **Issue**: Modern LLMs tend toward "helpful assistant" style regardless of register instructions. A patent-style prompt may produce text that reads like "a helpful explanation of a patent" rather than actual patent language. If register variation is shallow (vocabulary swaps rather than genuine structural/pragmatic differences), the experiment tests a weaker hypothesis than intended. The plan includes a 5% manual spot-check but no quantitative measure of register distinctiveness.
- **Recommendation**: Add a quantitative register distinctiveness check: compute mean pairwise register distance (e.g., Jaccard on token sets, or TF-IDF cosine) across products. Define a minimum threshold. If stimuli cluster by register more weakly than expected, iterate on prompts before proceeding. Budget prompt engineering iteration cycles explicitly.

### 4. Fictional-vs-Real Control Threshold (r > 0.7) Is Arbitrary
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Devil's Advocate (Challenged Assumption #8)
- **Plan Section**: Step 6.2 (Control Analyses)
- **Issue**: The Spearman r > 0.7 threshold for comparing fictional vs. real product RSA curves has no empirical basis. Without knowing the baseline split-half reliability of RSA curves (how much two disjoint subsets of real products agree), the threshold could be too lax or too strict. A correlation of 0.65 or 0.75 provides no interpretive guidance.
- **Recommendation**: Compute split-half reliability of the real-product RSA curve (split 40 real products into two subsets of 20, compute RSA on each, correlate). Use this as the empirical ceiling. The fictional-vs-real correlation should be compared against this baseline, not an arbitrary threshold.

### 5. Cross-Model Validation Has No Pre-Registered Interpretation
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Devil's Advocate (Gap #5)
- **Plan Section**: Execution Schedule (Llama as Tier 2), Step 6.2 (Control Analyses)
- **Issue**: If Qwen shows a clear protocol-layer effect but Llama does not, the plan provides no guidance on interpretation. Qwen and Llama differ simultaneously in size (27B vs 8B), architecture, training data, and quantization, making disagreement uninterpretable. The plan does not pre-register what "cross-model validation" means quantitatively.
- **Recommendation**: Pre-register what constitutes agreement (e.g., both show peak RSA in middle 60% of layers, both show H1 effect with p < 0.05) and what divergence means (architecture-specific vs. size-dependent vs. null effect). Acknowledge the confounds explicitly.

### 6. Partial RSA Length Correction Assumes Linearity
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Devil's Advocate (Challenged Assumption #7)
- **Plan Section**: Step 4.3 (RSA Computation), Trade-offs & Decisions (token range row)
- **Issue**: The plan allows 50-200 tokens and uses partial RSA to control for length. But partial RSA assumes the length confound is linear in its effect on pairwise distances. If short texts (tweets, 50 tokens) have qualitatively different representation properties than long texts (patents, 200 tokens) -- e.g., BOS/EOS effects dominating short texts -- then linear partial correlation will not fully remove the confound. The plan does not discuss nonlinear confound structures.
- **Recommendation**: Add a diagnostic: after the first RSA results, plot residual RSA correlation against stimulus length to check for remaining nonlinear patterns. If prompts target 80-150 but actual stimuli cluster tightly, note that partial RSA has limited length variance to work with.

### 7. Extraction Timing Estimate Is Too Optimistic
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Devil's Advocate (Challenged Assumption #2)
- **Plan Section**: Risk Assessment (Section 2.1), Execution Schedule
- **Issue**: The plan estimates ~0.5 sec/stimulus for extraction (7 minutes for 800 stimuli). But this assumes forward-pass-only time. The full pipeline (forward pass + hook capture at every layer + CPU transfer + mean pooling + gzip-compressed HDF5 write) is more realistically 2-5 seconds per stimulus, giving 25-65 minutes. This does not break the schedule on its own but compounds with other timing underestimates.
- **Recommendation**: Budget 30-60 minutes for Qwen 4-bit extraction, not 7-15 minutes. Adjust Evening 1 schedule accordingly.

### 8. Missing BoW Baseline Leakage Check
- **Severity**: MAJOR
- **Confidence**: MEDIUM
- **Source(s)**: Scope Auditor (Missing Items #1)
- **Plan Section**: Not covered; originally in the idea document (Section 7)
- **Issue**: The original idea explicitly included a "BoW baseline check; strip top-10 register-predictive tokens if needed" as risk mitigation for stimulus leakage. The draft plan dropped this. While the 40-class probe switch and within-category discrimination partially address the concern, the explicit BoW baseline from the original idea was never implemented.
- **Recommendation**: Add a lightweight BoW baseline: train a simple bag-of-words classifier on the same tasks (product, category, register). If BoW performance is high, the neural representations may be leveraging surface features. This is ~30 minutes of additional work and was part of the original design.

## Minor Findings

- **No NaN/Inf per-stimulus check during extraction**: The plan detects NaN/Inf after extraction (HDF5 validator) but not during. A single NaN stimulus silently corrupts all pairwise distances in the RDM. Add per-stimulus NaN check in the extraction loop. (Source: Devil's Advocate, Confidence: MEDIUM, Section: Step 3.5)

- **No artifact versioning strategy**: If Stage 2 is re-run after fixing a bug, old HDF5 files are overwritten and Stage 3 results become stale. No cache invalidation mechanism exists. (Source: Devil's Advocate, Confidence: LOW, Section: Steps 1.4, 3.5)

- **No GPT-4 variant specified or fallback for cross-generator control**: The plan does not specify which GPT-4 model (gpt-4, gpt-4-turbo, gpt-4o) or provide a fallback if the OpenAI API is unavailable. (Source: Devil's Advocate, Confidence: LOW, Section: Step 2.4)

- **Mean pooling vs. last-token pooling divergence not checked early**: If these two pooling methods produce divergent RSA results, the plan has no early diagnostic. Recommend comparing pooling methods on a subset after the first extraction. (Source: Devil's Advocate, Confidence: MEDIUM, Section: Steps 3.4, 6.4)

- **PCA-whitening timing unverified**: PCA on 800 x 3584 (or 800 x 5120) at 64 layers x 2 models could be slower than assumed. The plan does not profile this. (Source: Devil's Advocate, Confidence: LOW, Section: Step 4.1)

- **`auto-gptq` dependency should be optional**: Native transformers GPTQ loading (since v4.33) is preferred and reduces dependency risk. `auto-gptq` has had breaking changes between versions. (Source: Feasibility Checker, Confidence: MEDIUM, Section: Step 1.1)

- **9 visualization types exceed original scope**: The original idea called for 3 visualization types; the plan specifies 9 with a dedicated `style.py` module. (Source: Scope Auditor, Confidence: HIGH, Section: Step 6.4)

- **RTX 5090 vs RTX 3080 ambiguity**: Open Question #3 notes the user's machine profile says RTX 3080 but the experiment assumes RTX 5090. This must be resolved before implementation. (Source: Feasibility Checker, Confidence: HIGH, Section: Step 3.1)

- **Hidden state storage estimate too low**: The plan estimates ~2GB for all hidden states; the Feasibility Checker computes ~4GB compressed for all model configs with all components. (Source: Feasibility Checker, Confidence: MEDIUM, Section: Storage Budget)

## Contradictions Between Critics

| Critic A Position | Critic B Position | Assessment |
|---|---|---|
| **Devil's Advocate**: Debug config (Qwen2.5-1.5B) is inadequate as a development proxy because it has fundamentally different architecture than the 27B target. | **Scope Auditor**: Debug config is scope creep -- testing with random tensors or a small subset of stimuli on the real model is simpler and more useful. | Both are correct from different angles. The debug config is both insufficient (won't catch 27B-specific issues) AND over-scoped (downloading an additional model for limited benefit). **Resolution**: Skip the separate debug model. Test code logic with random tensors; test integration with the real model using 5 stimuli (the plan's 1-stimulus pilot, expanded slightly). |
| **Devil's Advocate**: Checkpoint/crash recovery is a genuine gap -- no versioning for intermediate artifacts, no stale-result detection. | **Scope Auditor**: Checkpoint/crash recovery system is over-engineered -- longest stage is ~30 minutes, rerunning from scratch is cheaper than building recovery. | The Scope Auditor is more persuasive for the checkpoint *system*. But the Devil's Advocate's concern about stale downstream artifacts is valid. **Resolution**: Skip the generalized checkpoint class. Use HDF5 incremental writes for extraction (already planned) and simple JSON saves for stimuli. Add a lightweight timestamp check at each stage entry: warn if input files are newer than output files. |
| **Feasibility Checker**: 5-evening schedule is achievable for Tiers 1-2, marginally feasible for Tiers 3-4. | **Scope Auditor**: 5-evening schedule requires 22-31 hours with current over-engineering; simplified version takes 12-18 hours. | Both agree the schedule is tight. The key insight is that the schedule's feasibility depends on implementation complexity. With the Scope Auditor's simplifications, Tiers 1-2 fit in 3 evenings and full scope fits in 4-5 evenings. Without simplifications, the plan likely needs 6-7 evenings. **Resolution**: Simplify the implementation to unlock the schedule. |

## Overall Assessment

- **Plan readiness**: Needs significant revision -- scientifically sound but implementation structure must be simplified and timing estimates corrected before execution begins.
- **Highest risk area**: Schedule -- the combination of over-engineered infrastructure, underestimated permutation testing costs, and an overloaded Evening 1 creates a situation where the researcher will likely run out of time before completing the full analysis. The tiered deferral strategy is excellent but will activate earlier than planned.
- **Strongest area**: Analytical design -- the RSA-primary, probes-secondary approach with control tasks, partial RSA, within-category discrimination, effect-size criteria, and permutation testing is thorough, well-justified, and appropriate for publishable results. All user decisions and domain expert recommendations are correctly reflected.
- **Number of findings**: 4 critical, 8 major, 9 minor

### Priority Actions Before Implementation

1. **Verify model identity** (Critical #4): Search HuggingFace for the exact Qwen GPTQ model, confirm layer count and hidden dim, update all estimates.
2. **Simplify implementation structure** (Critical #2): Adopt the Scope Auditor's 4-5 module structure. Eliminate Pydantic, checkpoint system, debug config, excessive test infrastructure. Save 6-10 hours.
3. **Fix permutation test scope** (Critical #3): Adopt tiered permutation strategy (100-permutation screen at all layers, 10,000-permutation test at top-5 + peak layer). Specify control task permutation count.
4. **Restructure Evening 1** (Critical #1): Either split into two sessions or rely on the simplified structure to fit within 4 hours.
5. **Add intermediate decision gate** (Major #1): Define what to do if first RSA results are null, weak, or strong.
6. **Calibrate memorization control threshold** (Major #4): Compute split-half reliability to replace the arbitrary r > 0.7.
7. **Add quantitative register distinctiveness check** (Major #3): Measure and threshold register variation before committing to the full experiment.
