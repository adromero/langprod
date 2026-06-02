# Feasibility Check: Brand Coherence Validation Plan

**Assessed by:** Feasibility Checker Agent
**Date:** 2026-03-28
**Plan assessed:** `06-draft-plan.md`

---

## Overall Feasibility: **MEDIUM**

The plan is technically sound and builds sensibly on a solid existing codebase, but several claims about existing functionality are imprecise, key new dependencies are not yet installed, and the effort estimates for the contrastive fine-tuning path understate the integration complexity. The sequential gating design is the plan's greatest strength — it ensures effort is not wasted on unvalidated foundations. The greatest risk is the contrastive fine-tuning step (Step 2), which is the hardest novel engineering and has the least existing infrastructure to build on.

---

## 1. Technical Feasibility

### 1.1 Verified Claims

| Claim | Verdict | Evidence |
|-------|---------|----------|
| `partial_rsa()` exists in `analysis.py` | **VERIFIED** | Lines 406-456. Accepts observed_rdm, model_rdm, nuisance_rdms. Uses OLS regression to remove nuisance, then Spearman on residuals. Tested in `tests/test_rsa.py::test_partial_rsa_removes_confound`. |
| Extraction pipeline supports forward hooks | **VERIFIED** | `ExtractionHooks` class in `extraction.py` (lines 140-236). Registers hooks on self_attn and mlp sub-modules of all transformer layers. Architecture-agnostic discovery via regex on `named_modules()`. |
| BoW baseline exists in `stimuli.py` | **VERIFIED** | `run_bow_baseline()` at line 1876. TF-IDF + logistic regression on product/category/register classification. 5-fold stratified CV. |
| 800 calibration stimuli exist | **VERIFIED** | `data/stimuli.json` contains stimulus records. Format: `{stimulus_id, product_id, category, register, variant, is_fictional, text, token_count, generator, core_attributes, generated_at}`. |
| Pipeline has working RSA analysis | **VERIFIED** | `cmd_analyze()` in `run.py` computes RDMs, RSA curves, partial RSA, permutation tests, condition similarities, and sanity checks. Output files (`rsa_product_identity.npy`, `rsa_register_identity.npy`, etc.) exist on disk. |
| Permutation testing is tiered (200 screen + 10000 full) | **VERIFIED** | `run_permutation_test_tiered()` in `analysis.py` lines 565-670. Configurable via `screen_permutations`, `full_permutations`, `top_k_layers_for_full_test`. |
| HDF5 extraction with gzip compression and resume | **VERIFIED** | `_init_h5_datasets()` and `_append_to_h5()` in `extraction.py`. Resume via `_get_existing_stimulus_ids()`. Compression opts = gzip level 4. |
| Model loads with GPTQ native support | **VERIFIED** | `load_model_and_tokenizer()` detects GPTQ by name convention, uses native transformers integration (>=4.40). |
| Existing tests validate claimed functionality | **VERIFIED** | 5 test files: `test_rdm.py`, `test_rsa.py`, `test_anisotropy.py`, `test_groupkfold.py`, `test_pooling.py`. Cover RDM computation, RSA correlation, partial RSA, permutation testing, anisotropy correction, pooling, and group K-fold splitting. |

### 1.2 Incorrect Claims / Discrepancies

| Claim | Issue | Severity |
|-------|-------|----------|
| **Plan says "extraction.py (add contrastive model path)"** | `extraction.py` currently only supports causal-LM models via `AutoModelForCausalLM`. A sentence-transformer encoder (E5, Jina) is NOT a causal-LM — it uses `AutoModel` or the `sentence-transformers` library. The hook-based extraction logic and mean-pool-no-special function are specific to causal-LM architecture. **A contrastive sentence-transformer would need a substantially different extraction path**, not a minor extension. | **HIGH** |
| **Plan says "analysis.py (add coherence RDM computation)"** | No `compute_coherence_rdm()` or `extract_contrastive_embeddings()` functions exist yet. The plan implies these are extensions, but the current `analysis.py` is RSA-focused with model RDMs based on metadata labels (product_id, category, register). A coherence RDM would need per-product pairwise channel-similarity computation — structurally different from the current label-based model RDMs. | **MEDIUM** — correctly flagged as "EXTEND" but understates the architectural difference. |
| **Plan says registers are "regulatory, marketing, retail, social, consumer_review"** | The existing pipeline uses 5 DIFFERENT registers: `marketing`, `regulatory`, `casual_social`, `patent`, `journalistic` (from `run.py` CONFIG). The plan's coherence channels do not map 1:1 to existing registers. The 800 calibration stimuli use the existing register set, not the coherence-experiment channels. | **MEDIUM** — the plan acknowledges this implicitly (calibration stimuli are "same product across registers"), but the register naming mismatch between calibration data and real-document channels needs explicit handling. |
| **Plan says "reuses existing BoW infrastructure" for TF-IDF baseline** | `run_bow_baseline()` in `stimuli.py` is a CLASSIFIER (TF-IDF + logistic regression for product/category/register classification), not a SIMILARITY metric. The coherence baselines need TF-IDF cosine similarity between document pairs — a fundamentally different computation. The existing function cannot be reused directly. | **MEDIUM** — the TF-IDF vectorizer code pattern can be adapted, but the claim of "reuse" is misleading. |
| **Plan says `probes.py` is "UNCHANGED"** | True for the code file itself, but the plan's Experiment 0 requires training register probes on contrastive-encoder representations to verify register suppression (accuracy < 0.7). The existing probes infrastructure reads from the HDF5 file produced by `extraction.py` with specific dataset names (`hidden_states_mean_no_special`). Contrastive-encoder embeddings would need a different HDF5 or adapter to feed into the existing probe pipeline. | **LOW** — straightforward to address but not "unchanged" in practice. |
| **Plan says `rsa_product_partial.npy` exists** | `run.py` computes and saves partial RSA product identity, but the file `data/rsa_product_partial.npy` does NOT exist on disk. The analyze step may not have been run with that section enabled, or it was cleared. | **LOW** — the code works; the artifact just needs regeneration. |

### 1.3 Technology Assessment

| Technology | Available? | Assessment |
|------------|-----------|------------|
| `sentence-transformers>=3.0` | **NOT INSTALLED** | Not in `pyproject.toml` dependencies or `.venv`. Must be added. The library is mature and well-maintained. No compatibility concerns with existing stack. |
| `trafilatura>=1.8` | **NOT INSTALLED** | Not in `.venv`. Needed for web document extraction. Standard library, no issues expected. |
| `beautifulsoup4>=4.12` | **NOT INSTALLED** | Not in `.venv`. Standard, no issues. |
| `peft>=0.10` (fallback only) | **NOT INSTALLED** | Needed only if Qwen2.5-7B LoRA fallback is activated. PEFT + LoRA is well-established. |
| PyTorch + CUDA | **INSTALLED** | `torch>=2.5` in pyproject.toml. RTX 5090 support requires CUDA 12.x with recent PyTorch nightly/stable — verify CUDA toolkit version. |
| `transformers>=4.40` | **INSTALLED** | In pyproject.toml. Supports native GPTQ loading. |
| `scipy`, `scikit-learn`, `numpy`, `h5py` | **INSTALLED** | All present in pyproject.toml. |

---

## 2. Complexity Analysis

### 2.1 Hardest Steps (ranked)

1. **Step 2: Contrastive Fine-Tuning (8-12h estimate)** — **HARDEST. Likely underestimated.**
   - The `sentence-transformers` library handles pair construction and training loop, but:
     - Constructing proper positive/negative pairs from the 800 stimuli requires careful product-aware sampling (80 products x 5 registers x 2 variants = 800 stimuli; positive pairs = same product, different register; negatives = different product).
     - The `MultipleNegativesRankingLoss` treats in-batch negatives, which may be insufficient for 80 products. May need hard-negative mining.
     - The `evaluate_register_factoring()` function (verify register probe drops below 0.7) requires integrating with the existing probe infrastructure — which expects HDF5 input from `extraction.py`, not sentence-transformer embeddings.
     - If the sentence-transformer path fails, the Qwen2.5-7B LoRA fallback adds significant complexity: gradient checkpointing, PEFT integration, custom loss function, memory pressure monitoring on 32GB VRAM.
   - **Realistic estimate: 12-20h** including debugging and evaluation integration.

2. **Step 7: Experiment 1 — Real-Document Sensitivity (8-12h estimate)** — **High complexity, estimate reasonable if documents are already collected.**
   - The 2-5 day document collection is external and cannot be parallelized with implementation.
   - Statistical analysis (Mann-Whitney U, AUC, Cohen's d) is straightforward with scipy/sklearn.
   - The face-validity check with 2-3 industry professionals adds calendar time, not implementation time.
   - Main risk: document preprocessing quality directly affects results. The `ingest.py` module is completely new.

3. **Step 9: Experiment 3 — Attribute-Level Drill-Down (6-8h estimate)** — **High hidden complexity.**
   - Generating attribute probes that test semantic similarity (not keyword matching) is non-trivial.
   - The plan acknowledges the "keyword domination" concern but the mitigation (synonym/paraphrase probes) is labor-intensive to validate.
   - Computing attribute-channel similarity requires a clear definition of "attribute presence" in embedding space.

4. **Step 5: Experiment 0 — Metric Exploration (6-8h estimate)** — **Moderate complexity, estimate reasonable.**
   - Systematic grid search over (correction x layer x aggregation) is straightforward.
   - Length variation and attribute removal simulation on existing stimuli is mechanical.
   - The "manually rewrite 10-20 stimuli" sub-task adds unaccounted labor.

### 2.2 Steps That Seem Simple But Aren't

- **Step 1: Document Ingestion (4-6h)** — HTML cleaning, boilerplate removal, and multi-product detection are each rabbit holes. Amazon listing chrome, regulatory filing formatting, social media markup all have edge cases. `trafilatura` helps but doesn't solve everything. Consider 6-8h.
- **Step 12: CLI Integration (2-3h)** — Adding subcommands is simple, but the gating logic (checking verdict.json, metric_selection.json existence) and the shared config management (coherence config vs. existing CONFIG) add complexity. 3-5h is more realistic.

---

## 3. Dependency Assessment

### 3.1 Dependency Ordering

The plan's sequential gating is sound:

```
Phase 0 (Steps 1,2,3,4 parallel) → Phase 1 (Step 5) → Phase 2 (Steps 6,7) → Phase 3 (Steps 8,9) → Phase 4 (Steps 10,11) → Phase 5 (Step 12)
```

**One ordering issue:** Step 2 (contrastive fine-tuning) should be completed BEFORE Step 5 (Experiment 0) can meaningfully begin, since Experiment 0 needs to evaluate the contrastive model. The plan states Steps 1, 2, and 4 are parallel within Phase 0, but Step 5 cannot start until Step 2 finishes. This is acknowledged in the critical path but worth emphasizing.

### 3.2 Circular Dependencies

None detected. The plan correctly separates:
- Calibration data (existing) from real-document data (new)
- Metric exploration (Exp 0) from metric validation (Exp 1)
- Each experiment writes independent artifacts

### 3.3 External Dependency Risks

| Dependency | Risk | Mitigation |
|------------|------|------------|
| `sentence-transformers` library | LOW | Mature, actively maintained, compatible stack |
| Jina/E5 model weights on HuggingFace | LOW | Publicly available, can be cached locally |
| Real-document collection (2-5 days) | **HIGH** | Bottleneck is human effort; quality varies; some channels may be inaccessible (FDA filings require EDGAR scraping). Plan correctly identifies this as blocking. |
| Industry professional review (Exp 1, 5) | **MEDIUM** | Requires human availability. Could block calendar time significantly. |
| Historical documents for Exp 4 | **MEDIUM** | Wayback Machine coverage is inconsistent. Some products may not have accessible archives. |

---

## 4. Resource Requirements

### 4.1 Hardware

| Resource | Required | Available | Assessment |
|----------|----------|-----------|------------|
| GPU VRAM for sentence-transformer fine-tuning | ~4-8 GB | 32 GB (RTX 5090) | **MORE THAN SUFFICIENT** |
| GPU VRAM for Qwen2.5-32B extraction | ~20-24 GB (GPTQ-Int4) | 32 GB | **SUFFICIENT** with headroom |
| GPU VRAM for Qwen2.5-7B LoRA fallback | ~12-16 GB | 32 GB | **SUFFICIENT** |
| System RAM for 800-stimulus RSA | ~4-8 GB | 32 GB | **SUFFICIENT** |
| Disk for HDF5 + model checkpoints | ~10-20 GB | Assumed sufficient | Check available space |

### 4.2 Skills

| Skill | Assessment |
|-------|------------|
| `sentence-transformers` fine-tuning | Moderate learning curve; good documentation exists |
| Document scraping/cleaning | Moderate; domain-specific edge cases |
| Statistical analysis (Mann-Whitney, AUC, Cohen's d) | Standard; scipy/sklearn have all needed functions |
| RSA methodology | Already demonstrated in existing codebase |

### 4.3 Access

| Access | Required | Status |
|--------|----------|--------|
| HuggingFace model hub | Sentence-transformer models, Jina/E5 | Assumed available |
| Real product documents | FDA filings, marketing pages, Amazon listings, social media | Requires manual collection |
| Industry professionals for face-validity | 2-3 people for Exps 1 and 5 | **NOT CONFIRMED** — plan flags as open question |

---

## 5. Codebase Discrepancies (Summary)

### Critical

1. **Extraction pipeline cannot directly support sentence-transformer models.** The plan claims `extraction.py` will be "extended" with a contrastive model path, but the entire extraction pipeline (hooks, pooling, HDF5 layout) is designed for causal-LMs. A sentence-transformer uses a fundamentally different architecture (encoder-only, CLS pooling or mean pooling, no causal attention). This requires either:
   - A parallel extraction function for sentence-transformers (cleanest approach), OR
   - Restructuring `extraction.py` to be architecture-agnostic (risky, regression-prone)

   The plan's architecture diagram shows the contrastive encoder feeding directly into the existing extraction pipeline, which is architecturally incorrect.

### Notable

2. **Register taxonomy mismatch.** Existing stimuli use `{marketing, regulatory, casual_social, patent, journalistic}`. The coherence experiments define channels as `{regulatory, marketing, retail, social, consumer_review}`. The contrastive fine-tuning on calibration data trains on the existing register taxonomy — the resulting model must generalize to the coherence channel taxonomy (retail, social, consumer_review are not in the training data). This transfer assumption is untested.

3. **BoW baseline is a classifier, not a similarity metric.** The plan's claim of "reusing existing BoW infrastructure" for TF-IDF coherence baseline overstates the reuse.

4. **Missing `rsa_product_partial.npy`.** The partial RSA code exists and works (tested), but the output artifact has not been generated. Minor — just needs a pipeline re-run.

5. **No `coherence/` package exists yet.** The plan proposes 15 new Python files in a new `coherence/` package. None exist. This is expected for a "NEW" plan, but the total new-code volume (~2000-3000 lines estimated) is substantial.

---

## 6. Effort Realism

| Phase | Plan Estimate | Feasibility Assessment | Adjustment |
|-------|---------------|----------------------|------------|
| Phase 0: Infrastructure | 17-25h | Step 2 underestimated by 4-8h; Step 1 underestimated by 2h | **23-35h** |
| Phase 1: Experiment 0 | 6-8h | Reasonable if contrastive model works. Manual rewriting adds 2-4h. | **8-12h** |
| Phase 2: Reporting + Exp 1 | 12-18h | Reasonable for implementation; document collection is external | **12-18h** (plus 2-5 days collection) |
| Phase 3: Exps 2-3 | 8-11h | Experiment 3 attribute-probe complexity may add 2-4h | **10-15h** |
| Phase 4: Exps 4-5 | 8-12h | Historical document access is the main risk | **8-14h** |
| Phase 5: Integration | 2-3h | Gating logic is slightly more complex than estimated | **3-5h** |
| **Total** | **53-77h** | | **64-99h** |

The plan's 53-77h estimate is optimistic by roughly 20-30%. A more realistic range is **64-99 hours** of implementation, plus document collection time.

---

## 7. Verdict

**MEDIUM FEASIBILITY — Proceed with revisions.**

The plan is well-structured, scientifically rigorous, and builds intelligently on a solid existing codebase. The sequential gating design is excellent and prevents wasted effort. However:

### Must-Fix Before Proceeding

1. **Revise the extraction architecture for sentence-transformers.** The plan must explicitly design a separate extraction path for encoder models, not assume `extraction.py` can be trivially extended. Recommend a `coherence/extraction.py` that handles sentence-transformer embedding computation independently, writing to a compatible HDF5 format.

2. **Address the register-to-channel taxonomy mismatch.** Explicitly document how contrastive training on `{marketing, regulatory, casual_social, patent, journalistic}` generalizes to real-document channels `{regulatory, marketing, retail, social, consumer_review}`. Consider whether additional training data for unrepresented channels (retail, consumer_review) is needed.

3. **Install missing dependencies before estimating Phase 0 effort.** Verify `sentence-transformers`, `trafilatura`, and `beautifulsoup4` install cleanly in the existing `.venv` without version conflicts.

### Recommended Adjustments

4. Increase Step 2 (contrastive fine-tuning) estimate to 12-20h.
5. Add a "Step 0.5: Environment Setup" task (1-2h) for installing and verifying new dependencies plus CUDA compatibility with sentence-transformer inference.
6. Clarify that `run_bow_baseline()` in `stimuli.py` is NOT reusable for the TF-IDF coherence baseline; `coherence/baselines.py` is genuinely new code.
7. Plan total effort should be quoted as 64-99h, not 53-77h.

### Strengths

- Sequential gating prevents wasted effort on unvalidated methodology
- Fallback paths (partial RSA if contrastive fails; simpler baselines if hidden states add no value)
- Pre-registration discipline (metric lock before real-document testing)
- Clear kill criteria at each stage
- Existing codebase is well-tested and well-documented
- Hardware is more than sufficient for all planned workloads
