# Reconciliation: Draft Plan vs. Red-Team Findings

## Executive Summary

The draft plan has strong experimental design but overbuilds infrastructure and makes optimistic assumptions about the contrastive fine-tuning path. Of 19 findings (5 Critical, 6 Major, 7 Minor, 1 Contradiction), I resolve 12 directly with plan changes, flag 4 for user decision (genuine trade-offs), and dismiss 3 as already handled or non-blocking. The net effect: a leaner plan (~45-65h), a mandatory real-document smoke test before infrastructure investment, a separate extraction module for sentence-transformers, and tighter Experiment 0 methodology. No experiments are removed; sequencing and gating are preserved as strengths.

---

## Resolved Findings (Changes Applied)

### R1: Add real-document smoke test before Phase 0 (resolves C1)

**Finding:** Contrastive fine-tuning on LLM-generated stimuli may not generalize. All three critics converge here.

**Decision:** Insert a new Step 0 ("Smoke Test") before any infrastructure work. 2-4 hours, zero new code.

**Specific changes:**
- Pick 2 real products with obvious coherence contrast (e.g., Tylenol vs. a startup supplement)
- Collect 3 documents each from distinct channels (6 documents total)
- Run through existing Qwen pipeline with partial RSA at layer 61
- Check: does within-product similarity exceed between-product similarity?
- **Kill criterion:** If the raw signal is not even directionally present in real documents with the existing pipeline, the entire contrastive approach needs rethinking before investing 17-25h in infrastructure
- **Effort:** 2-4h. No new modules. Uses existing `extraction.py` and `analysis.py`.

This is the single highest-value change. It derisk 25+ hours of infrastructure work with 3 hours of manual effort.

### R2: Separate extraction module for sentence-transformers (resolves C2)

**Finding:** `extraction.py` is built around `AutoModelForCausalLM` with causal-LM-specific hooks. Sentence-transformer encoders are architecturally incompatible.

**Decision:** Create `coherence/extraction.py` as a standalone module for sentence-transformer embedding computation. Do NOT extend the existing `extraction.py`.

**Specific changes:**
- New `coherence/extraction.py` with `extract_embeddings(texts, model_path)` returning numpy arrays
- Uses `sentence-transformers` library's built-in `model.encode()` — no forward hooks needed
- The existing `extraction.py` remains untouched (no "EXTEND" needed for the contrastive path)
- The base Qwen partial-RSA fallback path still uses existing `extraction.py`
- Revise Step 2 effort estimate from 8-12h to 12-18h (accounts for new module + testing)

### R3: Held-out split in Experiment 0 (resolves C4, J4)

**Finding:** Using 800 calibration stimuli for both contrastive training AND metric exploration is circular. 128+ metric combinations create overfitting risk.

**Decision:** Split calibration data 640/160 (train/validation). Reduce metric candidate space.

**Specific changes:**
- 640 stimuli for contrastive fine-tuning training
- 160 stimuli held out for Experiment 0 Part A metric exploration
- Reduce candidate space: test only {contrastive, partial-RSA-whitened} x {mean_pairwise, centroid_distance, silhouette} = 6 combinations (down from 128+). Justify the short list: mean_pairwise is the natural default, centroid_distance is robust to outliers, silhouette captures cluster separation. Others are variants unlikely to diverge materially.
- Apply Bonferroni correction when selecting the winner (p-threshold = 0.05/6 = 0.008)
- Record the full 6-way comparison in `metric_selection.json` alongside the winner

### R4: Align register taxonomy between calibration and real channels (resolves C5)

**Finding:** Calibration uses `{marketing, regulatory, casual_social, patent, journalistic}`. Real channels are `{regulatory, marketing, retail, social, consumer_review}`. Three real channels have no training counterpart.

**Decision:** Generate additional calibration stimuli for the missing registers before contrastive training.

**Specific changes:**
- Generate 160 additional calibration stimuli (80 products x 2 new registers: `retail` and `consumer_review`) using the same LLM pipeline as the original 800
- Remap existing `casual_social` to `social` in the training data
- Drop `patent` and `journalistic` from training (not in real channel taxonomy). These 320 stimuli become the held-out set for Experiment 0 instead of random splitting, which is even better: train on {marketing, regulatory, retail, social, consumer_review} and validate on the held-out patent/journalistic registers to test actual transfer
- **Revised split:** Train on 800 original + 160 new (960 total, 5 relevant registers), validate on a 160-stimulus held-out subset stratified across products
- This adds ~2-3h to Step 2 for stimulus generation and register remapping
- Net effect: the contrastive model sees every real-channel register during training

### R5: Strip commercial/productization infrastructure (resolves J1)

**Finding:** 15-file package with reporting, BCI scale, CLI subcommands, commercial framing, edge cases, and client data protocols violates seed document's anti-productization directive. 15-22h of premature effort.

**Decision:** Strip to validation-grade tooling. Defer all commercial infrastructure to post-validation.

**Specific changes:**
- **Keep:** `coherence/extraction.py` (new), `coherence/contrastive.py`, `coherence/losses.py`, `coherence/metrics.py`, `coherence/baselines.py`, `coherence/ingest.py`, one script per experiment (`experiment_0.py` through `experiment_5.py`)
- **Drop:** `coherence/pipeline.py`, `coherence/reporting.py`, `coherence/viz.py`, CLI subcommands in `run.py`, BCI 0-100 scale, commercial framing section, edge case taxonomy, client data protocol (M8), model migration plan (Open Question 6)
- **Drop from draft plan text:** "Commercial framing" section (lines 547-554), "Edge cases to scope" section (lines 556-564), M8/M9 risk mitigations, Open Questions 7-10
- Each experiment script runs standalone: `python -m coherence.experiment_0`, etc.
- Sequential gating is enforced by checking for `verdict.json` files at the start of each experiment, not by an orchestrator
- Estimated savings: 10-15h (reporting 4-6h + pipeline/CLI 2-3h + commercial deliverables ~4-6h)

### R6: Revise effort estimates (resolves J2)

**Finding:** Estimates are optimistic by 20-40%. Document collection is underestimated.

**Decision:** Revise estimates incorporating scope reduction (R5) and architecture fix (R2).

**Revised effort table:**

| Phase | Steps | Original Est. | Revised Est. | Notes |
|-------|-------|---------------|--------------|-------|
| Pre-Phase: Smoke Test | 0 | (new) | 2-4h | New step R1 |
| Phase 0: Infrastructure | 1, 2, 3, 4 | 17-25h | 20-28h | Step 2 increased per R2, R4 |
| Phase 1: Experiment 0 | 5 | 6-8h | 6-8h | Unchanged |
| Phase 2: Experiment 1 | 7 | 8-12h | 8-12h | Reporting stripped |
| Phase 3: Exps 2-3 | 8, 9 | 8-11h | 8-11h | Unchanged |
| Phase 4: Exps 4-5 | 10, 11 | 8-12h | 8-12h | Unchanged |
| **Total implementation** | | **53-77h** | **52-75h** | Scope reduction offsets estimate increases |
| **Document collection** | | **2-5 days** | **40-80h** | Made explicit per J2 |

Calendar time: 4-6 weeks (more realistic than original 3-5 weeks, accounts for collection).

### R7: Strengthen Experiment 1 statistical methodology (resolves J3)

**Finding:** Power is overstated at d=1.0; AUC >= baseline + 0.05 is meaningless at n=20.

**Decision:** Report minimum detectable effect size. Replace the 0.05 AUC threshold.

**Specific changes:**
- Add to Experiment 1 pre-registration: "At n=10 per group, Mann-Whitney U achieves 80% power for Cohen's d >= 1.0 (large effect). The minimum detectable effect at 80% power is approximately d=1.05. This experiment is NOT powered to detect medium effects (d=0.5-0.8). A non-significant result does not prove the metric is useless, only that the effect may be smaller than detectable at this sample size."
- Replace "AUC >= baseline + 0.05" with a DeLong test for comparing two correlated AUCs (p < 0.10, one-tailed). If the DeLong test is not significant, report the AUC difference descriptively but do not claim value-added.
- Keep AUC >= 0.85 as the absolute threshold (this is about the metric working, not about beating baselines)

### R8: Resolve consumer review aggregation (resolves J6)

**Finding:** Concatenating 5-10 reviews creates a document dominated by verbose reviews. Sampling strategy is unresolved but blocks document collection.

**Decision:** Embed reviews individually, use mean embedding. Fixed count.

**Specific changes:**
- Collect 10 consumer reviews per product: most recent 10 verified-purchase reviews with >= 50 words
- Embed each review individually through the locked metric
- The "consumer review" channel embedding = mean of the 10 individual embeddings
- Do NOT concatenate reviews into a single document
- Add this specification to `coherence/ingest.py` as a `aggregate_reviews()` function
- This resolves Open Question 5 from the draft plan

### R9: Document chunking strategy (resolves J5)

**Finding:** Chunking with mean-pooling obscures which document section carries the coherence signal.

**Decision:** Use the long-context encoder (Jina v3, 8192 tokens) as primary. Truncate documents exceeding 8192 tokens with a warning rather than implementing elaborate chunking.

**Specific changes:**
- Before document collection, run a length census on the target document types. Based on the domain analysis, most documents (marketing, retail, social, consumer review) are well under 8192 tokens. Only regulatory filings routinely exceed this.
- For documents > 8192 tokens: truncate to first 8192 tokens, log a warning, record the truncation in metadata
- Do NOT implement overlapping-window chunking infrastructure. If >20% of documents are truncated, revisit this decision.
- Remove the elaborate chunking specification from Step 1 (M2). Keep the `chunk_document()` function stub but implement it as simple truncation.
- This saves 1-2h of chunking implementation and avoids the mean-pooling interpretation problem

### R10: BoW baseline is new code, not reuse (resolves Minor #5)

**Finding:** `run_bow_baseline()` is a classifier, not a similarity metric. The plan's claim of "reuse" is misleading.

**Decision:** Acknowledge `coherence/baselines.py` is entirely new code. Revise Step 4 description.

**Specific change:** Remove "reuses existing BoW infrastructure" from Step 4 description. The TF-IDF coherence baseline computes document-level TF-IDF vectors and cosine similarity, which shares no code with the existing BoW classifier.

### R11: Generate missing partial RSA artifact (resolves Minor #6)

**Finding:** `rsa_product_partial.npy` has not been generated. Needed for Experiment 0 fallback path.

**Decision:** Add to pre-Phase 0 checklist: run the existing pipeline to generate `rsa_product_partial.npy`.

**Specific change:** Add to Smoke Test (Step 0) checklist: "Verify `data/rsa_product_partial.npy` exists. If not, run `python run.py rsa-partial` to generate it (~10 min)."

### R12: Add Experiment 0 failure protocol (resolves Minor #7)

**Finding:** Experiment 0 has no failure protocol. If both contrastive and partial RSA fail, there is no next step.

**Decision:** Add explicit failure protocol for Experiment 0.

**Specific change to Step 5:**
- If the best metric combination achieves d < 0.3 (trivial effect): **Kill the project.** The methodology cannot separate product from register on the calibration data. Pivot to a fundamentally different approach (e.g., supervised coherence predictor, or use an off-the-shelf NLI model for claim-level matching).
- If 0.3 <= d < 0.5 (small effect): Escalate to Qwen2.5-7B LoRA with adversarial register suppression. Budget an additional 8-12h. If this also fails, kill.
- If d >= 0.5 but degradation tests fail: The metric works on clean calibration data but is fragile. Investigate which degradation is problematic and address in preprocessing before proceeding to Experiment 1.

---

## Flagged for User Decision

### F1: Ground-truth methodology for Experiment 1 group assignment (C3)

**Finding:** "Known consistent" vs. "known inconsistent" criteria are subjective, potentially confounded with brand maturity/category regulation.

**Options:**

**(A) Formal inter-rater protocol (recommended):** Write operational definitions with specific observable evidence. Have 2 independent raters assign products. Require Cohen's kappa >= 0.6 for agreement. Document confound risks. Adds 4-8h of pre-study work but makes Experiment 1 interpretable.

**(B) Expert judgment with documentation:** The researcher makes all assignments with written justification per product. Faster (1-2h) but introduces single-rater bias. Acceptable for a research pipeline; unacceptable for a publishable study.

**(C) Hybrid:** The researcher makes initial assignments, one additional rater (colleague or domain contact) independently rates the same products. Compute agreement. If kappa < 0.4, revisit criteria. Moderate effort (3-5h).

**My recommendation:** Option C. This is a validation study, not a peer-reviewed publication. Perfect inter-rater protocol is overkill for the current stage, but single-rater is a genuine vulnerability the critics correctly identified. One additional rater is a reasonable middle ground.

### F2: Register probe threshold for contrastive model (Minor #1)

**Finding:** Probe accuracy < 0.7 means register is still classifiable 70% of the time. Near-chance (0.2 for 5 registers) would be more convincing.

**Options:**

**(A) Keep < 0.7 threshold.** Contrastive fine-tuning should suppress register, not eliminate it entirely. Some register information is useful (a regulatory filing SHOULD embed differently from a tweet even after content alignment). The goal is suppression, not annihilation.

**(B) Tighten to < 0.4.** Register should be nearly unclassifiable if the contrastive model is doing its job. This is a stricter test of register factoring.

**(C) Use relative improvement.** Require register probe accuracy to drop by >= 50% relative (from 1.0 to <= 0.5). This is more interpretable than an absolute threshold.

**My recommendation:** Option C (relative drop >= 50%, i.e., probe accuracy <= 0.5). The 0.7 threshold is indeed too lenient — a model that lets you classify register 70% of the time has not meaningfully suppressed it. But near-chance may be too aggressive and could indicate representation collapse. A 50% relative drop balances suppression with preserving useful structure.

### F3: Experiment 3 synonym test validity (Minor #3)

**Finding:** Testing synonyms tests vocabulary knowledge, not whether the metric measures coherence vs. lexical overlap. A proper control requires probing the correct attribute for the wrong product.

**Options:**

**(A) Add wrong-product control.** For each attribute probe, also compute similarity against documents for a DIFFERENT product in the same category. If the probe is equally similar to wrong-product documents, the metric is capturing category-level features, not product-specific content. Adds ~1h to Experiment 3.

**(B) Keep synonym test as-is, note limitation.** The synonym test is a reasonable first pass. Wrong-product controls are more rigorous but may not be worth the effort if Experiment 3 is already passing.

**My recommendation:** Option A. This is low-effort and directly addresses the critique. Add it.

### F4: Scope contradiction — effort vs. scope reduction (Contradiction)

**Finding:** Scope Auditor says "build less" (38-55h). Feasibility Checker says "budget more for what you're building" (64-99h). The synthesis resolved this as ~45-65h.

**My resolution:** I've applied both corrections in R5 and R6. The revised plan strips commercial infrastructure (saving 10-15h) but increases Step 2 estimates (adding 4-6h) and adds the smoke test (2-4h). Net implementation: 52-75h. Document collection adds 40-80h. The user should confirm this budget is acceptable given their other commitments.

**Specific question for the researcher:** The revised plan is ~52-75h implementation + 40-80h document collection, spread over 4-6 weeks part-time. Is this budget acceptable, or should we scope further (e.g., drop Experiments 4-5 from the initial plan)?

---

## Dismissed Findings

### D1: BCI 0-100 scale compression (Minor #2)

**Reasoning:** Already resolved by R5 (strip commercial infrastructure). The BCI scale is dropped entirely from the validation plan. It can be designed post-validation when real-document score distributions are known.

### D2: Expert validation conflates brand familiarity with coherence (Minor #4)

**Reasoning:** This is a valid concern for Experiment 5's expert evaluation, but the plan already mitigates it: experts predict rankings BEFORE seeing metric results (forced-choice, pre-registered). Familiarity bias affects both the expert and the metric equally — if experts rate familiar brands as "more coherent" and the metric agrees, that is either genuine correlation or shared bias, which is interpretable either way. No plan change needed. Note: if Experiment 5 results are suspicious, re-examine this as a post-hoc explanation.

### D3: Orchestration approach (Conflict 2 in draft plan)

**Reasoning:** Moot. R5 eliminates the CLI orchestration. Each experiment runs as a standalone script. The `run.py` extension is dropped. No conflict to resolve.

---

## Emergent Issues

### E1: The plan's load-bearing path has three untested links in sequence

Looking across findings, a pattern emerges: the contrastive fine-tuning (C1, C2, C5) depends on LLM-generated stimuli (C4, J4) evaluated on their own training distribution (C4) with register categories that don't match reality (C5). Each link individually might work; the chain failing at ANY point wastes everything built after it.

**Mitigation already applied:** The smoke test (R1) tests the END of the chain (does the signal exist in real documents?) before building the BEGINNING of the chain (contrastive infrastructure). The held-out split (R3) and register alignment (R4) strengthen the middle links. But the user should understand: even with these fixes, the contrastive path is a research bet. The partial RSA fallback exists for a reason.

### E2: Document collection is the actual bottleneck, not implementation

Multiple findings (J2, J6, C3) point to document collection as the underestimated dependency. Collecting 60-100 real documents, assigning ground-truth labels, sampling consumer reviews, and verifying channel coverage per product is manual, time-consuming work that cannot be parallelized with coding. The 40-80h estimate from the review is more realistic than the original "2-5 days."

**Recommendation:** Begin document collection informally during Phase 0 infrastructure work. Even before the smoke test, the researcher can start identifying candidate products and bookmarking sources. This parallelizes naturally.

### E3: The plan is methodologically sound but execution-fragile

All three critics praised the experimental design (sequential gating, pre-registration, kill criteria). The issues are ALL in implementation assumptions: wrong extraction architecture, training/test overlap, mismatched registers, undefined ground truth, optimistic estimates. This suggests the plan was designed by someone thinking about the science and reviewed by someone who has not yet implemented it. The gap is between "what the experiments test" (good) and "what the code actually does" (underspecified).

**Recommendation:** Before coding each Phase, write a 1-page implementation spec that a different person could execute. This forces the architectural decisions to be made explicit rather than discovered during coding.

---

## Reconciled Plan Changes Summary

Ordered by execution sequence:

1. **Add Step 0: Real-Document Smoke Test** (2-4h) — Pick 2 products, 3 docs each, run existing pipeline, check directional signal. Kill criterion: no directional signal = rethink approach before investing. [resolves C1]

2. **Generate partial RSA artifact** — Run existing pipeline to produce `rsa_product_partial.npy` if missing. [resolves Minor #6]

3. **Generate additional calibration stimuli** for `retail` and `consumer_review` registers (160 new stimuli). Remap `casual_social` to `social`. [resolves C5]

4. **Create separate `coherence/extraction.py`** for sentence-transformer embeddings. Do not extend existing `extraction.py`. [resolves C2]

5. **Split calibration data** 800/160 (or use register-based split: real-channel registers for training, patent/journalistic for transfer validation). [resolves C4, J4]

6. **Reduce metric candidate space** to 6 combinations with Bonferroni correction. [resolves C4]

7. **Add Experiment 0 failure protocol** with kill/escalate/investigate decision tree. [resolves Minor #7]

8. **Strip commercial infrastructure**: drop `reporting.py`, `viz.py`, `pipeline.py`, CLI subcommands, BCI scale, commercial framing, edge cases, client data protocol. [resolves J1, Minor #2]

9. **Resolve consumer review aggregation**: embed individually, mean-pool embeddings, 10 reviews per product. [resolves J6]

10. **Simplify chunking**: truncation at 8192 tokens with warning instead of overlapping windows. [resolves J5]

11. **Strengthen Experiment 1 statistics**: report minimum detectable effect size, replace AUC+0.05 with DeLong test. [resolves J3]

12. **Acknowledge `baselines.py` is new code**, not reuse of existing BoW. [resolves Minor #5]

13. **Revise effort estimates**: 52-75h implementation + 40-80h document collection, 4-6 weeks calendar. [resolves J2]

---

## Remaining Open Questions

1. **Ground-truth rater protocol** — Awaiting user decision on F1 (Options A/B/C). Recommend Option C (hybrid).

2. **Register probe threshold** — Awaiting user decision on F2. Recommend Option C (relative drop >= 50%).

3. **Experiment 3 wrong-product control** — Awaiting user decision on F3. Recommend Option A (add it).

4. **Budget confirmation** — 52-75h implementation + 40-80h collection acceptable? Or scope further?

5. **Base encoder selection** — Still need to benchmark Jina v3 vs. E5-large vs. BGE on calibration data. This is resolved naturally during Step 2 of the implementation.

6. **Number of manually rewritten stimuli for non-LLM transfer test** — Original Open Question 2 from draft plan remains unresolved. Recommend 10 as sufficient for directional signal.

7. **Minimum commercially relevant effect size** — Original Open Question 3 remains. Defer to post-Experiment-1 when real score distributions are available.

---

COMPLETED
