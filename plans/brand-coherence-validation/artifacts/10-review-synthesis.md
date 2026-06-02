# Review Synthesis: Brand Coherence Validation Plan

## Summary

Three independent red-team reviewers (Devil's Advocate, Scope Auditor, Feasibility Checker) assessed the draft validation plan. The plan's experimental design is well-structured — sequential gating, pre-registration, explicit kill criteria, and fallback paths are genuine strengths recognized by all three. However, all three converge on a central concern: the contrastive fine-tuning strategy introduces a new, largely untested dependency that the plan treats as a straightforward extension of existing infrastructure when it is architecturally novel. The plan also over-builds for validation (15-file package, reporting infrastructure, commercial framing) when the seed document explicitly prohibits premature productization. Effort estimates are optimistic by 20-40% depending on scope corrections.

**Finding counts:** 5 Critical, 6 Major, 7 Minor, 1 Contradiction.

---

## Critical Findings

### C1: Contrastive fine-tuning on LLM-generated stimuli may not generalize to real documents

- **Severity:** CRITICAL
- **Confidence:** HIGH (3/3 critics agree)
- **Sources:** Devil's Advocate (Assumptions 1, 2, 3; Failure Mode 2), Feasibility Checker (Discrepancy 1, Section 2.1), Scope Auditor (implicit in ground-truth gap)
- **Plan sections:** Step 2 (contrastive fine-tuning), Experiment 0
- **Issue:** The 800 calibration stimuli are LLM-generated, 80-150 words, with 5 stylized register variants. Real documents have fundamentally different register signatures (FDA boilerplate, Amazon listing chrome, tweet syntax, legal filing structure). The contrastive model will learn to suppress LLM-register artifacts, not real-world register variation. The register probe accuracy test (Experiment 0) evaluates on the same distribution as training, so this failure mode is undetectable until Experiment 1 — after 25-35 hours of infrastructure investment.
- **Impact:** If the contrastive model fails on real documents, the entire infrastructure built around it is wasted. The fallback (partial RSA) returns to the original problem.
- **Recommendation:** Add a minimal real-document smoke test (2-4 hours, zero new code) before building any infrastructure. Take 2 real products, collect 3 documents each, run through the existing Qwen pipeline with partial RSA, and verify the signal is directionally useful. This fills the gap between Experiment 0 (calibration) and Experiment 1 (full study). All three critics effectively advocate for this.

### C2: Extraction pipeline cannot support sentence-transformer models without major rework

- **Severity:** CRITICAL
- **Confidence:** HIGH (Feasibility Checker verified against codebase)
- **Sources:** Feasibility Checker (Discrepancy 1, Section 5 Critical #1), Devil's Advocate (Logic Gap 4)
- **Plan sections:** Step 2, Architecture diagram, "EXTEND extraction.py"
- **Issue:** `extraction.py` uses `AutoModelForCausalLM`, forward hooks on self_attn/mlp sub-modules, and HDF5 datasets with causal-LM-specific layouts. A sentence-transformer encoder (E5, Jina) is encoder-only with fundamentally different architecture. The plan says "add contrastive model path" as if this is a minor extension. It is not — it requires either a parallel extraction function or a substantial refactor.
- **Impact:** Step 2's 8-12h estimate is unrealistic. The Feasibility Checker estimates 12-20h. The architecture diagram is misleading.
- **Recommendation:** Design a separate `coherence/extraction.py` for sentence-transformer embedding computation. Do not attempt to extend the existing `extraction.py`. Revise Step 2 estimate to 12-20h.

### C3: Ground-truth methodology for "known consistent" vs. "known inconsistent" is undefined

- **Severity:** CRITICAL
- **Confidence:** HIGH (2/3 critics flag directly; 1 flags indirectly)
- **Sources:** Scope Auditor (Missing #1, Missing #3), Devil's Advocate (Assumption 7, Failure Mode 3)
- **Plan sections:** Step 7 (Experiment 1), product selection criteria
- **Issue:** The plan lists criteria for group assignment (large brand, unified agency, etc.) but does not specify: (a) who makes these judgments, (b) how many raters, (c) whether inter-rater reliability is measured, (d) what observable evidence qualifies a product. The criteria are also confounded — large established brands with regulatory alignment also have more standardized language, which could produce tighter embeddings regardless of actual messaging coherence.
- **Impact:** Experiment 1 is the primary go/no-go gate. If group assignments are subjective and confounded, a passing result is uninterpretable.
- **Recommendation:** Before product selection: (a) write falsifiable operational definitions with specific observable evidence, (b) have at least 2 independent raters assign products, (c) measure inter-rater agreement (Cohen's kappa), (d) document the confound risk (brand maturity, category regulation) and plan to analyze it.

### C4: Metric exploration in Experiment 0 has too many degrees of freedom, undermining pre-registration

- **Severity:** CRITICAL
- **Confidence:** MEDIUM (Devil's Advocate flags directly; Scope Auditor flags the BCI scale as premature; Feasibility Checker does not address)
- **Sources:** Devil's Advocate (Assumption 6), partially Scope Auditor (BCI premature)
- **Plan sections:** Step 5 (Experiment 0 Part A)
- **Issue:** Experiment 0 explores {raw, mean-centered, whitened, contrastive} x {8 layers for Qwen} x {mean_pairwise, min_pairwise, centroid_distance, silhouette} x {cosine} = 128+ combinations. Selecting the highest-effect-size combination and "locking" it is still overfitting to the calibration distribution. The anti-p-hacking intent is undermined by the breadth of exploration.
- **Impact:** The locked metric may capitalize on chance properties of the calibration data and fail on real documents.
- **Recommendation:** Either (a) pre-register a much smaller candidate space (e.g., 3-5 combinations based on prior reasoning), or (b) use a held-out validation split of the calibration data (e.g., 60/40 split: explore on 60%, validate on 40%), or (c) apply a correction for multiple comparisons when selecting the winner.

### C5: Register taxonomy mismatch between calibration data and real-document channels

- **Severity:** CRITICAL
- **Confidence:** HIGH (2/3 critics flag directly)
- **Sources:** Feasibility Checker (Discrepancy 2, Section 5 Notable #2), Devil's Advocate (Assumption 1, implicit)
- **Plan sections:** Step 2, Experiment 0, channel taxonomy
- **Issue:** Calibration stimuli use registers `{marketing, regulatory, casual_social, patent, journalistic}`. Real-document channels are `{regulatory, marketing, retail, social, consumer_review}`. Three of the five real channels (retail, social as distinct from casual_social, consumer_review) have no corresponding register in the training data. The contrastive model is trained to suppress registers it has seen; it has no information about retail listing format, consumer review style, or brand social media voice.
- **Impact:** The contrastive model may fail to suppress register for channel types not represented in training data, producing unreliable coherence scores for exactly the channels that matter commercially.
- **Recommendation:** Explicitly document the mapping and mismatch. Consider generating additional calibration stimuli in the missing register styles (retail, consumer_review) before contrastive training — or at minimum, test the transfer explicitly in Experiment 0 Part B.

---

## Major Findings

### J1: Plan over-builds infrastructure; violates seed document's anti-productization directive

- **Severity:** MAJOR
- **Confidence:** HIGH (2/3 critics agree)
- **Sources:** Scope Auditor (Scope Creep items, Over-Engineering findings, Requirement Coverage violations), Feasibility Checker (Section 2.2, effort realism)
- **Plan sections:** Phase 0 (Steps 1, 6, 12), File Structure, Commercial Framing, Edge Cases
- **Issue:** The plan specifies a 15-file `coherence/` package, pipeline orchestrator with CLI subcommands, three report generators, four visualization types, BCI 0-100 scale, commercial framing with pricing, edge case taxonomy, and client data protocols. The seed document says "no platform architecture" and "no go-to-market strategy." The Scope Auditor identifies 8 scope-creep items. The plan allocates 53-77h but ~15-22h is infrastructure that serves commercialization, not validation.
- **Impact:** 15-22 hours of wasted effort building infrastructure before the core methodology is validated. Psychological commitment to the infrastructure may bias interpretation of experimental results.
- **Recommendation:** Strip to validation grade: `contrastive.py`, `metrics.py`, `baselines.py`, and one script per experiment. Drop `pipeline.py`, `reporting.py`, `viz.py`, CLI integration, BCI scale, commercial framing, edge cases, and client data protocol. Save 15-22h (Scope Auditor estimate) or redirect that effort toward addressing Critical findings.

### J2: Effort estimates are optimistic by 20-40%

- **Severity:** MAJOR
- **Confidence:** HIGH (2/3 critics agree on direction; estimates differ in magnitude)
- **Sources:** Feasibility Checker (Section 6: 64-99h vs. 53-77h), Scope Auditor (Section 4: 38-55h if scoped to validation only)
- **Plan sections:** Effort & Timeline table
- **Issue:** Step 2 (contrastive fine-tuning) is underestimated by 4-8h due to the extraction architecture mismatch. Document collection (2-5 days) is underestimated — 60-100 documents means 40-80h of manual work per the Devil's Advocate. Step 1 (ingestion) is underestimated by 2h. Total realistic effort is 64-99h at current scope, or 38-55h if scoped down to validation only.
- **Impact:** Timeline slippage from 3-5 weeks to 5-8 weeks at current scope. Risk of cutting corners on later experiments.
- **Recommendation:** If scope is reduced per J1, quote 38-55h implementation + 40-80h document collection. If scope is kept, quote 64-99h + collection. Either way, document collection should be on the critical path timeline.

### J3: Statistical power is overstated; value-added threshold is too weak

- **Severity:** MAJOR
- **Confidence:** MEDIUM (Devil's Advocate flags both; Scope Auditor does not dispute)
- **Sources:** Devil's Advocate (Assumption 8, Logic Gap 5)
- **Plan sections:** Step 7 (Experiment 1), pass criteria
- **Issue:** (a) Power ~0.80 is quoted for Cohen's d=1.0 (large effect). At d=0.6 (medium), power drops to ~0.46. The plan does not state the minimum detectable effect size. (b) The value-added test (AUC >= baseline + 0.05) is statistically meaningless at n=20 — the confidence interval around each AUC is likely wider than 0.05. A formal comparison (DeLong test) or a higher threshold (0.15) is needed.
- **Impact:** Experiment 1 may miss a real but moderate effect (false negative), or may declare value-added when the difference is noise (false positive on value claim).
- **Recommendation:** (a) Report the minimum detectable effect size at 80% power for n=10/group. (b) Replace the AUC >= 0.05 threshold with either a DeLong test for AUC comparison or raise the threshold to >= 0.15.

### J4: Training/evaluation data overlap in Experiment 0

- **Severity:** MAJOR
- **Confidence:** MEDIUM (Devil's Advocate flags directly)
- **Sources:** Devil's Advocate (Assumption 3)
- **Plan sections:** Step 2 (training), Step 5 (Experiment 0 Part A)
- **Issue:** The 800 calibration stimuli are used for both contrastive fine-tuning training AND metric exploration in Experiment 0 Part A. Part A cannot independently validate the contrastive model because it evaluates on the training set. The degradation tests in Part B manipulate the same LLM-generated text rather than introducing genuinely novel inputs.
- **Impact:** Experiment 0 may show strong results that do not transfer to real documents, creating false confidence before the expensive Experiment 1.
- **Recommendation:** At minimum, use a held-out split (e.g., 640 train / 160 test). Better: include a small set of genuinely novel inputs (the smoke test documents, or human-rewritten stimuli) in the Experiment 0 evaluation.

### J5: Chunking strategy for long documents is unexamined

- **Severity:** MAJOR
- **Confidence:** MEDIUM (Devil's Advocate flags directly; Feasibility Checker mentions long-context encoder as mitigation)
- **Sources:** Devil's Advocate (Assumption 4)
- **Plan sections:** M2 (document preprocessing), Step 1 (ingestion)
- **Issue:** Chunking documents into 2000-token windows with 500-token overlap and mean-pooling embeddings obscures which part of the document carries the coherence signal. A regulatory filing's "Methods" section may be highly coherent with marketing while the "Adverse Events" section is not. Mean-pooling collapses this distinction. The plan chose a long-context encoder (Jina v3, 8192 tokens) to reduce chunking, but some documents (regulatory filings) will still exceed 8192 tokens.
- **Impact:** Coherence scores for long documents may be noisy and hard to interpret, weakening Experiment 1 results for products with long regulatory filings.
- **Recommendation:** (a) Document the expected length distribution of target documents. (b) If most documents fit within 8192 tokens, chunking is a rare edge case — handle with truncation + warning rather than elaborate chunking infrastructure. (c) If chunking is needed, consider max-pooling or attention-weighted pooling rather than mean-pooling, and test the impact in Experiment 0.

### J6: Consumer review aggregation method is unprincipled

- **Severity:** MAJOR
- **Confidence:** MEDIUM (Devil's Advocate flags directly; Scope Auditor flags the sampling strategy as unresolved)
- **Sources:** Devil's Advocate (Assumption 10), Scope Auditor (Missing #2)
- **Plan sections:** Decision #7 (two-tier), Open Question 5
- **Issue:** Concatenating 5-10 consumer reviews into a single document creates an artificial text dominated by the most verbose review. Individual reviews vary wildly in content, quality, and relevance. The sampling strategy (how many, which ones, concatenation vs. individual analysis) is left as an open question but directly affects Experiment 1 data collection.
- **Impact:** Blocks document collection. Noisy market coherence scores if resolved poorly.
- **Recommendation:** Resolve before document collection. Consider embedding reviews individually and using the mean embedding (not concatenated text), which avoids verbose-review dominance. Set a fixed count (e.g., 10 reviews per product) with explicit selection criteria (most recent, minimum length, verified purchase).

---

## Minor Findings

1. **Register probe threshold too lenient (Assumption challenge).** A probe accuracy of 0.7 means register is still classifiable 70% of the time. Near-chance (0.2 for 5 registers) would be more convincing. The plan should justify why 0.7 is sufficient.

2. **BCI 0-100 scale will compress real scores to the low end (Assumption challenge).** Calibration stimuli have all attributes in every register — they represent maximum coherence. Real documents will always score lower, making the scale counterintuitive. Defer the scale to post-validation.

3. **Experiment 3 synonym test does not address keyword domination (Assumption challenge).** Testing synonyms tests vocabulary knowledge, not whether the metric measures coherence vs. lexical overlap. A proper control requires probing the correct attribute for the wrong product.

4. **Expert validation may conflate brand familiarity with coherence (Assumption challenge).** Well-known brands may be rated "more coherent" by default due to familiarity and perceived quality.

5. **BoW baseline is a classifier, not a similarity metric (Feasibility Checker).** The existing `run_bow_baseline()` cannot be reused for TF-IDF coherence baseline. `coherence/baselines.py` is genuinely new code; the plan's claim of "reuse" is misleading.

6. **Missing `rsa_product_partial.npy` artifact (Feasibility Checker).** The partial RSA code works but the output file has not been generated. Needs a pipeline re-run before Experiment 0.

7. **Experiment 0 failure protocol is missing (Scope Auditor).** The plan has failure protocols for Experiment 1 but not for Experiment 0. If both contrastive fine-tuning and partial RSA fail to produce adequate register suppression, the plan lacks a concrete next step.

---

## Contradictions Between Critics

### Scope Auditor vs. Feasibility Checker on Effort

The **Scope Auditor** argues the plan should be reduced to 38-55h by stripping productization infrastructure, while the **Feasibility Checker** argues the plan as-scoped will actually take 64-99h (20-30% more than estimated). These are not true contradictions — the Scope Auditor recommends cutting scope, the Feasibility Checker assesses the current scope — but they diverge on the correct response. The Scope Auditor says "build less," the Feasibility Checker says "budget more time for what you're building."

**Resolution:** Both are correct within their frame. The plan should be descoped per the Scope Auditor (saving 15-22h), and the remaining work should be budgeted per the Feasibility Checker's more conservative estimates. This yields approximately 45-65h of implementation, a middle ground that is both scope-appropriate and realistically estimated.

No other material contradictions were found. The three critics are remarkably convergent on the core issues, differing mainly in emphasis and framing.

---

## Overall Assessment

**Readiness:** NOT READY for execution. 5 Critical findings must be addressed first.

**Highest risk:** The contrastive fine-tuning strategy (C1, C2, C5). All three critics identify this as the plan's load-bearing element and its greatest vulnerability. The model has never seen real-world register variation, the training data is small, the evaluation occurs on the training distribution, and the extraction infrastructure does not support the chosen model type. A real-document smoke test before building infrastructure would derisk this in 2-4 hours.

**Strongest area:** The experimental design. Sequential gating, pre-registration discipline, explicit kill criteria, and fallback paths are recognized by all three critics as genuine strengths. The six experiments address the right questions in the right order. The plan's scientific rigor is not in question — its engineering scope and a few untested technical assumptions are.

**Recommended action sequence:**
1. Run a 2-4 hour real-document smoke test (C1)
2. Resolve ground-truth methodology (C3)
3. Design separate sentence-transformer extraction path (C2)
4. Address register taxonomy mismatch (C5)
5. Reduce metric exploration space or add held-out split (C4)
6. Strip commercial/productization elements (J1)
7. Revise effort estimates (J2)
8. Then proceed with Phase 0

---

COMPLETED
