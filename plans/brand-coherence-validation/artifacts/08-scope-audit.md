# Scope Audit: Brand Message Coherence Validation Plan

## 1. Scope Alignment

### In Scope (correctly included)

| Plan Element | Traces to |
|---|---|
| Experiment 0: metric exploration on calibration data | Decision #4 (add Exp 0), Decision #6 (pilot on calibration data) |
| Contrastive fine-tuning as primary register correction | Decision #3 |
| Experiment 1 with n=10 per group (20 total) | Decision #8, seed Experiment 1 |
| Baselines (TF-IDF, BERTScore) in Experiment 1 only | Decision #5 |
| Two-tier reporting (brand coherence + market coherence) | Decision #7 (consumer reviews as separate tier) |
| Sequential gating across Experiments 0-5 | Seed document: "We don't move to the next until the current one passes" |
| Experiments 2-5 matching seed document scope | Seed Experiments 2-5 |
| Pre-registered metric lock before real-document testing | Decision #4 rationale ("avoid p-hacking") |

### Scope Creep

| Item | Severity | Justification |
|---|---|---|
| **Voice Consistency Score (Conflict 4 resolution)** | Low | Deferred to Phase 2, but the detailed design (base model register signatures, 2x2 diagnostic) is premature. The seed document explicitly does not include product features beyond validation. Remove the design detail; a one-line note is sufficient. |
| **Commercial framing section (lines 547-554)** | Medium | The seed document says "We don't think about ... pricing models, or go-to-market strategy." The plan includes naming recommendations ("Semantic Content Alignment"), price-point justification, and deliverable framing. This belongs in a separate commercial artifact, not the validation plan. |
| **Client data handling protocol (M8)** | Medium | Secure deletion, per-engagement directories, data retention terms -- all premature. The seed says "Those are premature until the methodology is validated." Move to a post-validation checklist. |
| **Edge cases section (lines 557-564)** | Medium | Co-branded products, international variations, seasonal editions, negative press scenarios -- these are engagement scoping questions, not validation plan items. Remove or move to a future commercial spec. |
| **Data compounding strategy (Open Question 9)** | Low | Calibration improvement, supervised fine-tuning from engagement data, anonymized benchmarks -- all post-validation concerns. |
| **BCI 0-100 scale (line 540)** | Low | Designing a calibrated index scale is premature. The validation needs raw scores and effect sizes, not a polished scoring system. |
| **Step 6: Full reporting infrastructure** (4-6h) | Medium | Three report generators, four visualization types, JSON+Markdown export, BCI index -- this is product-grade reporting before the core metric is validated. For validation, a simple script printing scores to stdout suffices. |
| **Model migration plan (Open Question 6)** | Low | 12-18 month recalibration cycles are a post-commercialization concern. |

### Missing / Under-specified

| Gap | Impact |
|---|---|
| **Product selection criteria are vague.** "Known consistent" and "known inconsistent" need concrete, falsifiable operational definitions with examples BEFORE selection begins. The plan acknowledges this but does not provide the definitions. | High -- confirmation bias risk in Experiment 1. |
| **Consumer review sampling strategy is unresolved.** Decision #7 says consumer reviews are a separate tier, but the plan leaves the number of reviews per product as an open question. This directly affects Experiment 1 data collection. | Medium -- blocks document collection. |
| **How will "professional judgment" ground truth be established?** Experiments 1 and 2 rely on knowing which products have consistent/inconsistent messaging and which channels are outliers. The plan does not specify who makes these judgments, how many raters, or whether inter-rater reliability is measured. | High -- threatens validity of the primary gate. |
| **No explicit budget for Experiment 0 failure.** The plan has a failure protocol for Experiment 1 but not for Experiment 0. If contrastive fine-tuning AND partial RSA both fail, what happens? The decision framework mentions it but the implementation plan lacks a concrete fallback step count. | Medium |

---

## 2. Over-Engineering Findings

| Finding | YAGNI? | Simpler Alternative |
|---|---|---|
| **15-file `coherence/` package with 6 experiment modules, a pipeline orchestrator, and CLI subcommands** | Yes | For a 5-experiment validation, a single Jupyter notebook per experiment (or at most 3-4 scripts) would be faster to iterate on. The module structure assumes the code will be reused commercially, which the seed document says not to assume. |
| **`coherence/pipeline.py` with sequential gating, `--force` flags, and `verdict.json` schema** | Yes | A README checklist ("run exp0 first, check results, then run exp1") achieves the same gating with zero infrastructure code. The 2-3 hours estimated for Step 12 produce no validation value. |
| **Three separate report generators + four visualization types** | Yes | Experiment analysis code can produce its own plots inline. A dedicated reporting package is product code, not validation code. |
| **`coherence/ingest.py` with multi-product detection, channel-aware boilerplate removal, chunk overlap** | Partially | Some preprocessing is needed, but the spec reads like a production ETL pipeline. For 20 products with 3-5 documents each (60-100 documents), manual spot-checking with a simple text-cleaning function is sufficient. |
| **Training BOTH sentence-transformer AND testing base Qwen with partial RSA in Experiment 0** | No | This is justified. The user decided on contrastive fine-tuning (Decision #3) with RSA as fallback. Testing both before committing is sound. |
| **Adversarial gradient-reversal term as escalation option** | Borderline | Mentioned as an option, not required. Acceptable as long as it stays an option. |

**Core concern:** The plan allocates 53-77 hours of implementation, but the seed document estimated the entire 5-experiment sequence at roughly 12-19 days of effort (mostly document collection, not coding). The implementation plan is 3-4x heavier than the seed expected, driven by infrastructure that serves commercialization rather than validation.

---

## 3. Requirement Coverage

| Seed Requirement | Plan Coverage | Notes |
|---|---|---|
| Experiment 1: real-document sensitivity, 10 products (now 20) | Step 7, fully covered | Pass criteria upgraded from seed's informal criterion to AUC >= 0.85, which is reasonable. |
| Experiment 2: channel attribution on inconsistent products | Step 8, fully covered | Pass criterion relaxed from 3/5 to 6/10 (proportionally same). |
| Experiment 3: attribute-level drill-down | Step 9, fully covered | Probe design expanded with two context levels. Good. |
| Experiment 4: temporal coherence drift | Step 10, fully covered | Added control product for secular trend correction. Good addition. |
| Experiment 5: competitive benchmarking with expert validation | Step 11, fully covered | Strengthened: experts predict BEFORE seeing results. Good. |
| Decision #3: contrastive fine-tuning for register correction | Step 2, fully covered | |
| Decision #4: Experiment 0 for metric exploration | Step 5, fully covered | |
| Decision #5: baselines in Experiment 1 only | Step 4 + Step 7, covered | Baselines built in Phase 0 but only compared in Experiment 1. Correct. |
| Decision #6: pilot on calibration data | Step 5 Part B, covered | |
| Decision #7: consumer reviews as separate tier | Two-tier architecture throughout, covered | |
| Decision #8: n=10 per group | Step 7, covered | |
| Seed: no platform architecture | **VIOLATED** | The `coherence/` package with pipeline orchestration, CLI subcommands, and reporting infrastructure is de facto platform architecture. |
| Seed: no go-to-market strategy | **VIOLATED** | Commercial framing, naming, pricing commentary present. |

---

## 4. Effort Assessment

| Phase | Plan Estimate | Audit Assessment | Delta |
|---|---|---|---|
| Phase 0: Infrastructure | 17-25h | 8-12h if scoped to validation only (drop reporting module, simplify ingestion, skip pipeline orchestration) | -9 to -13h |
| Phase 1: Experiment 0 | 6-8h | 6-8h (appropriately scoped) | 0 |
| Phase 2: Reporting + Exp 1 | 12-18h | 8-12h without reporting infrastructure (analysis code produces results directly) | -4 to -6h |
| Phase 3: Exps 2-3 | 8-11h | 8-11h (appropriately scoped) | 0 |
| Phase 4: Exps 4-5 | 8-12h | 8-12h (appropriately scoped) | 0 |
| Phase 5: Integration | 2-3h | 0h (eliminate; not needed for validation) | -2 to -3h |
| **Total** | **53-77h** | **38-55h** | **-15 to -22h** |

The experiments themselves (Phases 1, 3, 4) are well-scoped. The bloat is concentrated in infrastructure (Phase 0 over-building) and premature productization (Phase 2 reporting, Phase 5 integration).

---

## 5. Verdict

**PASS WITH CONDITIONS.** The plan faithfully covers all six user decisions and all five seed experiments. The experimental design is sound -- sequential gating, pre-registered metrics, contrastive fine-tuning with fallback, strengthened expert validation. The specialist conflict resolutions are reasonable.

**Required changes before execution:**

1. **Strip commercial/product elements.** Remove the commercial framing section, edge case taxonomy, BCI 0-100 scale, client data protocol, and model migration plan. These belong in a separate post-validation artifact.

2. **Simplify infrastructure to validation grade.** Replace the 15-file `coherence/` package with a lighter structure: `contrastive.py` (training), `metrics.py` (scoring), `baselines.py` (comparators), and one script per experiment. Drop `pipeline.py`, `reporting.py`, `coherence/viz.py`, and the CLI subcommand integration. Save ~15-22 hours.

3. **Specify ground-truth methodology.** Before product selection for Experiment 1, define: (a) who judges "known consistent" vs. "known inconsistent" (the researcher alone, or independent raters?), (b) what observable evidence qualifies a product for each group, (c) whether inter-rater reliability is measured. This is the single biggest validity threat.

4. **Resolve consumer review sampling before document collection.** Pick a number (e.g., 5 reviews per product, concatenated) and commit. This is a blocking dependency for Experiment 1.

**Advisory (not blocking):**

- The Experiment 0 simulation tests (Part B: length variation, attribute removal, human rewrites) are valuable but the "< 0.2 SD shift" robustness criterion is arbitrary. Consider treating Part B results as informational rather than as a hard gate.
- Open Question 10 (buyer validation conversations) is excellent advice from the domain expert. Consider scheduling these before investing in Experiment 1 document collection.

---

COMPLETED
