# Brand Message Coherence: Validation Plan

## Summary

This plan validates a quantitative methodology for measuring how consistently a product's core semantic message survives translation across communication channels (regulatory, marketing, retail, social, consumer review). The methodology uses mean-centered hidden-state representations from Qwen2.5-32B at middle layers (~20-40) to produce a per-product "coherence score" across channels. Contrastive fine-tuning of a sentence encoder is available as an escalation if mean centering alone proves insufficient.

The plan builds on a completed Protocol Layer Hypothesis experiment that produced an 800-stimulus calibration dataset and a working extraction/RSA pipeline. The central technical challenge was that register (communication style) dominates product identity in the model's representational geometry (RSA r=0.670 vs. r=0.371). A smoke test on 6 real products (18 documents) demonstrated that **mean centering resolves this**: at layer 30 with mean centering, within-category coherence differentiation is clear (skincare gap +0.215, supplements gap +0.211), without any contrastive training.

User decisions incorporated:

1. Register correction via contrastive fine-tuning as escalation, mean centering as primary (Decision 3, updated by smoke test findings)
2. Experiment 0 for metric exploration on calibration data (Decision 4)
3. Baselines (TF-IDF, BERTScore) in Experiment 1 only (Decision 5)
4. Pilot on existing calibration stimuli before real-document collection (Decision 6)
5. Two-tier reporting: brand coherence (controlled channels) + market coherence (including consumer reviews) (Decision 7)
6. n=10 per group (20 products total) for Experiment 1 (Decision 8)
7. Ground-truth: hybrid rater protocol -- the researcher assigns, one additional rater independently rates, compute agreement (Decision 9)
8. Register probe threshold: relative drop >= 50%, i.e., probe accuracy <= 0.5 after contrastive fine-tuning (Decision 10)
9. Wrong-product control added to Experiment 3 (Decision 11)
10. Full scope accepted: all 6 experiments, 40-63h implementation + 40-80h document collection, 4-6 weeks (Decision 12, revised downward by smoke test findings)

**Total estimated effort:** 40-63 hours of implementation + 40-80 hours of document collection, over 4-6 weeks calendar time.

### Smoke Test Results (Step 0 — completed)

A smoke test on 6 real CPG products (Aquaphor, CeraVe, Drunk Elephant in skincare; Nature Made, OLLY, Bloom in supplements) across 3 channels each produced the following findings:

1. **Mean centering is transformative.** Raw cosine similarities at layer 61 compress into a 0.89-0.97 band with a 0.039 within/between gap. Mean centering at layer 30 expands this to a -0.43 to +0.50 range with consistent within-category differentiation.
2. **Coherence differentiation signal peaks at middle layers (~30), not at the product identity peak (~61).** This is because coherence is closer to "do these documents talk about the same things?" (broad semantics, middle layers) than "which specific product is this?" (fine-grained identity, late layers).
3. **Whitening is broken at these sample sizes.** With N << D (6-800 samples, 5120 dimensions), whitening produces degenerate results. Removed from the candidate space.
4. **Contrastive fine-tuning is not load-bearing.** Mean centering alone separates consistent from inconsistent brands within the same category. Contrastive training is retained as an escalation path if Experiment 1 shows insufficient discrimination.
5. **Layer selection is model-specific.** The optimal layer (30 for Qwen2.5-32B) will differ for other models. Layer sweep must be re-run per model. The relative position (middle third) may transfer.
6. **Vocabulary narrowness may inflate scores.** Aquaphor (simple product, narrow vocabulary) scored 0.499 while all others scored below 0.10. The plan should control for this confound.

---

## Problem Statement

### What we have

A working research pipeline (Python: `stimuli.py`, `extraction.py`, `analysis.py`, `viz.py`, `run.py`) that:

- Generated 800 calibration stimuli (80 products x 5 registers x 2 paraphrase variants)
- Extracted hidden states from Qwen2.5-32B across all 64 layers
- Demonstrated via RSA that product identity signal exists at late layers (r=0.371 at layer 61), though register dominates (r=0.670 at every layer)
- Showed perfect register classification (F1=1.0) and near-perfect category classification (F1=0.99) via linear probes

### What we need to prove

1. That the product-identity signal can be isolated from the register signal (mean centering demonstrated in smoke test; contrastive fine-tuning as escalation)
2. That the resulting "coherence metric" discriminates between products with known-consistent vs. known-inconsistent messaging on real-world documents
3. That the metric provides channel-level attribution (which channel diverges)
4. That the metric can drill down to attribute-level gaps (what content is lost)
5. That the metric captures temporal drift and produces competitive rankings professionals recognize

### What would kill the project

- ~~If a real-document smoke test shows zero directional signal with the existing pipeline (Step 0 gate)~~ **PASSED** — directional signal confirmed in both categories with mean centering.
- If mean centering AND contrastive fine-tuning both fail to provide coherence differentiation on calibration data (Experiment 0 gate)
- If the metric cannot distinguish known-consistent from known-inconsistent products on real documents (Experiment 1 gate)
- If simple baselines (TF-IDF, BERTScore) achieve comparable discrimination (value-added gate in Experiment 1)

---

## Proposed Approach

### Core methodology

1. **Real-document smoke test** — **COMPLETED.** Confirmed directional coherence signal in 6 real products across both skincare and supplements categories using mean-centered Qwen2.5-32B representations at middle layers.

2. **Use mean-centered hidden states from Qwen2.5-32B as the primary approach.** The smoke test demonstrated that simple mean centering at layer ~30 produces clear within-category coherence differentiation (combined gap +0.426) without any model training. This uses the existing extraction pipeline with zero new infrastructure.

3. **Contrastive fine-tuning is the escalation path** if mean centering proves insufficient on the full 20-product Experiment 1. A sentence-transformer encoder fine-tuned on the 960-stimulus calibration set can be trained in ~30 min and provides a second approach to test.

4. **Pre-register the metric formula** in Experiment 0 (exploratory phase on calibration data), then lock it before Experiment 1. The metric produces two scores per product:
   - **Brand Coherence Score**: mean cosine similarity across brand-controlled channel pairs (regulatory, marketing, retail, social)
   - **Market Coherence Score**: mean cosine similarity across all channel pairs including consumer reviews

5. **Validate sequentially** through six experiments with explicit pass/fail gates.

6. **Layer selection is model-specific.** The optimal layer (30 for Qwen2.5-32B's 64 layers) is in the middle third of the network. If the model changes, the layer sweep in Experiment 0 must be re-run. The absolute layer number does not transfer across architectures; the relative position (middle third) may.

### What "coherence" means and does not mean

The methodology measures **semantic content alignment** -- whether the same product story (claims, attributes, positioning) is told across channels, independent of how it is told. It explicitly does NOT measure:

- Voice/tone consistency (the contrastive fine-tuning deliberately collapses register)
- Visual/experiential consistency (outside scope of text analysis)
- Consumer perception alignment (measured separately in the "market coherence" tier)

---

## Architecture

### System overview

```
+-----------+    +----------------------------+    +-----------------+
| Documents |---->|  Existing Extraction       |---->|   Coherence     |
| (ingest + |    |  (extraction.py)           |    |   Analysis      |
|  existing |    |  Qwen2.5-32B hidden states |    |  (metrics.py    |
|  stimuli) |    |  + mean centering          |    |   + baselines)  |
+-----------+    +----------------------------+    +-----------------+
                                                           |
                 +----------------------------+            |
                 | Contrastive Fine-Tuner     | (ESCALATION ONLY)
                 | coherence/contrastive.py   |            |
                 | coherence/extraction.py    |---->--------+
                 +----------------------------+
```

### Component inventory

| Component | Status | Module |
|-----------|--------|--------|
| Document ingestion and cleaning | **NEW** | `coherence/ingest.py` |
| Coherence metrics | **NEW** | `coherence/metrics.py` |
| Baseline comparators | **NEW** | `coherence/baselines.py` |
| Per-experiment scripts | **NEW** | `coherence/experiment_0.py` through `coherence/experiment_5.py` |
| Existing hidden-state extraction | **REUSED** | `extraction.py` (primary path — Qwen2.5-32B) |
| Existing analysis (mean centering) | **REUSED** | `analysis.py` (`correct_anisotropy()` with `mean_centering`) |
| Sentence-transformer extraction | **NEW, ESCALATION** | `coherence/extraction.py` (only if contrastive path activated) |
| Contrastive fine-tuning | **NEW, ESCALATION** | `coherence/contrastive.py`, `coherence/losses.py` (only if mean centering insufficient) |
| Existing pipeline | **UNCHANGED** | `stimuli.py`, `probes.py`, `run.py`, existing tests |

**Key architectural change (post-smoke-test):** The primary path now reuses the existing `extraction.py` (Qwen2.5-32B) and `analysis.py` (mean centering) with zero new infrastructure. The contrastive sentence-transformer path is retained as an escalation: if Experiment 0 or 1 shows that mean centering at the optimal layer doesn't provide sufficient discrimination, the contrastive modules are built then. This reduces the critical-path implementation from 20-28h to 8-14h.

**Orchestration:** No pipeline orchestrator or CLI infrastructure. Each experiment runs as a standalone script (`python -m coherence.experiment_0`, etc.). Sequential gating is enforced by checking for `verdict.json` files at the start of each experiment script.

### Data flow

**Step 0 (Smoke Test):**
```
2 real products x 3 docs each --> existing extraction.py --> existing analysis.py (partial RSA)
                                                                       |
                                                                       v
                                                      Directional signal check (manual)
```

**Experiment 0 (Calibration Pilot):**
```
data/stimuli.json (960 stimuli: 800 original + 160 new registers)
    |
    +--> coherence/contrastive.py --> data/coherence/models/encoder/
    |                                        |
    |                                        v
    |                               coherence/extraction.py (contrastive path)
    |                                        |
    |                                        v
    |                               data/coherence/exp0/contrastive_embeddings.npy
    |
    +--> data/Qwen_*_hidden_states.h5 (already exists)
                |
                v
         coherence/metrics.py (6 metric combinations, Bonferroni correction)
                |
                v
         data/coherence/exp0/metric_selection.json (LOCKED)
```

**Experiments 1-5 (Real Documents):**
```
data/coherence/portfolios/exp{N}/ --> ingest.py --> coherence/extraction.py --> coherence/metrics.py
                                                                                        |
                                                                                        v
                                                               data/coherence/exp{N}/results/
                                                               data/coherence/exp{N}/verdict.json
```

---

## Implementation Plan

### Step 0: Real-Document Smoke Test — COMPLETED

**Result: PASS.** Directional coherence signal confirmed on 6 real CPG products (18 documents).

**What was tested:**
- 6 products: Aquaphor, CeraVe (skincare, expected high coherence), Drunk Elephant (skincare, expected low), Nature Made (supplements, expected high), OLLY, Bloom (supplements, expected low)
- 3 channels each: regulatory/packaging, marketing, consumer review
- Extracted via existing Qwen2.5-32B pipeline, analyzed across 10 layers x 2 corrections

**Key findings:**
- Raw layer 61: within-product mean 0.9534, between-product mean 0.9149 (gap 0.039 — compressed, noisy)
- **Mean-centered layer 30: within-category coherence differentiation clear** (skincare gap +0.215, supplements gap +0.211, combined +0.426)
- High-coherence products score higher than low-coherence products in both categories
- Whitening produces degenerate results (N << D) — removed from candidates
- Coherence differentiation peaks at middle layers (~30), not at the product identity peak (~61)
- Aquaphor (0.499) is an outlier — may reflect vocabulary narrowness, not just coherence. Control needed.

**Implication:** Mean centering is sufficient for the primary path. Contrastive fine-tuning moves to escalation. Infrastructure effort reduced by ~12h.

**Artifacts:** `data/coherence/smoke-test/stimuli_extended.json`, `smoke_test_v2_results.json`, `Qwen_*_hidden_states.h5`

---

### Phase 0: Infrastructure (Steps 1-4, ~12-18 hours)

Build the foundational modules. Steps 1, 3, and 4 can proceed in parallel. Step 2 (contrastive fine-tuning) is deferred to escalation — only built if mean centering proves insufficient in Experiment 0 or 1.

**Step 1: Document Ingestion (`coherence/ingest.py`)** -- 4-6h

- `RealDocument` dataclass with product_id, channel, text, source_url, date_collected, is_brand_controlled
- `load_documents()`, `clean_document()` (channel-aware boilerplate removal), `to_stimuli_format()`
- Document length handling: simple truncation at 8192 tokens with a warning logged for truncated documents. No overlapping-window chunking infrastructure. If more than 20% of documents are truncated, revisit this decision.
- `validate_product_set()` to verify minimum channel coverage per product
- `aggregate_reviews()`: for consumer reviews, collect 10 per product (most recent verified-purchase reviews with >= 50 words), embed each individually, and return the mean embedding. Do NOT concatenate reviews into a single document.
- Channel taxonomy: `regulatory`, `marketing`, `retail`, `social`, `consumer_review`
- Cleaning: strip HTML, copyright notices, Amazon template chrome, hashtags/mentions, normalize Unicode
- Multi-product detection: flag documents mentioning multiple product names
- Tests: `test_ingest.py` covering cleaning, schema conversion, truncation, review aggregation, validation

**Step 2: DEFERRED — Contrastive Fine-Tuning (escalation only, 12-18h if activated)**

The smoke test demonstrated that mean centering on the existing Qwen2.5-32B pipeline provides sufficient coherence differentiation without contrastive training. This step is deferred and only activated if:
- Experiment 0 shows mean centering has effect size d < 0.5 on calibration data, OR
- Experiment 1 shows AUC < 0.85 with mean centering but the signal is directionally present

If activated, the step includes:
- **(a) Register alignment (2-3h):** Generate 160 additional calibration stimuli for `retail` and `consumer_review` registers, remap `casual_social` to `social`
- **(b) Contrastive fine-tuning (6-10h):** Sentence-transformer base (Jina v3 or E5-large) with `MultipleNegativesRankingLoss`. Pass criteria: register probe accuracy drops by >= 50% relative (to <= 0.5), product probe accuracy > 0.8
- **(c) Sentence-transformer extraction module (4-5h):** New `coherence/extraction.py` using `model.encode()`. Separate from existing `extraction.py`.

**This step is NOT on the critical path.** The primary path uses existing infrastructure only.

**Step 3: Coherence Metrics (`coherence/metrics.py`)** -- 3-4h

- `compute_pairwise_coherence()`: cosine similarity matrix between all channel embeddings
- `compute_coherence_score()`: aggregates to scalar brand_coherence and market_coherence
  - Candidate aggregation methods (explored in Experiment 0, restricted to 3): mean pairwise, centroid distance, silhouette coefficient
- `identify_outlier_channel()`: channel with lowest mean similarity to all others
- `compute_attribute_coherence()`: attribute-probe x channel similarity matrix for Experiment 3
- Two-tier architecture: `CONTROLLED_CHANNELS = {"regulatory", "marketing", "retail", "social"}`
- Tests: perfect coherence (identical embeddings -> 1.0), zero coherence (orthogonal -> ~0.0), two-tier divergence, outlier detection

**Step 4: Baselines (`coherence/baselines.py`)** -- 2-3h

- `compute_tfidf_coherence()`: TF-IDF cosine similarity between document pairs. This is entirely new code -- it shares no code with the existing BoW classifier in `stimuli.py`, which is a classification pipeline, not a similarity metric.
- `compute_bertscore_coherence()`: sentence-BERT mean embedding cosine similarity
- `compare_methods()`: Spearman correlation between all method pairs, value-added assessment
- Same output structure as `compute_coherence_score()` for direct comparison
- Tests: identical texts -> 1.0, unrelated texts -> low, method comparison structure

**Completion criteria for Phase 0:**
- All three primary modules (ingest, metrics, baselines) pass their unit tests
- All modules produce outputs in compatible formats
- Contrastive modules (Step 2) only built if escalation triggered — not a Phase 0 gate

---

### Phase 1: Experiment 0 -- Metric Exploration and Lock (Step 5, ~6-8 hours)

**Step 5: Experiment 0 (`coherence/experiment_0.py`)**

This is the exploratory phase on calibration data. No new documents collected.

**Data split:**
- Training set: 960 stimuli across 5 real-channel registers (used for contrastive fine-tuning)
- Held-out validation set: 160 stimuli from the `patent` and `journalistic` registers (not in the training set). This tests actual transfer -- the metric must work on register styles the contrastive model has never seen.

**Part A: Metric space exploration (on 160 held-out stimuli)**

- Primary candidate space: mean-centered base Qwen at layers {20, 25, 30, 35, 40, 50, 55, 60, 61} x {mean_pairwise, centroid_distance, silhouette} = 27 combinations (screen), then top-3 layers x 3 aggregations = 9 for formal testing with Bonferroni correction (p-threshold = 0.05/9 = 0.0056)
- For each combination: compute per-product coherence across registers, compute effect size (within-product coherence vs. random-pairing coherence)
- Record the full comparison in `metric_selection.json` alongside the winner
- **Vocabulary narrowness control:** compute a lexical diversity metric (type-token ratio) per product across channels. If coherence score correlates strongly with inverse lexical diversity (Spearman rho > 0.5), the metric may be measuring vocabulary narrowness, not coherence. Flag and investigate.
- If the best mean-centering combination achieves d < 0.5: activate Step 2 (contrastive fine-tuning) and add contrastive encoder to the candidate space
- **Whitening is excluded** — degenerate at N << D (confirmed in smoke test with both 6 and 800 samples)

**Part B: Real-document condition simulation**

- Length variation: truncate stimuli to 20-30 words (tweets), expand to 300-500 words (filings)
- Attribute removal: remove 1-3 core attributes from stimuli (simulate incomplete channels)
- Non-LLM text: manually rewrite ~10 stimuli in human style (sufficient for directional signal)
- Measure metric robustness under each degradation
- Robustness criterion: metric change < 0.2 standard deviations for moderate degradations

**Part C: Metric lock**

- Output: `data/coherence/exp0/metric_selection.json` -- specifies:
  - Model: base Qwen layer number (primary) OR contrastive encoder path (escalation)
  - Correction: mean_centering (whitening excluded)
  - Aggregation: mean_pairwise/centroid_distance/silhouette (whichever won)
  - Distance: cosine
  - Vocabulary narrowness control: pass/flag status
- This file must exist before Experiment 1 can run (hard gate)

**Failure protocol for Experiment 0:**

- If best mean-centering combination achieves d < 0.3 (trivial effect): Activate Step 2 (contrastive fine-tuning). If contrastive also achieves d < 0.3, **kill the project.**
- If 0.3 <= d < 0.5 with mean centering (small effect): Activate Step 2 and compare. Use whichever approach achieves higher d.
- If d >= 0.5 with mean centering: Lock the mean-centering metric. Contrastive fine-tuning is not needed.
- If d >= 0.5 but degradation tests fail: The metric works on clean calibration data but is fragile. Investigate which degradation is problematic and address in preprocessing before proceeding to Experiment 1.

**Completion criteria:**
- Selected metric shows within-product coherence > between-product coherence (effect size d > 0.5) on the 160 held-out stimuli
- Length-variation degradation is tolerable (< 0.2 SD shift)
- Metric formula is locked in `metric_selection.json`

---

### Phase 2: Experiment 1 -- Real-Document Sensitivity (Step 6, ~8-12 hours)

**Step 6: Experiment 1 (`coherence/experiment_1.py`)**

The core validation experiment. **This is the primary go/no-go gate.**

*Ground-truth: hybrid rater protocol (Decision 9):*

- Write operational definitions with specific observable evidence BEFORE selecting products:
  - "Known consistent": large brand, strict regulatory alignment, unified agency, documented brand guidelines publicly available, single-brand product page
  - "Known inconsistent": startup/challenger brand, multiple agencies, known rebranding, social media voice disconnected from packaging, or demonstrably fragmented messaging
- The researcher makes initial assignments with written justification per product
- One additional rater independently rates the same 20 products using the same operational definitions
- Compute Cohen's kappa for inter-rater agreement. If kappa < 0.4, revisit criteria before proceeding.
- Effort: 3-5h for the hybrid rater protocol

*Product selection:*

- 10 per group, 20 total (Decision 8)
- Bias toward products where ground truth confidence is high
- Document confound risks: brand maturity, category regulation level, documentation completeness

*Document collection:*

- For each product: 3-5 real documents from distinct channels
- Minimum channels per product: 3 (regulatory + marketing + at least one of retail/social/review)
- Channel taxonomy with tier flags:
  - Brand-controlled: regulatory, marketing, retail (if brand-authored), social (brand's own posts)
  - Third-party: consumer_review
- Consumer reviews: 10 per product, most recent verified-purchase reviews with >= 50 words, embedded individually, mean-pooled into a single channel embedding
- Each document: raw text + metadata (source_url, date_collected, channel, word_count)
- Preprocessing via `coherence/ingest.py`: boilerplate removal, encoding normalization, truncation at 8192 tokens

*Analysis:*

1. Extract embeddings using locked metric (contrastive model or base Qwen + correction)
2. Compute brand_coherence and market_coherence for all 20 products
3. Primary test: Mann-Whitney U between consistent and inconsistent groups (p < 0.05)
4. Effect size: Cohen's d (target: >= 0.8) and rank-biserial correlation
5. Classification: ROC AUC (target: >= 0.85), misclassification count (target: <= 2 of 20)
6. Baselines (Decision 5): compute TF-IDF and BERTScore coherence on the same documents
7. Value-added test: DeLong test comparing correlated AUCs (p < 0.10, one-tailed). If DeLong test is not significant, report AUC difference descriptively but do not claim value-added.
8. Secondary analysis: rank-order correlation across all 20 products (Spearman rho)
9. Face-validity check: present full ranking to 2-3 industry professionals -- no "embarrassingly wrong" outliers

*Statistical transparency:*

At n=10 per group, Mann-Whitney U achieves 80% power for Cohen's d >= 1.0 (large effect). The minimum detectable effect at 80% power is approximately d=1.05. This experiment is NOT powered to detect medium effects (d=0.5-0.8). A non-significant result does not prove the metric is useless, only that the effect may be smaller than detectable at this sample size.

*Pass criteria:*

- AUC >= 0.85 on consistent vs. inconsistent classification
- Mann-Whitney U p < 0.05
- At most 2 misclassifications out of 20
- DeLong test p < 0.10 vs. at least one baseline (or descriptive AUC advantage if DeLong is non-significant)
- No product that a domain expert would consider "obviously wrong"

*Failure protocol:*

- If AUC < 0.70: fundamental methodology failure. Debug the register correction. Consider alternative approaches (supervised coherence predictor, register-appropriateness metric).
- If 0.70 <= AUC < 0.85: marginal. Investigate confounds (brand size, category effects). Consider increasing n or refining document preprocessing.
- If AUC >= 0.85 but baselines match (DeLong p >= 0.10): the hidden-state method adds no value over simpler approaches. Adopt the best baseline.

**Completion criteria:**
- All 20 products scored with brand_coherence and market_coherence
- Statistical tests completed with documented results
- Baseline comparison completed (including DeLong test)
- Pass/fail verdict recorded in `data/coherence/exp1/verdict.json`

---

### Phase 3: Experiments 2-3 -- Diagnostic Validation (Steps 7-8, ~8-11 hours)

Phase 3 proceeds ONLY if Experiment 1 passes.

**Step 7: Experiment 2 -- Channel Attribution (`coherence/experiment_2.py`)** -- 2-3h

Reanalysis of Experiment 1 data. No new document collection.

- For each of the 10 "known inconsistent" products:
  - Compute pairwise coherence matrix (channel x channel)
  - Identify outlier channel (lowest mean similarity to all others)
  - Compare against ground truth (human-judged outlier channels, annotated before running analysis)
  - Assess clarity: gap between outlier and second-lowest channel
- Pass criterion: correct outlier identification for >= 6 of 10 inconsistent products
- Secondary analysis: also run on the 10 "known consistent" products -- the metric should NOT identify a strong outlier (all channels roughly equivalent)

**Step 8: Experiment 3 -- Attribute-Level Drill-Down (`coherence/experiment_3.py`)** -- 6-8h

The most commercially valuable but technically riskiest experiment.

- Select 3 products with known attribute-level messaging gaps
- For each product, identify 3-5 key attributes with known channel-specific presence/absence
- Generate attribute probes at two context levels:
  - "sentence": single sentence mentioning the product + attribute
  - "paragraph": 3-4 sentences with product context
- Extract embeddings for probes and channel documents
- Compute attribute-channel similarity matrix: (n_attributes x n_channels)
- Compare against ground truth: which attributes are present/absent in which channels
- **Wrong-product control (Decision 11):** For each attribute probe, also compute similarity against documents for a DIFFERENT product in the same category. If the probe is equally similar to wrong-product documents, the metric is capturing category-level features, not product-specific content. Adds ~1h.
- Address synonym concern: test whether probes using synonyms/paraphrases (not identical keywords) still show high similarity
- Pass criterion: >= 2 of 3 products correctly characterized (>= 70% of attribute-channel pairs correct), AND wrong-product control shows meaningfully lower similarity than correct-product documents

**Completion criteria:**
- Experiment 2: outlier channel correctly identified for >= 6 of 10 products
- Experiment 3: attribute-level gaps correctly identified for >= 2 of 3 products; wrong-product control passes
- Both verdicts recorded

---

### Phase 4: Experiments 4-5 -- Extended Validation (Steps 9-10, ~8-12 hours)

Experiments 4 and 5 are technically independent of each other and can run in parallel after Experiment 1 passes. Data collection for these can begin during earlier phases.

**Step 9: Experiment 4 -- Temporal Coherence Drift (`coherence/experiment_4.py`)** -- 4-6h

- Select 2-3 products with accessible historical materials (Wayback Machine, press release archives)
- Plus 1 control product in the same category with stable messaging
- Collect materials from >= 3 time points (launch, mid-period, current)
- Compute coherence at each time point
- Subtract control product's trajectory to correct for secular language trends
- Check whether coherence changes correspond to known brand events (agency change, rebrand, category expansion)
- Pass criterion: coherence trajectory is non-random (Kendall tau test) and directionally correct for >= 2 of 3 products after control correction

**Step 10: Experiment 5 -- Competitive Coherence Benchmarking (`coherence/experiment_5.py`)** -- 4-6h

- Select one CPG category with 5+ competing brands
- Collect same channels per brand (regulatory, marketing, retail, social, consumer_review)
- Compute coherence scores and rank brands
- Experts predict rankings BEFORE seeing metric results (forced-choice, pre-registered)
- Generate expert evaluation form before running the metric
- 3 experts independently rank the brands
- Compute Kendall tau between metric ranking and each expert ranking
- Compute inter-expert Kendall tau as a baseline
- Pass criterion: metric-expert agreement >= inter-expert agreement, AND >= 7 of 10 pairwise brand comparisons agree with expert majority

**Completion criteria:**
- Experiment 4 verdict recorded
- Experiment 5 expert forms generated, evaluations collected, agreement computed
- Both verdicts recorded

---

## Risk Mitigations

### Critical (before infrastructure)

**M1: Real-document smoke test (Step 0).** — **COMPLETED, PASSED.**
- Confirmed directional signal in 6 real products across 2 categories with mean centering.
- Mean centering at layer 30 provides combined within-category gap of +0.426.

**M2: Mean centering as primary, contrastive fine-tuning as staged escalation.**
- Primary path: mean-centered Qwen2.5-32B at optimal layer (smoke test suggests ~30). Uses existing `extraction.py` and `analysis.correct_anisotropy()`. Zero new training.
- Escalation: sentence-transformer contrastive model, activated only if mean centering achieves d < 0.5 in Experiment 0. Pass criteria: register probe accuracy <= 0.5.
- This replaces the original plan's "contrastive first, partial RSA fallback" with "mean centering first, contrastive escalation."

**M3: Vocabulary narrowness control.**
- Smoke test showed Aquaphor (simple product, narrow vocabulary) scored 0.499 while all others scored below 0.10. Products with narrow vocabulary may inflate coherence scores.
- Compute type-token ratio per product across channels. If coherence correlates with inverse lexical diversity (Spearman rho > 0.5), the metric partially measures vocabulary narrowness.
- Mitigation: report vocabulary-adjusted coherence scores alongside raw scores, or include lexical diversity as a covariate.

### High priority (before Experiment 1)

**M4: Pre-registered metric definition with held-out validation.**
- 6 candidate combinations (not 128+), with Bonferroni correction
- Validated on held-out patent/journalistic register stimuli (not the training distribution)
- Locked in `metric_selection.json` before any real-document testing

**M5: Calibration transfer validation in Experiment 0.**
- Aggressive degradation tests: length variation (20-500 words), attribute removal (30%), non-LLM text rewrites
- If metric becomes random under moderate degradation, rethink before investing in document collection

**M6: Baseline battery.**
- TF-IDF cosine similarity and BERTScore alongside Experiment 1
- If any baseline matches hidden-state performance, adopt the simpler approach
- DeLong test for formal AUC comparison (not arbitrary 0.05 threshold)

**M7: Memory profiling for long documents.**
- Before collecting real documents: run extraction on synthetic inputs at 500, 1000, 2000, 5000, 8192 tokens
- Record peak GPU memory, verify 8192-token truncation ceiling is safe

### Medium priority (before later experiments)

**M8: Reproducibility verification.**
- Run extraction twice on same 5 stimuli, verify identical outputs
- Lock requirements (`requirements.lock` or pinned `pyproject.toml`)

---

## Document Collection Specification

| Channel | Source | Author | Tier | Typical Length |
|---------|--------|--------|------|----------------|
| Regulatory | FDA filings, drug facts panels, ingredient lists | Brand (constrained) | Brand | 500-5000 words |
| Marketing | Product website, primary product page, ad copy | Brand | Brand | 100-2000 words |
| Retail | Amazon listing, Walmart listing, Google Shopping | Brand (platform-mediated) | Brand | 100-500 words |
| Social | Brand's own social media posts (Twitter/X, TikTok, Instagram) | Brand | Brand | 20-300 words |
| Consumer Review | Amazon reviews, social media mentions | Third party | Market | 50-500 words per review |

Each document requires: raw_text, source_url, date_collected, channel, word_count, is_brand_controlled.

**Consumer review handling:** Collect 10 most recent verified-purchase reviews (>= 50 words) per product. Embed each review individually through the locked metric. The consumer_review channel embedding = mean of the 10 individual review embeddings. Do NOT concatenate reviews into a single document.

**Document length handling:** Use the long-context encoder (Jina v3, 8192 tokens) as the primary model. For documents exceeding 8192 tokens, truncate to first 8192 tokens, log a warning, record truncation in metadata. No overlapping-window chunking. If more than 20% of documents are truncated, revisit this decision.

**Document collection effort:** 40-80 hours of manual work, primarily for Experiment 1 (20 products x 3-5 documents each = 60-100 documents, plus 200 consumer reviews). Begin identifying candidate products and bookmarking sources during Phase 0 infrastructure work to parallelize.

---

## Trade-offs and Accepted Limitations

### T1: Mean centering (primary) vs. contrastive encoder (escalation)

**Chose:** Mean-centered Qwen2.5-32B at optimal layer as primary. Contrastive encoder as escalation.
**Trade-off:** Mean centering is simpler but less principled — it removes the global mean direction but doesn't explicitly learn to suppress register. The contrastive approach would create a purpose-built embedding space.
**Why accepted:** The smoke test demonstrated that mean centering produces clear within-category coherence differentiation (+0.215 skincare, +0.211 supplements) with zero training and zero new infrastructure. Building a contrastive training pipeline before confirming mean centering is insufficient would violate the plan's "validate before building" principle.

### T2: Pre-registered metric lock vs. flexibility

**Chose:** Experiment 0 explores a restricted 6-combination space, then locks the winner before any real-document testing.
**Trade-off:** If the locked metric underperforms on real documents, there is no post-hoc tuning without invalidating the pre-registration.
**Why accepted:** Methodological rigor. Post-hoc metric tuning on Experiment 1 data would be p-hacking. If the locked metric fails, the correct response is a new Experiment 0 iteration or a different approach.

### T3: Model-specific layer selection

**Chose:** Accept that the optimal layer is specific to Qwen2.5-32B and must be re-calibrated per model.
**Trade-off:** If the model becomes unavailable or a better model emerges, Experiment 0's layer sweep must be re-run. Scores from different models are not directly comparable.
**Why accepted:** Model-specificity is inherent to any hidden-state approach. The alternative (contrastive fine-tuning to create a model-agnostic embedding space) adds 12-18h of complexity and doesn't eliminate model dependence — it just moves it to the encoder architecture. The layer sweep is cheap (~1h compute) and can be amortized across engagements using the same model.

### T4: Content coherence only (no voice/tone)

**Chose:** Content coherence only for validation; voice consistency deferred.
**Trade-off:** Clients may expect voice/tone analysis. The contrastive fine-tuning deliberately collapses register, making it unable to detect voice inconsistencies.
**Why accepted:** Validating two metrics simultaneously doubles complexity. Content coherence is the novel contribution. Voice consistency (using base model register signatures) can be added after core validation -- the infrastructure for it is free since the base model is already available.

### T5: Validation-grade tooling only

**Chose:** Minimal scripts, no reporting infrastructure, no CLI, no commercial framing.
**Trade-off:** Results require manual interpretation; no polished deliverables.
**Why accepted:** Building production infrastructure before validating the core methodology wastes effort if the methodology fails. Commercial infrastructure (reporting, visualization, BCI scale, pipeline orchestration) is deferred to post-validation.

---

## Emergent Risks to Monitor

**The contrastive path is deferred, not eliminated.** The smoke test removed contrastive fine-tuning from the critical path, but mean centering has its own risks: (1) the optimal layer may shift with different document types or lengths, (2) the vocabulary narrowness confound (Aquaphor effect) may inflate scores for simple products, (3) mean centering removes one global direction but doesn't guarantee that the remaining directions encode coherence rather than other confounds. If Experiment 1 produces ambiguous results with mean centering, the contrastive escalation path adds 12-18h but addresses these concerns more directly.

**Document collection is the actual bottleneck.** Multiple analysis threads converge on this: collecting 60-100 real documents, assigning ground-truth labels, sampling consumer reviews, and verifying channel coverage per product is 40-80 hours of manual work that cannot be parallelized with coding. Begin informally during Phase 0.

**Expert validation confound.** In Experiment 5, experts ranking brands by coherence may draw on brand familiarity rather than messaging analysis. The plan mitigates this with forced-choice prediction BEFORE seeing metric results. If results are suspicious, re-examine familiarity as a post-hoc explanation.

---

## File Structure

```
langprod/
+-- coherence/
|   +-- __init__.py
|   +-- ingest.py              (Step 1: document ingestion)
|   +-- contrastive.py         (Step 2: contrastive fine-tuning)
|   +-- losses.py              (Step 2: loss functions)
|   +-- extraction.py          (Step 2: sentence-transformer embeddings)
|   +-- metrics.py             (Step 3: coherence metric computation)
|   +-- baselines.py           (Step 4: TF-IDF, BERTScore baselines)
|   +-- experiment_0.py        (Step 5: metric exploration)
|   +-- experiment_1.py        (Step 6: real-document sensitivity)
|   +-- experiment_2.py        (Step 7: channel attribution)
|   +-- experiment_3.py        (Step 8: attribute drill-down)
|   +-- experiment_4.py        (Step 9: temporal drift)
|   +-- experiment_5.py        (Step 10: competitive benchmarking)
+-- tests/
|   +-- test_ingest.py
|   +-- test_coherence_metrics.py
|   +-- test_baselines.py
+-- data/
|   +-- coherence/
|       +-- models/
|       |   +-- encoder/        (contrastive model checkpoint)
|       +-- exp0/               (metric exploration results)
|       +-- exp1/               (real-document sensitivity)
|       +-- exp2/               (channel attribution)
|       +-- exp3/               (attribute drill-down)
|       +-- exp4/               (temporal drift)
|       +-- exp5/               (competitive benchmarking)
|       +-- portfolios/         (raw document manifests)
+-- pyproject.toml              (new dependencies)
```

### New dependencies

```toml
"sentence-transformers>=3.0"    # contrastive fine-tuning + sentence embeddings
"trafilatura>=1.8"              # web document extraction
"beautifulsoup4>=4.12"          # HTML cleaning
```

Note: `peft>=0.10` is needed only if Qwen2.5-7B LoRA fallback is activated.

---

## Estimated Effort and Timeline

| Phase | Steps | Effort | Dependencies |
|-------|-------|--------|-------------|
| Pre-Phase: Smoke Test | 0 | 2-4h | None (existing pipeline) |
| Phase 0: Infrastructure | 1, 2, 3, 4 (parallel) | 20-28h | Smoke test passes |
| Phase 1: Experiment 0 | 5 | 6-8h | Phase 0 |
| Phase 2: Experiment 1 | 6 | 8-12h | Phase 1 + document collection |
| Phase 3: Experiments 2-3 | 7, 8 | 8-11h | Phase 2 pass |
| Phase 4: Experiments 4-5 | 9, 10 | 8-12h | Phase 2 pass + historical collection |
| **Total implementation** | | **52-75h** | |
| **Document collection** | | **40-80h** | Begins during Phase 0 |

**Critical path:** Step 0 (smoke test) --> Phase 0 (Step 2: contrastive training) --> Phase 1 (Experiment 0: metric lock) --> Phase 2 (Experiment 1: real-document validation).

**Calendar time estimate:** 4-6 weeks (part-time effort alongside other work, plus document collection lead time). More realistic than original 3-5 week estimate; accounts for collection effort.
