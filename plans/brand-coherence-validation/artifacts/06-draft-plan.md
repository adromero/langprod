# Brand Message Coherence: Unified Validation Plan

## Summary

This plan validates a quantitative methodology for measuring how consistently a product's core semantic message survives translation across communication channels (regulatory, marketing, retail, social, consumer). The methodology uses LLM hidden-state representations — either from a contrastive-finetuned encoder or from selected layers of Qwen2.5-32B with register correction — to produce a "coherence score" per product across channels.

The plan builds on a completed Protocol Layer Hypothesis experiment that produced an 800-stimulus calibration dataset and a working extraction/RSA pipeline. The central technical challenge is that register (communication style) dominates product identity in the model's representational geometry (RSA r=0.670 vs. r=0.371). The plan addresses this through contrastive fine-tuning with a partial-RSA fallback, validated across six experiments (0-5) with sequential gating.

Key user decisions incorporated:
1. Register correction via contrastive fine-tuning (with partial RSA fallback)
2. New Experiment 0 for metric exploration on calibration data
3. Baselines (TF-IDF, BERTScore) in Experiment 1 only
4. Pilot on existing 800 calibration stimuli before real-document collection
5. Two-tier reporting: brand coherence (controlled channels) + market coherence (including consumer reviews)
6. n=10 per group (20 products total) for Experiment 1

**Total estimated effort:** 53-77 hours of implementation, plus 2-5 days of document collection.

---

## Problem Statement

### What we have

A working research pipeline (Python: `stimuli.py`, `extraction.py`, `analysis.py`, `viz.py`, `run.py`) that:
- Generated 800 calibration stimuli (80 products x 5 registers x 2 paraphrase variants)
- Extracted hidden states from Qwen2.5-32B across all 64 layers
- Demonstrated via RSA that product identity signal exists at late layers (r=0.371 at layer 61), though register dominates (r=0.670 at every layer)
- Showed perfect register classification (F1=1.0) and near-perfect category classification (F1=0.99) via linear probes

### What we need to prove

1. That the product-identity signal can be isolated from the register signal (via contrastive fine-tuning or partial RSA)
2. That the resulting "coherence metric" discriminates between products with known-consistent vs. known-inconsistent messaging on real-world documents
3. That the metric provides channel-level attribution (which channel diverges)
4. That the metric can drill down to attribute-level gaps (what content is lost)
5. That the metric captures temporal drift and produces competitive rankings professionals recognize

### What would kill the project

- If contrastive fine-tuning fails AND partial RSA is insufficient to separate register from product signal (Experiment 0 gate)
- If the metric cannot distinguish known-consistent from known-inconsistent products on real documents (Experiment 1 gate)
- If simple baselines (TF-IDF, BERTScore) achieve comparable discrimination (value-added gate in Experiment 1)

---

## Proposed Approach

### Core methodology

1. **Train a register-invariant encoder** using contrastive fine-tuning on the 800-stimulus calibration set. Positive pairs = same product across registers; negative pairs = different products. This creates an embedding space where register is suppressed and product identity is the dominant axis.

2. **Pre-register the metric formula** in Experiment 0 (exploratory phase on calibration data), then lock it before any real-document testing. The metric produces two scores per product:
   - **Brand Coherence Score**: mean cosine similarity across brand-controlled channel pairs (regulatory, marketing, retail, social)
   - **Market Coherence Score**: mean cosine similarity across all channel pairs including consumer reviews

3. **Validate sequentially** through six experiments with explicit pass/fail gates.

### What "coherence" means and does not mean

Following the domain analysis, the methodology measures **semantic content alignment** — whether the same product story (claims, attributes, positioning) is told across channels, independent of how it is told. It explicitly does NOT measure:
- Voice/tone consistency (the contrastive fine-tuning deliberately collapses register)
- Visual/experiential consistency (outside scope of text analysis)
- Consumer perception alignment (measured separately in the "market coherence" tier)

This scoping must be communicated precisely in all commercial framing. The domain expert recommends the term "Semantic Content Alignment" or "Message Consistency Index" over the ambiguous "coherence."

---

## Architecture

### System overview

```
                    ┌──────────────────────────┐
                    │   Contrastive Fine-Tuner  │ (NEW: finetune.py)
                    │   Training: 800 calibration│
                    │   Output: encoder model    │
                    └───────────┬────────────────┘
                                │ model checkpoint
                                v
┌──────────┐    ┌───────────────────────────┐    ┌─────────────────┐
│ Documents │───>│    Extraction Pipeline    │───>│    Coherence     │
│ (ingest + │    │   (contrastive encoder OR │    │    Analysis      │
│  existing │    │    base Qwen + partial    │    │   (metrics,      │
│  stimuli) │    │    RSA fallback)          │    │    two-tier      │
└──────────┘    └───────────────────────────┘    │    reporting)    │
                                                  └─────────────────┘
```

### Component inventory

| Component | Status | Module |
|-----------|--------|--------|
| Document ingestion & cleaning | **NEW** | `coherence/ingest.py` |
| Contrastive fine-tuning | **NEW** | `coherence/contrastive.py`, `coherence/losses.py` |
| Coherence metrics | **NEW** | `coherence/metrics.py` |
| Baseline comparators | **NEW** | `coherence/baselines.py` |
| Two-tier reporting | **NEW** | `coherence/reporting.py`, `coherence/viz.py` |
| Experiment orchestration | **NEW** | `coherence/pipeline.py`, `coherence/experiment_*.py` |
| Hidden-state extraction | **EXTEND** | `extraction.py` (add contrastive model path) |
| RSA analysis | **EXTEND** | `analysis.py` (add coherence RDM computation) |
| Visualization | **EXTEND** | `viz.py` (add coherence-specific plots) |
| CLI orchestration | **EXTEND** | `run.py` (add coherence subcommands) |
| Existing pipeline | **UNCHANGED** | `stimuli.py`, `probes.py`, existing tests |

### Separate orchestrator decision

The architecture uses a **separate orchestrator** approach: `run.py` retains the existing Protocol Layer Hypothesis commands unchanged; new `coherence-*` subcommands are added to `run.py` but route to the `coherence/` package. All coherence data writes to `data/coherence/` to avoid any interference with the existing `data/` artifacts. Both paths share the underlying `extraction.py` and `analysis.py` modules via new functions (not modifications to existing functions).

### Data flow

**Experiment 0 (calibration pilot):**
```
data/stimuli.json (800 existing) ──> finetune.py ──> data/coherence/models/encoder/
                                                          │
data/Qwen_*_hidden_states.h5 ──> coherence/metrics.py ◄──┘
                                          │
                                          v
                            data/coherence/exp0/metric_selection.json (LOCKED)
```

**Experiments 1-5 (real documents):**
```
data/coherence/portfolios/exp{N}/ ──> ingest.py ──> extraction.py ──> coherence/metrics.py
                                                                              │
                                                                              v
                                                          data/coherence/exp{N}/results/
```

---

## Specialist Conflicts Resolved

### Conflict 1: Base model for contrastive fine-tuning

**Architect** recommends a sentence-transformer encoder (E5-large-v2, 335M params) fine-tuned with `sentence-transformers` library and `MultipleNegativesRankingLoss`.

**Implementer** proposes Qwen2.5-7B with LoRA adapters and custom NT-Xent contrastive loss, plus an optional gradient-reversal adversarial term.

**Resolution: Start with the sentence-transformer approach (Architect's recommendation), with Qwen2.5-7B as a fallback.**

Rationale:
- The sentence-transformer path is simpler (existing library handles pair construction), faster to train (~30 min vs. hours), and produces purpose-built embeddings. It fits easily on the RTX 5090 with room to spare.
- The 512-token context window limitation is addressable via chunking (already planned) or by using a long-context variant like `jinaai/jina-embeddings-v3` (8192 tokens).
- Qwen2.5-7B with LoRA is a valid fallback if the sentence-transformer fails to suppress register sufficiently, but it adds complexity (PEFT dependency, gradient checkpointing, memory pressure) that should only be incurred if needed.
- The Implementer's adversarial gradient-reversal term is a valuable addition to either approach if basic contrastive loss is insufficient. Keep it as an escalation option.

**Decision for Experiment 0:** Train both a sentence-transformer contrastive model AND test the base Qwen2.5-32B with partial RSA register correction. Compare register-suppression effectiveness. Lock the winner before Experiment 1.

### Conflict 2: Orchestration approach (run.py extension vs. separate run_coherence.py)

**Architect** recommends a separate `run_coherence.py` orchestrator to avoid regression risk to the existing pipeline.

**Implementer** proposes extending `run.py` with new `coherence-*` subcommands, following the existing CLI pattern.

**Resolution: Extend `run.py` with coherence subcommands, but route all logic to the `coherence/` package.**

Rationale:
- A single entry point is simpler for the user. Two orchestrators create confusion about which to run.
- The risk of regression is mitigated by routing all new subcommands to `coherence/pipeline.py` and its sub-modules — the new code does not touch existing `cmd_*` functions.
- The Architect's concern about shared module divergence is addressed by adding NEW functions to `extraction.py` and `analysis.py` (e.g., `extract_contrastive_embeddings()`, `compute_coherence_rdm()`) rather than modifying existing ones.
- Shared utilities (path helpers, config loading, seed management) are already in `run.py` and can be imported.

### Conflict 3: Experiment 1 design — binary groups vs. continuum ranking

**Domain Expert** recommends selecting 20 products spanning a continuum of expected coherence and validating via rank-order correlation (Spearman rho) with expert consensus, arguing the binary design introduces confirmation bias.

**Implementer** implements the binary design (10 "known consistent" + 10 "known inconsistent") with Mann-Whitney U test and AUC, per the original plan and Decision #8.

**Resolution: Use the binary design as the primary analysis, add the continuum analysis as a secondary check.**

Rationale:
- The binary design has a clear, pre-registered pass/fail criterion (AUC >= 0.85, at most 2 misclassifications). A continuum design's pass criterion (minimum Spearman rho?) is harder to pre-register convincingly.
- However, the domain expert's point about confirmation bias is valid. Mitigation: (a) document operational criteria for "known consistent" and "known inconsistent" BEFORE product selection, (b) after binary analysis, compute the rank-order correlation across all 20 products as a secondary measure, (c) add a face-validity check by presenting the full 20-product ranking to 2-3 industry professionals.
- The n=10 per group (Decision #8) provides sufficient power for the binary test (Mann-Whitney U power ~0.80 for Cohen's d=1.0).

### Conflict 4: What the coherence metric measures vs. what clients want

**Domain Expert** identifies that brand managers primarily mean voice/tone consistency when they say "coherence," but the contrastive fine-tuning (Decision #3) deliberately collapses register/voice.

**Architect and Implementer** treat register suppression as the correct technical approach.

**Resolution: Measure content coherence as the primary metric; add a voice consistency score using the base model as a secondary metric.**

Rationale:
- The domain expert's recommendation R6 (Option 3) is elegant: use the base Qwen2.5-32B model (before fine-tuning) to compute a "Voice Consistency Score" alongside the content-coherence score from the contrastive model. This leverages the finding that register is perfectly classifiable, turning the "nuisance variable" into a product feature.
- Implementation: during extraction, compute embeddings from both the contrastive model (content coherence) and the base model at an early/mid layer (voice similarity). Report both.
- This creates a 2x2 diagnostic: {high/low content coherence} x {high/low voice consistency}, which is commercially powerful. ("Your core message is consistent but your brand voice is fragmented across channels.")
- **However**, this is a Phase 2 enhancement. For validation (Experiments 0-5), focus solely on the content-coherence metric. Voice consistency can be added after the core methodology is validated.

### Conflict 5: Model choice — E5-large vs. long-context encoder

**Architect** suggests E5-large-v2 (512-token context) with chunking for long documents, or alternatively `jinaai/jina-embeddings-v3` (8192-token context).

**Risk Analyst** warns that documents over ~8000 tokens will OOM on the base Qwen model, and that chunking loses document-level structure.

**Resolution: Use a long-context sentence encoder (Jina v3 or similar) as the primary candidate; test E5-large with chunking as a fallback in Experiment 0.**

Rationale:
- Most real documents (regulatory filings, marketing pages) fall under 8192 tokens. A long-context encoder avoids the lossy chunking step for the vast majority of inputs.
- E5-large with chunking remains a valid fallback if the long-context model underperforms on short texts or fine-tuning.
- Test both in Experiment 0 and lock the winner.

### Conflict 6: Experiment 1 pass criterion stringency

**Seed document** requires "no overlap, or at worst one misclassification out of 10."

**Implementer** relaxes to "AUC >= 0.85, or at most 2 misclassifications out of 20" (adjusted for n=20).

**Domain Expert** says brand managers accept "directionally correct" tools, lowering the bar, but also demands zero "embarrassingly wrong" results.

**Resolution: Use the Implementer's statistical criterion (AUC >= 0.85, Mann-Whitney p < 0.05) as the hard gate. Add the domain expert's face-validity check as a soft requirement.**

Rationale:
- AUC >= 0.85 with n=20 is a rigorous statistical bar that accounts for the larger sample size.
- The face-validity check (no embarrassingly wrong outlier in the ranking) is important for commercial credibility but is subjective. It informs whether to proceed to commercialization, not whether the metric "works."
- Require the hidden-state method to outperform at least one baseline by AUC >= 0.05 (value-added test).

---

## Implementation Plan

### Phase 0: Infrastructure (Steps 1-4, ~17-25 hours)

Build the foundational modules. Steps 1, 2, and 4 can proceed in parallel.

**Step 1: Document Ingestion (`coherence/ingest.py`)** — 4-6h
- `RealDocument` dataclass with product_id, channel, text, source_url, date_collected, is_brand_controlled
- `load_documents()`, `clean_document()` (channel-aware boilerplate removal), `to_stimuli_format()`
- `chunk_document()` for long-document handling (overlapping chunks, configurable max_tokens)
- `validate_product_set()` to verify minimum channel coverage per product
- Channel taxonomy: `regulatory`, `marketing`, `retail`, `social`, `consumer_review`
- Cleaning: strip HTML, copyright notices, Amazon template chrome, hashtags/mentions, normalize Unicode
- Multi-product detection: flag documents mentioning multiple product names
- Tests: `test_ingest.py` covering cleaning, schema conversion, chunking, validation

**Step 2: Contrastive Fine-Tuning (`coherence/contrastive.py`, `coherence/losses.py`)** — 8-12h
- Primary path: sentence-transformer base (e.g., `jinaai/jina-embeddings-v3` or `intfloat/e5-large-v2`)
  - Use `sentence-transformers` library with `MultipleNegativesRankingLoss`
  - Training data: 800 calibration stimuli, positive pairs = same product different register
  - ~30 min training on RTX 5090
- Fallback path: Qwen2.5-7B with LoRA + custom NT-Xent loss
  - LoRA rank 8-16, gradient checkpointing mandatory
  - Optional adversarial register-suppression term (gradient reversal)
  - ~1-3 hours training
- `evaluate_register_factoring()`: verify register probe accuracy drops below 0.7, product RSA increases by >= 50%, no representation collapse
- Pass criteria for contrastive model before proceeding:
  - Register probe accuracy on held-out data < 0.7 (down from 1.0)
  - Same-product cross-register mean cosine similarity increases by >= 0.1
  - Product probe accuracy remains > 0.8

**Step 3: Coherence Metrics (`coherence/metrics.py`)** — 3-4h
- `compute_pairwise_coherence()`: cosine similarity matrix between all channel embeddings
- `compute_coherence_score()`: aggregates to scalar brand_coherence and market_coherence
  - Candidate aggregation methods (explored in Exp 0): mean pairwise, min pairwise, centroid distance, silhouette coefficient, partial RSA
- `compute_coherence_z_score()`: normalize against calibration distribution
- `identify_outlier_channel()`: channel with lowest mean similarity to all others
- `compute_attribute_coherence()`: attribute-probe x channel similarity matrix for Experiment 3
- Two-tier architecture: `CONTROLLED_CHANNELS = {"regulatory", "marketing", "retail", "social"}`
- Tests: perfect coherence (identical embeddings -> 1.0), zero coherence (orthogonal -> ~0.0), two-tier divergence, outlier detection

**Step 4: Baselines (`coherence/baselines.py`)** — 2-3h
- `compute_tfidf_coherence()`: TF-IDF cosine similarity (reuses existing BoW infrastructure)
- `compute_bertscore_coherence()`: sentence-BERT mean embedding cosine similarity
- `compare_methods()`: Spearman correlation between all method pairs, value-added assessment
- Same output structure as `compute_coherence_score()` for direct comparison
- Tests: identical texts -> 1.0, unrelated texts -> low, method comparison structure

**Completion criteria for Phase 0:**
- All four modules pass their unit tests
- Contrastive model trains and produces embeddings that pass register-factoring evaluation
- All modules produce outputs in compatible formats (same dict structure for metrics)

### Phase 1: Experiment 0 — Metric Exploration & Lock (Step 5, ~6-8 hours)

**Step 5: Experiment 0 (`coherence/experiment_0.py`)**

This is the exploratory phase on existing calibration data (Decision #4, #6). No new documents collected.

**Part A: Metric space exploration**
- For each of {raw, mean-centered, whitened, contrastive-finetuned}:
  - For each layer in {20, 40, 55, 60, 61, 62, 63, 64} (for base Qwen; single embedding for contrastive):
    - Compute per-product coherence across registers
    - Compute per-register coherence across products (control)
    - Compute effect size: within-product coherence vs. random-pairing coherence
- For the contrastive model: also vary the base encoder choice (if multiple candidates)
- Select the (correction method, layer/model, aggregation) combination with highest effect size

**Part B: Real-document condition simulation (Decision #6)**
- Length variation: truncate stimuli to 20-30 words (tweets), expand to 300-500 words (filings)
- Attribute removal: remove 1-3 core attributes from stimuli (simulate incomplete channels)
- Non-LLM text: manually rewrite 10-20 stimuli in human style (or source real human descriptions)
- Measure metric robustness under each degradation
- Robustness criterion: metric change < 0.2 standard deviations for moderate degradations

**Part C: Metric lock**
- Output: `data/coherence/exp0/metric_selection.json` — specifies:
  - Model: contrastive encoder path OR base Qwen layer number
  - Correction: raw/mean-centered/whitened
  - Pooling: mean/last-token
  - Aggregation: mean_pairwise/min_pairwise/centroid_distance/silhouette
  - Distance: cosine
- This file must exist before Experiment 1 can run (hard gate)

**Completion criteria:**
- Selected metric shows within-product coherence > between-product coherence (effect size d > 0.5)
- Length-variation degradation is tolerable (< 0.2 SD shift)
- Metric formula is locked in `metric_selection.json`

### Phase 2: Reporting & Experiment 1 (Steps 6-7, ~12-18 hours)

**Step 6: Two-Tier Reporting (`coherence/reporting.py`, `coherence/viz.py`)** — 4-6h
- `CoherenceReport` dataclass with brand_coherence, market_coherence, z-scores, outlier_channel, pairwise_matrix
- `generate_product_report()`, `generate_portfolio_report()`, `generate_competitive_report()`
- Export to JSON and Markdown
- Visualizations: pairwise heatmap, coherence comparison bar chart, channel radar, temporal drift line chart
- Brand Coherence Index (BCI): 0-100 scale calibrated against the 800-stimulus reference distribution
- Relationship diagnostic: {high/low brand coherence} x {high/low market coherence} interpretation

**Step 7: Experiment 1 — Real-Document Sensitivity (`coherence/experiment_1.py`)** — 8-12h

The core validation experiment. **This is the primary go/no-go gate.**

*Product selection (pre-registered criteria):*
- Document operational definitions BEFORE selecting products:
  - "Known consistent": large brand, strict regulatory alignment, unified agency, documented brand guidelines publicly available, single-brand product page
  - "Known inconsistent": startup/challenger brand, multiple agencies, known rebranding, social media voice disconnected from packaging, or demonstrably fragmented messaging
- Select 10 per group (Decision #8), 20 total
- Bias toward products where ground truth confidence is high

*Document collection:*
- For each product: 3-5 real documents from distinct channels
- Minimum channels per product: 3 (regulatory + marketing + at least one of retail/social/review)
- Channel taxonomy with tier flags:
  - Brand-controlled: regulatory, marketing, retail (if brand-authored), social (brand's own posts)
  - Third-party: consumer_review
- Each document: raw text + metadata (source_url, date_collected, channel, word_count)
- Preprocessing via `coherence/ingest.py`: boilerplate removal, encoding normalization, length handling

*Analysis:*
1. Extract embeddings using locked metric (contrastive model or base Qwen + correction)
2. Compute brand_coherence and market_coherence for all 20 products
3. Primary test: Mann-Whitney U between consistent and inconsistent groups (p < 0.05)
4. Effect size: Cohen's d (target: >= 0.8) and rank-biserial correlation
5. Classification: ROC AUC (target: >= 0.85), misclassification count (target: <= 2 of 20)
6. Baselines (Decision #5): compute TF-IDF and BERTScore coherence on the same documents
7. Value-added test: hidden-state AUC must exceed best baseline AUC by >= 0.05
8. Secondary analysis: rank-order correlation across all 20 products (Spearman rho)
9. Face-validity check: present full ranking to 2-3 industry professionals — no "embarrassingly wrong" outliers

*Pass criteria:*
- AUC >= 0.85 on consistent vs. inconsistent classification
- Mann-Whitney U p < 0.05
- At most 2 misclassifications out of 20
- Hidden-state method outperforms at least one baseline by AUC >= 0.05
- No product that a domain expert would consider "obviously wrong"

*Failure protocol:*
- If AUC < 0.70: fundamental methodology failure. Debug the register correction. Consider alternative approaches (supervised coherence predictor, register-appropriateness metric).
- If 0.70 <= AUC < 0.85: marginal. Investigate confounds (brand size, category effects). Consider increasing n or refining document preprocessing.
- If AUC >= 0.85 but baselines match: the hidden-state method adds no value. Pivot to simpler approach.

**Completion criteria:**
- All 20 products scored with brand_coherence and market_coherence
- Statistical tests completed with documented results
- Baseline comparison completed
- Pass/fail verdict recorded in `data/coherence/exp1/verdict.json`

### Phase 3: Experiments 2-3 — Diagnostic Validation (Steps 8-9, ~8-11 hours)

Phase 3 proceeds ONLY if Experiment 1 passes.

**Step 8: Experiment 2 — Channel Attribution (`coherence/experiment_2.py`)** — 2-3h

Reanalysis of Experiment 1 data. No new document collection.

- For each of the 10 "known inconsistent" products:
  - Compute pairwise coherence matrix (channel x channel)
  - Identify outlier channel (lowest mean similarity to all others)
  - Compare against ground truth (human-judged outlier channels, annotated before running analysis)
  - Assess clarity: gap between outlier and second-lowest channel
- Pass criterion: correct outlier identification for >= 6 of 10 inconsistent products
- Secondary analysis: also run on the 10 "known consistent" products — the metric should NOT identify a strong outlier (all channels roughly equivalent)

**Step 9: Experiment 3 — Attribute-Level Drill-Down (`coherence/experiment_3.py`, `coherence/probe_stimuli.py`)** — 6-8h

The most commercially valuable but technically riskiest experiment.

- Select 3 products with known attribute-level messaging gaps
- For each product, identify 3-5 key attributes with known channel-specific presence/absence
- Generate attribute probes at two context levels:
  - "sentence": single sentence mentioning the product + attribute
  - "paragraph": 3-4 sentences with product context
- Extract embeddings for probes and channel documents
- Compute attribute-channel similarity matrix: (n_attributes x n_channels)
- Compare against ground truth: which attributes are present/absent in which channels
- Address critique concern about keyword domination: test whether probes using synonyms/paraphrases (not identical keywords) still show high similarity
- Pass criterion: >= 2 of 3 products correctly characterized (>= 70% of attribute-channel pairs correct)

**Completion criteria:**
- Experiment 2: outlier channel correctly identified for >= 6 of 10 products
- Experiment 3: attribute-level gaps correctly identified for >= 2 of 3 products
- Both verdicts recorded

### Phase 4: Experiments 4-5 — Extended Validation (Steps 10-11, ~8-12 hours)

Experiments 4 and 5 are technically independent of each other and can run in parallel after Experiment 1 passes. They share data dependencies only with the locked metric and the contrastive model.

**Step 10: Experiment 4 — Temporal Coherence Drift (`coherence/experiment_4.py`)** — 4-6h

- Select 2-3 products with accessible historical materials (Wayback Machine, press release archives)
- Plus 1 control product in the same category with stable messaging
- Collect materials from >= 3 time points (launch, mid-period, current)
- Compute coherence at each time point
- Subtract control product's trajectory to correct for secular language trends
- Check whether coherence changes correspond to known brand events (agency change, rebrand, category expansion)
- Pass criterion: coherence trajectory is non-random (Kendall tau test) and directionally correct for >= 2 of 3 products after control correction

**Step 11: Experiment 5 — Competitive Coherence Benchmarking (`coherence/experiment_5.py`)** — 4-6h

- Select one CPG category with 5+ competing brands
- Collect same channels per brand (regulatory, marketing, retail, social, consumer_review)
- Compute coherence scores and rank brands
- Stronger design than original (addressing critique): experts predict rankings BEFORE seeing metric results (forced-choice, pre-registered)
- Generate expert evaluation form before running the metric
- 3 experts independently rank the brands
- Compute Kendall tau between metric ranking and each expert ranking
- Compute inter-expert Kendall tau as a baseline
- Pass criterion: metric-expert agreement >= inter-expert agreement, AND >= 7 of 10 pairwise brand comparisons agree with expert majority

**Completion criteria:**
- Experiment 4 verdict recorded
- Experiment 5 expert forms generated, evaluations collected, agreement computed
- Both verdicts recorded

### Phase 5: Integration (Step 12, ~2-3 hours)

**Step 12: CLI Integration and Pipeline Orchestration (`coherence/pipeline.py`)**

- Add subcommands to `run.py`:
  - `coherence-finetune`, `coherence-exp0` through `coherence-exp5`, `coherence-report`, `coherence-all`
- Implement sequential gating:
  - Exp 0 always runs first
  - Exp 1 gates on Exp 0 (metric_selection.json must exist)
  - Exp 2 runs on Exp 1 data (no new data gate, but Exp 1 must have run)
  - Exps 3, 4, 5 gate on Exp 1 pass (verdict.json passed=true)
  - `--force` flag for overriding gates during debugging
- Each experiment writes `verdict.json` with `{passed, timestamp, details, scores}`
- All outputs under `data/coherence/exp{N}/`

**Completion criteria:**
- All subcommands accessible via `python run.py --help`
- Pipeline gating enforced (cannot run Exp 1 without Exp 0 lock, cannot run Exp 3 without Exp 1 pass)
- Full pipeline runnable via `python run.py coherence-all`

---

## Risk Mitigations

### Critical (before Experiment 0)

**M1: Staged contrastive fine-tuning with fallback to partial RSA.**
- Train both a sentence-transformer contrastive model and test base Qwen with partial RSA
- Define pass/fail for the contrastive model: register probe accuracy < 0.7, product RSA increase >= 50%, no representation collapse
- If contrastive fine-tuning fails after reasonable hyperparameter search (~20 runs), fall back to partial RSA (already in `analysis.py`)
- Timeline: 1 week

**M2: Document preprocessing specification.**
- Define before collecting any documents:
  - Length normalization: chunk documents > 4000 tokens (2000-token windows, 500-token overlap); flag documents < 50 tokens
  - Boilerplate removal: HTML tags, nav elements, Amazon template, cookie notices, footer/header
  - Encoding: UTF-8 normalization, NFC Unicode, smart quote conversion
  - Multi-product detection: flag and split or exclude
- Implement as part of `coherence/ingest.py`
- Timeline: included in Step 1

**M3: Pre-registered metric definition.**
- Before Experiment 0 results: commit to the candidate metric space (which layers, which distances, which aggregations will be explored)
- After Experiment 0: lock the single winner in `metric_selection.json`
- Record in the decision log
- Timeline: 1 day

### High Priority (before Experiment 1)

**M4: Calibration transfer validation in Experiment 0.**
- Aggressive degradation tests: length variation (20-500 words), attribute removal (30%), non-LLM text rewrites
- If metric becomes random under moderate degradation, the methodology needs rethinking before investing in document collection
- Timeline: included in Step 5

**M5: Baseline battery.**
- TF-IDF cosine similarity, sentence-BERT embeddings, BM25 keyword overlap
- If any baseline matches hidden-state performance on Experiment 1, the RSA-based approach adds no value
- This is a legitimate experimental outcome, not a failure of implementation
- Timeline: included in Step 4

**M6: Memory profiling for long documents.**
- Before collecting real documents: run extraction on synthetic inputs at 500, 1000, 2000, 5000, 10000 tokens
- Record peak GPU memory, determine safe maximum, implement hard truncation
- Timeline: 2-3 hours

### Medium Priority (before commercial engagements)

**M7: Reproducibility verification.**
- Run extraction twice on same 5 stimuli, verify bit-identical outputs
- Lock requirements (Docker container or `requirements.lock`)
- Timeline: 1 day

**M8: Client data handling protocol.**
- Per-engagement isolated directories: `engagements/<client>-<date>/`
- Local-only processing (no cloud API calls with client data)
- Secure deletion script (`cleanup.py`)
- Data retention and deletion contractual terms
- Timeline: 1-2 days

**M9: Model version pinning.**
- Pin model version, tokenizer, transformers library, CUDA toolkit
- Plan for recalibration every 12-18 months (re-extraction, re-training contrastive model)
- Communicate to clients that scores are model-version-specific
- Timeline: ongoing

---

## Domain Requirements

### Document collection specification

| Channel | Source | Author | Tier | Typical length |
|---------|--------|--------|------|----------------|
| Regulatory | FDA filings, drug facts panels, ingredient lists | Brand (constrained) | Brand | 500-5000 words |
| Marketing | Product website, primary product page, ad copy | Brand | Brand | 100-2000 words |
| Retail | Amazon listing, Walmart listing, Google Shopping | Brand (platform-mediated) | Brand | 100-500 words |
| Social | Brand's own social media posts (Twitter/X, TikTok, Instagram) | Brand | Brand | 20-300 words |
| Consumer review | Amazon reviews, Yelp, social media mentions | Third party | Market | 50-500 words |

Each document requires: raw_text, source_url, date_collected, channel, word_count, is_brand_controlled.

### Scoring requirements

- **Brand Coherence Index (BCI):** 0-100 scale. Calibrated against the 800-stimulus reference distribution. 100 = all attributes present in identical semantic framing across all brand-controlled channels. 0 = channels discuss completely unrelated content.
- **Channel-pair decomposition:** N x N similarity matrix for diagnostic drill-down.
- **Outlier channel identification:** the single channel most divergent from the consensus.
- **Attribute-level drill-down (if Experiment 3 passes):** per-attribute presence/absence per channel.
- **Competitive percentile (if Experiment 5 passes):** rank within category.
- **Two-tier presentation:** brand coherence (primary, actionable) and market coherence (context, not directly actionable by the brand).

### Commercial framing

- Frame as "Semantic Content Alignment," not "coherence" (avoids conflation with voice/tone)
- Explicitly scope: measures whether the same product story is told across channels, independent of style
- Does NOT measure: visual identity, brand voice/tone, consumer perception, emotional positioning
- Voice/tone consistency is a Phase 2 addition (using base model register signatures)
- Deliverable is a consulting report with strategic recommendations, not a dashboard
- Price point ($25-75K) is justified by interpretation and recommendations, not by the metric alone

### Edge cases to scope

- **Multi-product brands:** methodology applies at individual product level; multi-product pages need content isolation
- **Co-branded products:** scope in engagement contract — coherence of the co-brand entity or per-brand contribution
- **International variations:** each market treated as separate study; multilingual is a future extension
- **Private label / store brands:** likely not viable (too few channels)
- **Products undergoing rebranding:** high-value use case (measure rollout completeness)
- **Seasonal / limited editions:** exclude promotional content; scope to core product messaging
- **Negative press / recalls:** valuable diagnostic — high brand coherence + low market coherence = external narrative divergence

---

## Trade-offs & Decisions

### T1: Contrastive encoder vs. base Qwen layer analysis

**Chose:** Contrastive encoder as primary, base Qwen as fallback.
**Trade-off:** Loses the rich 64-layer representational profile from the original experiment. The coherence metric becomes a single number from a single model.
**Why accepted:** The validation plan needs a single coherence score, not a layer-by-layer profile. The contrastive approach directly addresses the central problem (register dominance). The base Qwen path via partial RSA is preserved as a fallback.

### T2: Pre-registered metric lock vs. flexibility

**Chose:** Experiment 0 explores freely, then locks metric before any real-document testing.
**Trade-off:** If the locked metric underperforms on real documents, there is no post-hoc tuning without invalidating the pre-registration.
**Why accepted:** Methodological rigor. Post-hoc metric tuning on Experiment 1 data would be p-hacking. If the locked metric fails, the correct response is to debug the metric (new Experiment 0 iteration) or reconsider the approach — not to tune on the test set.

### T3: Sentence-transformer vs. Qwen-scale contrastive model

**Chose:** Sentence-transformer primary, Qwen2.5-7B fallback.
**Trade-off:** Smaller model may have weaker representational capacity; 512-token context window requires chunking.
**Why accepted:** Training is ~100x faster, inference is ~100x faster, memory requirements are minimal. If the sentence-transformer cannot suppress register, Qwen2.5-7B with LoRA is a viable escalation.

### T4: Binary group design vs. continuum ranking for Experiment 1

**Chose:** Binary as primary, continuum as secondary.
**Trade-off:** Binary design has clearer pass/fail criterion but risks confirmation bias.
**Why accepted:** Pre-registered operational criteria for group assignment mitigate bias. The continuum analysis (Spearman rho across all 20) provides a secondary check. Face-validity review catches embarrassing outliers.

### T5: Content coherence only vs. content + voice coherence

**Chose:** Content coherence only for validation; voice consistency as Phase 2.
**Trade-off:** Clients may expect voice/tone analysis. The methodology cannot measure this during validation.
**Why accepted:** Trying to validate two metrics simultaneously doubles the experimental complexity. Content coherence is the novel contribution. Voice consistency (using base model register signatures) can be added after the core is validated, and the infrastructure for it is free (base model is already available).

### T6: Sequential gating vs. parallel experiments

**Chose:** Sequential gating with parallel opportunities.
**Trade-off:** Strict sequential execution is slower than running all experiments in parallel.
**Why accepted:** Experiments 2, 3, 4, 5 depend on the metric being valid (Experiment 1). Running them on an unvalidated metric wastes effort. However, the data collection for Experiments 4 and 5 can begin in parallel with Experiment 1, and the actual analysis runs after Experiment 1 passes.

---

## Open Questions

### High priority (resolve before Experiment 0)

1. **Which long-context sentence encoder?** Need to select the base encoder for contrastive fine-tuning. Candidates: `jinaai/jina-embeddings-v3` (8192 tokens), `intfloat/e5-large-v2` (512 tokens + chunking), `BAAI/bge-large-en-v1.5` (512 tokens). Run a quick benchmark on calibration data to compare base embedding quality before fine-tuning.

2. **How many manually rewritten stimuli for the non-LLM transfer test in Experiment 0?** The risk analyst recommends 10-20 human-rewritten stimuli. This is labor-intensive. Is 10 sufficient for a directional signal?

3. **What is the minimum commercially relevant effect size?** Cohen's d >= 0.8 is proposed for Experiment 1, but what coherence score difference translates to actionable insight for a client? Answering this requires input from potential buyers (domain expert recommendation R8).

### Medium priority (resolve before Experiment 1)

4. **Multi-product document handling strategy.** Some documents mention multiple products. Options: (a) manually segment, (b) exclude, (c) use the full document and accept noise. Need a decision before document collection begins.

5. **Consumer review sampling strategy.** A single review is noisy. How many reviews per product should be collected? Concatenate or individually analyze? The domain expert suggests 5-10 curated reviews per product, concatenated into a single "consumer voice" document for the market coherence tier.

6. **Model migration plan.** When Qwen2.5-32B becomes obsolete, what is the recalibration protocol? The risk analyst recommends a planned 12-18 month recalibration cycle with version pinning. Accept model-version-specific baselines and communicate this to clients.

### Low priority (resolve before commercialization)

7. **Competitive differentiation narrative.** The domain expert notes no existing tool measures cross-channel semantic coherence of brand-authored materials. This is the genuine whitespace. But the pitch needs to articulate why this is better than "have ChatGPT compare your documents" or "hire an intern." The answer is: quantitative, reproducible, benchmarked, and backed by a validated methodology. This needs to be articulated before the first sales conversation.

8. **Voice consistency as a Phase 2 product feature.** The domain expert's option 3 (register similarity using base model as a "Voice Consistency Score") is compelling and nearly free to implement. Decide the timeline for adding this after core validation.

9. **Data compounding strategy.** Each engagement generates labeled real-world data. Plan for: (a) calibration improvement over time, (b) potential supervised fine-tuning once enough labeled examples exist, (c) anonymized benchmark publication. This requires client consent and data retention provisions in the engagement contract.

10. **Buyer validation.** The domain expert's recommendation R8 is to have 2-3 conversations with potential buyers before completing all five experiments. Show a mock report, ask if they would pay $50K, learn what they actually want. This costs zero and could redirect the entire effort. Schedule before Experiment 1 document collection begins.

---

## File Structure

```
langprod/
├── coherence/
│   ├── __init__.py
│   ├── ingest.py           (Step 1: document ingestion)
│   ├── contrastive.py      (Step 2: contrastive fine-tuning)
│   ├── losses.py           (Step 2: loss functions)
│   ├── metrics.py          (Step 3: coherence metric computation)
│   ├── baselines.py        (Step 4: TF-IDF, BERTScore baselines)
│   ├── experiment_0.py     (Step 5: metric exploration)
│   ├── reporting.py        (Step 6: two-tier reports)
│   ├── viz.py              (Step 6: coherence visualizations)
│   ├── experiment_1.py     (Step 7: real-document sensitivity)
│   ├── experiment_2.py     (Step 8: channel attribution)
│   ├── experiment_3.py     (Step 9: attribute drill-down)
│   ├── probe_stimuli.py    (Step 9: attribute probe generation)
│   ├── experiment_4.py     (Step 10: temporal drift)
│   ├── experiment_5.py     (Step 11: competitive benchmarking)
│   └── pipeline.py         (Step 12: orchestration & gating)
├── tests/
│   ├── test_ingest.py
│   ├── test_coherence_metrics.py
│   ├── test_baselines.py
│   └── test_reporting.py
├── data/
│   └── coherence/
│       ├── models/
│       │   └── encoder/     (contrastive model checkpoint)
│       ├── exp0/            (metric exploration results)
│       ├── exp1/            (real-document sensitivity)
│       ├── exp2/            (channel attribution)
│       ├── exp3/            (attribute drill-down)
│       ├── exp4/            (temporal drift)
│       ├── exp5/            (competitive benchmarking)
│       └── portfolios/      (raw document manifests)
├── run.py                   (extended with coherence-* subcommands)
└── pyproject.toml           (new dependencies)
```

### New dependencies

```toml
"sentence-transformers>=3.0"    # contrastive fine-tuning + BERTScore baseline
"trafilatura>=1.8"              # web document extraction
"beautifulsoup4>=4.12"          # HTML cleaning
```

Note: `peft>=0.10` is needed only if Qwen2.5-7B LoRA fallback is activated.

---

## Estimated Effort & Timeline

| Phase | Steps | Effort | Dependencies |
|-------|-------|--------|-------------|
| Phase 0: Infrastructure | 1, 2, 3, 4 (parallel) | 17-25 hours | None |
| Phase 1: Experiment 0 | 5 | 6-8 hours | Phase 0 |
| Phase 2: Reporting + Exp 1 | 6, 7 | 12-18 hours | Phase 1 + document collection (2-5 days) |
| Phase 3: Exps 2-3 | 8, 9 | 8-11 hours | Phase 2 pass |
| Phase 4: Exps 4-5 | 10, 11 | 8-12 hours | Phase 2 pass + historical collection |
| Phase 5: Integration | 12 | 2-3 hours | Phases 0-4 |
| **Total** | | **53-77 hours** | |

**Critical path:** Phase 0 (Step 2: contrastive training) -> Phase 1 (Experiment 0: metric lock) -> Phase 2 (Experiment 1: real-document validation).

**Calendar time estimate:** 3-5 weeks (assuming part-time effort alongside other work, plus document collection lead time).

---

## Decision Framework

After the experiments, the project lands in one of these positions:

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| All 6 experiments pass | Methodology validated end-to-end | Package as commercial offering |
| Exps 0-2 pass, 3+ fail | Metric works at product level, not attribute level | Offer as "coherence scorecard" — simpler but viable |
| Exp 1 passes but baselines match | Hidden states add no value over simpler methods | Pivot to simpler approach (TF-IDF/BERT-based coherence) |
| Exp 0 fails (contrastive + partial RSA both fail) | Cannot separate register from content | Fundamental rethink: supervised coherence predictor, or different approach entirely |
| Exp 1 fails (metric doesn't discriminate) | Doesn't work on real documents | Debug preprocessing, length handling, document quality; if still fails, methodology is not viable |
| Exps 1-4 pass, Exp 5 fails (experts disagree with ranking) | Metric measures something real but not commercially recognized | Reframe the offering around what the metric DOES capture; investigate surprise insights |
