# Implementation Plan: Brand Coherence Validation Experiments

## Overview

This plan implements the validated and refined research design for Brand Message Coherence experiments 0-5, incorporating all decisions from the critique review:

1. **Contrastive fine-tuning** of a smaller model to factor out register signal (Decision 3)
2. **Experiment 0** as a metric exploration phase on calibration data (Decision 4)
3. **Baselines** (TF-IDF, BERTScore) alongside Experiment 1 only (Decision 5)
4. **Pilot on existing 800 calibration stimuli** before real-document collection (Decision 6)
5. **Two-tier reporting**: brand coherence (controlled channels) + market coherence (including consumer reviews) (Decision 7)
6. **n=10 per group** (20 products total) for Experiment 1 (Decision 8)

The implementation extends the existing `langprod` pipeline — a Python codebase with modules `stimuli.py`, `extraction.py`, `analysis.py`, `viz.py`, and `run.py` — using its established patterns: JSON stimuli, HDF5 hidden states, numpy arrays, scipy/sklearn analysis, and a CLI driven by `run.py` subcommands.

## Prerequisites

- Existing pipeline operational: `run.py generate`, `extract`, `analyze`, `probe`, `report` all functional
- 800 calibration stimuli in `data/stimuli.json` (80 products x 5 registers x 2 variants)
- HDF5 hidden states extracted from Qwen2.5-32B (`data/Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4_hidden_states.h5`)
- Python 3.11+, torch, transformers, scipy, scikit-learn, h5py already installed
- New dependencies needed: `sentence-transformers` (for BERTScore/sentence-BERT baselines), `peft` (for efficient fine-tuning)

---

## Implementation Steps

### Step 1: Real-Document Ingestion Module

**Complexity:** Medium
**Files:**
- Create: `coherence/ingest.py`
- Create: `coherence/__init__.py`
- Modify: `pyproject.toml` (add `beautifulsoup4`, `trafilatura` to dependencies)
- Create: `tests/test_ingest.py`

**Description:**
Build an adapter that ingests real-world documents (variable length, different channels) and normalizes them into the existing stimulus format expected by `extraction.py`. The existing pipeline expects dicts with `stimulus_id` and `text` keys; this module converts real documents into that shape with additional metadata for channel, source URL, date collected, and product association.

**Details:**
```
coherence/ingest.py:
    - RealDocument dataclass:
        product_id: str
        channel: str  (one of: regulatory, marketing, retail, social, review)
        text: str
        source_url: str | None
        date_collected: str
        is_brand_controlled: bool  (True for regulatory/marketing/social; False for review)

    - load_documents(path: str | Path) -> list[RealDocument]:
        Load from JSON/JSONL file with validation.

    - clean_document(doc: RealDocument) -> RealDocument:
        Strip boilerplate: navigation elements, copyright notices, Amazon template
        text, hashtags, mentions. Normalize whitespace. Return cleaned copy.

    - to_stimuli_format(docs: list[RealDocument]) -> list[dict]:
        Convert to stimulus dicts with stimulus_id = f"{product_id}_{channel}_{hash[:8]}",
        preserving all metadata fields alongside required 'stimulus_id' and 'text'.

    - chunk_document(doc: RealDocument, max_tokens: int = 150, overlap: int = 30) -> list[RealDocument]:
        For documents exceeding max_tokens, split into overlapping chunks.
        Each chunk gets stimulus_id = f"{original_id}_chunk{N}".
        This addresses the length-variation concern (critique gap #4).

    - validate_product_set(docs: list[RealDocument], min_channels: int = 3) -> dict:
        Verify each product has at least min_channels distinct channel documents.
        Return validation report.
```

**Tests:**
- `test_clean_document`: verify boilerplate removal (HTML tags, copyright, hashtags)
- `test_to_stimuli_format`: verify output matches extraction.py expected format
- `test_chunk_document`: verify chunk boundaries respect token limits with overlap
- `test_validate_product_set`: verify validation catches incomplete product sets

**Acceptance Criteria:**
- `load_documents()` successfully loads a sample JSON of 5 products x 3 channels
- `clean_document()` removes at least HTML tags, copyright text, navigation boilerplate
- `to_stimuli_format()` output is directly consumable by `extraction.extract_hidden_states()`
- `chunk_document()` produces chunks within the specified token range
- All unit tests pass

**Dependencies:** None (first step)

---

### Step 2: Contrastive Fine-Tuning Module

**Complexity:** High
**Files:**
- Create: `coherence/contrastive.py`
- Create: `coherence/losses.py`
- Modify: `pyproject.toml` (add `peft>=0.10`, `sentence-transformers>=2.6`)

**Description:**
Implement contrastive fine-tuning of a smaller model (e.g., Qwen2.5-7B or a sentence-transformer base) using the existing 800 calibration stimuli. The contrastive loss pulls same-product representations together and pushes different-product representations apart, *regardless of register*. This creates an embedding space where register is factored out and product identity is the dominant axis — directly addressing the critique's central concern about register dominance (RSA r=0.670 vs r=0.371).

**Details:**
```
coherence/losses.py:
    - supervised_contrastive_loss(embeddings, product_labels, register_labels, temperature=0.07):
        Positive pairs: same product, any register.
        Negative pairs: different product, any register.
        Uses NT-Xent (normalized temperature-scaled cross entropy).
        Register labels are NOT used in pair selection — this is the key design choice
        that forces the model to learn register-invariant product representations.

    - register_adversarial_term(embeddings, register_labels, lambda_adv=0.1):
        Optional adversarial term that penalizes register discriminability.
        Gradient reversal: maximize register classification loss.
        Only used if basic contrastive loss is insufficient.

coherence/contrastive.py:
    - ContrastiveConfig dataclass:
        base_model: str = "Qwen/Qwen2.5-7B-Instruct"
        lora_rank: int = 16
        lora_alpha: int = 32
        learning_rate: float = 2e-5
        batch_size: int = 16
        epochs: int = 10
        temperature: float = 0.07
        use_adversarial: bool = False
        lambda_adv: float = 0.1
        output_dir: str = "data/contrastive_model/"

    - build_training_pairs(stimuli: list[dict]) -> Dataset:
        From 800 calibration stimuli, construct positive (same product) and
        negative (different product) pairs. Each product has 10 stimuli
        (5 registers x 2 variants), yielding C(10,2)=45 positive pairs per
        product and abundant negatives.

    - fine_tune(config: ContrastiveConfig, stimuli: list[dict]) -> Path:
        1. Load base model with LoRA adapters (via peft)
        2. Register forward hooks on late layers (reuse extraction.py pattern)
        3. Train with contrastive loss
        4. Save LoRA weights only
        Return path to saved model.

    - extract_coherence_embeddings(model_path: Path, texts: list[str]) -> np.ndarray:
        Load fine-tuned model, extract mean-pooled late-layer representations.
        Returns (N, D) array in the register-factored embedding space.

    - evaluate_register_factoring(model_path: Path, stimuli: list[dict]) -> dict:
        Run RSA on fine-tuned embeddings to verify:
        (a) product-identity RSA increased relative to base model
        (b) register-identity RSA decreased relative to base model
        Return comparison metrics.
```

**Tests:**
- `test_contrastive_loss_same_product`: loss is lower for same-product pairs than different-product
- `test_contrastive_loss_gradient`: verify gradients flow correctly
- `test_build_training_pairs`: verify pair construction covers all products and registers
- `test_register_adversarial_term`: verify gradient reversal direction

**Acceptance Criteria:**
- Contrastive loss converges on a small subset (5 products) within 3 epochs
- After fine-tuning on calibration data, product-identity RSA increases by at least 50% relative to base model at the same layer
- Register-identity RSA decreases by at least 30% relative to base model
- `evaluate_register_factoring()` passes both checks above

**Dependencies:** None (can proceed in parallel with Step 1)

---

### Step 3: Coherence Metric Computation Module

**Complexity:** Medium
**Files:**
- Create: `coherence/metrics.py`
- Create: `tests/test_coherence_metrics.py`

**Description:**
Define the coherence metric formally, pre-registered before any experiments. The metric operates on the register-factored embedding space (from Step 2) and produces two scores per product: brand coherence (controlled channels only) and market coherence (all channels including reviews). This implements Decision 7 (two-tier reporting).

**Details:**
```
coherence/metrics.py:
    - compute_pairwise_coherence(embeddings: np.ndarray, channel_labels: list[str]) -> np.ndarray:
        Compute cosine similarity between every pair of channel embeddings for a
        single product. Returns an (n_channels, n_channels) similarity matrix.

    - compute_coherence_score(embeddings: np.ndarray, channel_labels: list[str],
                              brand_controlled_channels: set[str] | None = None) -> dict:
        Aggregate pairwise coherence into scalar scores:
        - "brand_coherence": mean cosine similarity across brand-controlled channel pairs only
          (regulatory, marketing, retail, social — exclude reviews)
        - "market_coherence": mean cosine similarity across ALL channel pairs (including reviews)
        - "min_pairwise": minimum pairwise similarity (identifies the weakest link)
        - "outlier_channel": the channel with lowest mean similarity to all others
        - "pairwise_matrix": full similarity matrix for drill-down

        Uses the CONTROLLED_CHANNELS constant to determine which channels are brand-controlled.

    - CONTROLLED_CHANNELS = {"regulatory", "marketing", "retail", "social"}
    - ALL_CHANNELS = {"regulatory", "marketing", "retail", "social", "review"}

    - compute_coherence_z_score(score: float, calibration_scores: np.ndarray) -> float:
        Convert a raw coherence score to a z-score relative to the calibration
        distribution. This makes scores interpretable: "your brand is 1.2 standard
        deviations above the calibration mean."

    - identify_outlier_channel(pairwise_matrix: np.ndarray, channel_labels: list[str]) -> dict:
        From the pairwise similarity matrix, identify the channel with lowest
        mean similarity to all other channels. Return channel name, its mean
        similarity, and the second-lowest for comparison.

    - compute_attribute_coherence(product_embeddings: np.ndarray,
                                   attribute_probe_embeddings: np.ndarray,
                                   channel_labels: list[str],
                                   attribute_labels: list[str]) -> dict:
        For Experiment 3: compute similarity between each attribute probe and each
        channel document. Returns a (n_attributes, n_channels) similarity matrix
        showing which attributes are present in which channels.
```

**Tests:**
- `test_perfect_coherence`: identical embeddings yield brand_coherence = 1.0
- `test_zero_coherence`: orthogonal embeddings yield brand_coherence near 0.0
- `test_two_tier_separation`: brand_coherence != market_coherence when review diverges
- `test_outlier_detection`: known-outlier channel is correctly identified
- `test_attribute_coherence_matrix_shape`: correct shape with correct labels

**Acceptance Criteria:**
- `compute_coherence_score()` returns all documented keys
- Two-tier scores diverge when review channel differs from controlled channels
- Outlier detection correctly identifies a synthetically displaced channel
- All unit tests pass

**Dependencies:** Step 2 (needs the embedding space definition, though can use raw embeddings for testing)

---

### Step 4: Baseline Comparators

**Complexity:** Low
**Files:**
- Create: `coherence/baselines.py`
- Create: `tests/test_baselines.py`

**Description:**
Implement TF-IDF and BERTScore baseline coherence metrics that run alongside the hidden-state method in Experiment 1 (Decision 5). These serve as sanity checks: if a bag-of-words approach achieves comparable separation to the LLM hidden-state method, the methodology adds no value.

**Details:**
```
coherence/baselines.py:
    - compute_tfidf_coherence(texts: list[str], channel_labels: list[str]) -> dict:
        Fit TF-IDF on the texts, compute pairwise cosine similarity.
        Return same structure as compute_coherence_score() for direct comparison:
        {"brand_coherence": float, "market_coherence": float, "pairwise_matrix": np.ndarray}

    - compute_bertscore_coherence(texts: list[str], channel_labels: list[str],
                                   model_name: str = "all-MiniLM-L6-v2") -> dict:
        Encode texts with a sentence-transformer model, compute pairwise cosine similarity.
        Return same structure as compute_coherence_score().

    - compare_methods(hidden_state_scores: dict, tfidf_scores: dict,
                      bertscore_scores: dict) -> dict:
        Compute correlation between the three methods' coherence scores.
        Return Spearman r between each pair of methods and a "value_added" assessment:
        does the hidden-state method separate groups that the baselines cannot?
```

**Tests:**
- `test_tfidf_coherence_identical`: identical texts yield coherence = 1.0
- `test_tfidf_coherence_different`: unrelated texts yield low coherence
- `test_compare_methods_structure`: output dict has expected keys
- `test_bertscore_coherence_runs`: smoke test with 3 short texts

**Acceptance Criteria:**
- Both baseline methods produce coherence scores in [0, 1]
- `compare_methods()` returns valid Spearman correlations
- All unit tests pass

**Dependencies:** None (independent of Steps 1-3, but used alongside Step 3 in Experiment 1)

---

### Step 5: Experiment 0 — Metric Exploration on Calibration Data

**Complexity:** Medium
**Files:**
- Create: `coherence/experiment_0.py`
- Modify: `run.py` (add `coherence-exp0` subcommand)

**Description:**
Exploratory analysis on the existing 800 calibration stimuli to determine the coherence metric's properties before committing to a formula (Decision 4). This is the "pilot on calibration data" (Decision 6). Systematically tests: which layers, what pooling, what distance metric, mean vs. last-token pooling, and the effect of contrastive fine-tuning.

**Details:**
```
coherence/experiment_0.py:
    - run_metric_exploration(config: dict, stimuli_meta: list[dict], h5_path: str) -> dict:
        1. Load calibration hidden states from HDF5
        2. For each of [raw, mean-centered, whitened, contrastive-finetuned]:
            a. For each layer in [20, 40, 55, 60, 61, 62, 63, 64] (targeted sweep):
                - Compute per-product coherence across registers
                - Compute per-register coherence across products (control)
                - Compute effect size: product coherence vs. random-pairing coherence
            b. Select the (correction, layer) combination with highest effect size
        3. Simulate real-document conditions on calibration data:
            a. Truncate stimuli to 20-30 words (simulate tweets)
            b. Expand stimuli to 300-500 words (simulate regulatory filings)
            c. Remove 1-2 attributes from stimuli (simulate incomplete channel docs)
            d. Re-run coherence computation on degraded stimuli
            e. Measure metric robustness: how much does coherence change?
        4. Pre-register the metric formula: lock (correction, layer, pooling, aggregation)

        Returns comprehensive results dict with all metric variants and selections.

    - simulate_length_variation(stimuli: list[dict], target_lengths: list[int]) -> list[dict]:
        Create length-varied versions of calibration stimuli by truncation or
        padding with neutral filler. Tests metric sensitivity to document length.

    - simulate_attribute_removal(stimuli: list[dict], removal_rate: float = 0.3) -> list[dict]:
        Remove a fraction of attribute mentions from stimuli text.
        Tests whether the metric detects missing attributes.

    - generate_exp0_report(results: dict, output_dir: str) -> Path:
        Generate a structured JSON + markdown summary of Experiment 0 findings.
        Includes the pre-registered metric formula.

run.py modifications:
    - Add subcommand: `python run.py coherence-exp0`
    - Follows existing pattern: set_global_seed, import module, run, save results
```

**Tests:**
- `test_metric_exploration_runs`: smoke test on synthetic 20-stimulus data
- `test_simulate_length_variation`: output stimuli have correct lengths
- `test_simulate_attribute_removal`: text content is actually reduced

**Acceptance Criteria:**
- Experiment 0 produces a locked metric formula (layer, correction, pooling, aggregation)
- The selected metric shows higher within-product coherence than between-product coherence on calibration data (effect size d > 0.5)
- Length-variation simulation shows metric change is < 0.2 standard deviations for moderate length changes (50-200 words)
- Results are saved in a structured format compatible with the pipeline

**Dependencies:** Step 2 (contrastive model needed for one of the correction methods), Step 3 (metric definitions)

---

### Step 6: Two-Tier Reporting Module

**Complexity:** Medium
**Files:**
- Create: `coherence/reporting.py`
- Create: `coherence/viz.py`
- Create: `tests/test_reporting.py`

**Description:**
Implement the two-tier reporting framework (Decision 7): brand coherence (controlled channels only) and market coherence (including consumer reviews). Generate structured reports and visualizations suitable for a consulting deliverable.

**Details:**
```
coherence/reporting.py:
    - CoherenceReport dataclass:
        product_id: str
        product_name: str
        brand_coherence: float
        market_coherence: float
        brand_coherence_z: float  (z-score relative to calibration)
        market_coherence_z: float
        outlier_channel: str | None
        pairwise_matrix: dict  (channel -> channel -> similarity)
        channel_rankings: list[dict]  (sorted by mean similarity)
        tier: str  ("brand" or "market")

    - generate_product_report(product_scores: dict, calibration_stats: dict) -> CoherenceReport:
        Assemble a single product's report from its coherence scores.

    - generate_portfolio_report(product_reports: list[CoherenceReport]) -> dict:
        Aggregate across products: portfolio-level brand coherence, market coherence,
        rankings, and identification of the least/most coherent products.

    - generate_competitive_report(portfolio_reports: dict[str, list[CoherenceReport]]) -> dict:
        For Experiment 5: cross-brand comparison within a category.
        Rank brands by coherence score.

    - export_report_json(report: CoherenceReport | dict, output_path: str) -> Path:
        Export to JSON for programmatic consumption.

    - export_report_markdown(report: CoherenceReport | dict, output_path: str) -> Path:
        Export to human-readable markdown for consulting deliverable.

coherence/viz.py:
    - plot_pairwise_heatmap(pairwise_matrix: np.ndarray, channel_labels: list[str],
                            product_name: str, output_path: str) -> Path:
        Heatmap of channel-pair similarities for a single product.

    - plot_coherence_comparison(product_reports: list[CoherenceReport],
                                output_path: str) -> Path:
        Bar chart comparing brand_coherence and market_coherence across products.
        Uses existing CB_PALETTE from viz.py.

    - plot_channel_radar(pairwise_matrix: np.ndarray, channel_labels: list[str],
                         product_name: str, output_path: str) -> Path:
        Radar/spider chart showing each channel's mean similarity to all others.

    - plot_temporal_drift(time_points: list[str], coherence_scores: list[float],
                          product_name: str, output_path: str) -> Path:
        Line plot for Experiment 4: coherence over time.
```

**Tests:**
- `test_generate_product_report_fields`: all required fields present
- `test_portfolio_report_rankings`: products sorted correctly by coherence
- `test_export_json_roundtrip`: export and re-import produces same data
- `test_two_tier_scores_differ`: brand and market scores differ when review diverges

**Acceptance Criteria:**
- Reports contain both brand_coherence and market_coherence tiers
- Z-scores are computed relative to calibration distribution
- Outlier channel is identified correctly
- Both JSON and markdown exports are valid
- Visualization functions produce valid PNG files

**Dependencies:** Step 3 (metric computation), Step 5 (calibration statistics for z-scores)

---

### Step 7: Experiment 1 — Real-Document Sensitivity

**Complexity:** High
**Files:**
- Create: `coherence/experiment_1.py`
- Create: `data/real_documents/` directory structure
- Create: `data/real_documents/product_manifest.json` (schema definition)
- Modify: `run.py` (add `coherence-exp1` subcommand)

**Description:**
The core validation experiment. Select 20 CPG products (10 "known consistent" + 10 "known inconsistent" per Decision 8), collect 3-5 real documents per product from different channels, and test whether the coherence metric separates the two groups. Run TF-IDF and BERTScore baselines alongside (Decision 5).

**Details:**
```
coherence/experiment_1.py:
    - PRODUCT_SELECTION_CRITERIA:
        Document the operational definition of "known consistent" and "known inconsistent"
        as constants/docstring BEFORE product selection. Criteria:
        - Consistent: large brand, strict regulatory alignment, unified agency,
          documented brand guidelines publicly available
        - Inconsistent: startup/challenger brand, multiple agencies, known rebranding,
          social media voice disconnected from packaging

    - Exp1Config dataclass:
        n_consistent: int = 10
        n_inconsistent: int = 10
        min_channels_per_product: int = 3
        channels: list[str] = ["regulatory", "marketing", "retail", "social", "review"]
        alpha: float = 0.05
        metric_formula: dict  (locked from Experiment 0)
        run_baselines: bool = True

    - run_experiment_1(config: Exp1Config, documents_path: str,
                       contrastive_model_path: str) -> dict:
        1. Load and validate real documents
        2. Clean and chunk documents (via ingest module)
        3. Extract embeddings using contrastive fine-tuned model
        4. Compute coherence scores (brand + market tier) for all 20 products
        5. Statistical test: Mann-Whitney U between consistent and inconsistent groups
           (non-parametric, appropriate for n=10 per group)
        6. Compute effect size: Cohen's d and rank-biserial correlation
        7. Run baselines: TF-IDF coherence, BERTScore coherence on same documents
        8. Compare hidden-state separation to baseline separation
        9. Generate classification report: can the metric classify each product
           into the correct group?

        Returns results dict with all scores, statistical tests, and baseline comparisons.

    - compute_separation(consistent_scores: np.ndarray,
                         inconsistent_scores: np.ndarray) -> dict:
        Mann-Whitney U test, Cohen's d, rank-biserial correlation, ROC AUC.
        Report number of misclassifications at optimal threshold.

    - generate_exp1_report(results: dict, output_dir: str) -> Path:
        Structured report with pass/fail assessment.
        Pass criterion: AUC >= 0.85, or at most 2 misclassifications out of 20.
        (Relaxed from original "no overlap" given larger n and statistical framing.)
```

**Tests:**
- `test_compute_separation_perfect`: perfect separation yields AUC = 1.0
- `test_compute_separation_random`: random scores yield AUC near 0.5
- `test_run_experiment_1_structure`: results dict has all expected keys (mock data)

**Acceptance Criteria:**
- The coherence metric achieves AUC >= 0.85 on the consistent vs. inconsistent classification
- At most 2 misclassifications out of 20 products
- The hidden-state method outperforms at least one baseline (TF-IDF or BERTScore) by AUC >= 0.05
- Statistical test (Mann-Whitney U) achieves p < 0.05
- Full results saved with all intermediate data for reproducibility

**Dependencies:** Step 1 (document ingestion), Step 2 (contrastive model), Step 3 (metrics), Step 4 (baselines), Step 5 (locked metric formula)

---

### Step 8: Experiment 2 — Channel Attribution

**Complexity:** Low
**Files:**
- Create: `coherence/experiment_2.py`
- Modify: `run.py` (add `coherence-exp2` subcommand)

**Description:**
Reanalysis of Experiment 1 data: for the inconsistent products, does the pairwise coherence matrix correctly identify which channel is the outlier?

**Details:**
```
coherence/experiment_2.py:
    - run_experiment_2(exp1_results: dict, ground_truth_outliers: dict) -> dict:
        1. Load pairwise matrices from Experiment 1 for inconsistent products
        2. For each inconsistent product, identify the outlier channel
           (channel with lowest mean similarity to all others)
        3. Compare against ground truth (human-judged outlier channels)
        4. Score: fraction of products where metric identifies the correct outlier

        ground_truth_outliers: dict mapping product_id -> expected outlier channel
        (provided as human annotation before running the experiment)

    - analyze_pairwise_patterns(pairwise_matrix: np.ndarray,
                                 channel_labels: list[str]) -> dict:
        Detailed breakdown: which channel pairs are most/least similar,
        whether the outlier is clear (large gap to second-lowest) or ambiguous.

    - Pass criterion: correct outlier identification for >= 6 of 10 inconsistent products.
```

**Tests:**
- `test_outlier_identification_clear`: synthetic matrix with obvious outlier is detected
- `test_outlier_identification_ambiguous`: handle case where no clear outlier exists

**Acceptance Criteria:**
- Correct outlier channel identified for >= 6 of 10 inconsistent products
- Clear gap between outlier and second-lowest channel for correctly identified cases
- Report generated with per-product pairwise analysis

**Dependencies:** Step 7 (Experiment 1 results)

---

### Step 9: Experiment 3 — Attribute-Level Drill-Down

**Complexity:** High
**Files:**
- Create: `coherence/experiment_3.py`
- Create: `coherence/probe_stimuli.py`

**Description:**
Test whether the methodology can identify *what content* is getting lost in translation. Create attribute-probe stimuli, compute similarity between probes and channel documents, check against known attribute-level gaps.

**Details:**
```
coherence/probe_stimuli.py:
    - generate_attribute_probes(product: dict, attributes: list[str],
                                 context_level: str = "paragraph") -> list[dict]:
        For each attribute, generate a probe stimulus: a short paragraph that
        mentions the product AND the specific attribute in context.
        context_level: "sentence" (1 sentence), "paragraph" (3-4 sentences)
        Test both formulations as the critique noted decontextualized probes may fail.

    - PROBE_TEMPLATE: f-string template for probe generation

coherence/experiment_3.py:
    - Exp3Config dataclass:
        n_products: int = 3
        attributes_per_product: int = 3-5
        context_levels: list[str] = ["sentence", "paragraph"]

    - run_experiment_3(config: Exp3Config, documents_path: str,
                       ground_truth_attributes: dict,
                       contrastive_model_path: str) -> dict:
        1. For 3 selected products with known attribute-level gaps:
            a. Generate probe stimuli for each key attribute
            b. Extract embeddings for probes and channel documents
            c. Compute attribute-channel similarity matrix
            d. Compare against ground truth (which attributes present/absent per channel)
        2. Score: fraction of attribute-channel pairs correctly classified

        Pass criterion: >= 2 of 3 products correctly characterized.
```

**Tests:**
- `test_generate_attribute_probes`: correct number of probes generated with product context
- `test_attribute_similarity_matrix_shape`: (n_attributes, n_channels) output

**Acceptance Criteria:**
- Attribute probes include product context (not just isolated attribute mentions)
- Attribute-channel similarity matrix correctly identifies >= 70% of present/absent pairs for at least 2 of 3 products
- Both "sentence" and "paragraph" context levels tested; report which performs better

**Dependencies:** Step 2 (contrastive model), Step 3 (metrics), Step 7 (real documents from Experiment 1)

---

### Step 10: Experiment 4 — Temporal Coherence Drift

**Complexity:** Medium
**Files:**
- Create: `coherence/experiment_4.py`

**Description:**
Test whether the metric detects messaging drift over time by analyzing historical documents at multiple time points.

**Details:**
```
coherence/experiment_4.py:
    - Exp4Config dataclass:
        n_products: int = 3
        min_time_points: int = 3
        control_products: int = 1  (same-category brand with stable messaging)

    - run_experiment_4(config: Exp4Config, temporal_documents_path: str,
                       brand_events: dict, contrastive_model_path: str) -> dict:
        1. For each product, compute coherence at each time point
        2. Compute coherence delta between time points
        3. Compare trajectories against known brand events
        4. Subtract control product's trajectory to correct for secular language trends
           (addresses critique risk #7 about web language evolution)
        5. Score: correlation between coherence changes and brand event timing

        brand_events: dict mapping product_id -> list of (date, event_description)

        Pass criterion: coherence trajectory is non-random (Kendall tau test)
        and directionally correct for >= 2 of 3 products after control correction.
```

**Tests:**
- `test_temporal_trajectory_monotone`: synthetic declining coherence detected
- `test_control_correction`: secular trend subtraction works correctly

**Acceptance Criteria:**
- Coherence trajectories show interpretable patterns for >= 2 of 3 products
- Control-corrected trajectories differ from raw trajectories
- Brand management events correspond to coherence changes

**Dependencies:** Step 2 (contrastive model), Step 3 (metrics)

---

### Step 11: Experiment 5 — Competitive Coherence Benchmarking

**Complexity:** Medium
**Files:**
- Create: `coherence/experiment_5.py`

**Description:**
Test whether the metric produces competitive rankings that industry professionals find credible. Stronger design than original: experts predict rankings before seeing results (forced-choice design, addressing critique risk #8).

**Details:**
```
coherence/experiment_5.py:
    - Exp5Config dataclass:
        category: str  (e.g., "toothpaste" or "protein_bars")
        n_brands: int = 5
        n_experts: int = 3
        channels_per_brand: list[str] = ["regulatory", "marketing", "retail", "social", "review"]

    - run_experiment_5(config: Exp5Config, category_documents_path: str,
                       contrastive_model_path: str) -> dict:
        1. Compute coherence scores for all brands in category
        2. Rank brands by brand_coherence and market_coherence
        3. Generate ranking for expert evaluation

        Returns metric rankings (not expert evaluations, which are manual).

    - evaluate_expert_agreement(metric_ranking: list[str],
                                 expert_rankings: list[list[str]]) -> dict:
        1. Compute Kendall tau between metric ranking and each expert ranking
        2. Compute inter-expert Kendall tau (baseline for agreement)
        3. Compute mean overlap: fraction of pairwise brand comparisons where
           metric agrees with majority expert opinion

        Pass criterion: metric-expert Kendall tau >= inter-expert Kendall tau,
        AND mean overlap >= 0.7 (at least 7 of 10 pairwise comparisons correct).

    - generate_expert_evaluation_form(brands: list[str], output_path: str) -> Path:
        Generate a structured form for experts to provide their rankings
        BEFORE seeing the metric results (pre-registered design).
```

**Tests:**
- `test_evaluate_expert_agreement_perfect`: perfect agreement yields tau = 1.0
- `test_evaluate_expert_agreement_random`: random rankings yield tau near 0.0

**Acceptance Criteria:**
- Competitive rankings generated for all brands in the category
- Expert evaluation form generated with clear instructions
- Agreement metrics computed correctly
- Metric-expert agreement >= inter-expert agreement

**Dependencies:** Step 2 (contrastive model), Step 3 (metrics), Step 6 (reporting)

---

### Step 12: CLI Integration and Pipeline Orchestration

**Complexity:** Low
**Files:**
- Modify: `run.py` (add coherence subcommands)
- Create: `coherence/pipeline.py`

**Description:**
Wire all experiment modules into the existing CLI, following the `run.py` pattern of subcommands with `cmd_*` functions.

**Details:**
```
run.py additions:
    New subcommands:
    - `python run.py coherence-exp0` — Run Experiment 0 (metric exploration)
    - `python run.py coherence-exp1` — Run Experiment 1 (real-document sensitivity)
    - `python run.py coherence-exp2` — Run Experiment 2 (channel attribution)
    - `python run.py coherence-exp3` — Run Experiment 3 (attribute drill-down)
    - `python run.py coherence-exp4` — Run Experiment 4 (temporal drift)
    - `python run.py coherence-exp5` — Run Experiment 5 (competitive benchmarking)
    - `python run.py coherence-finetune` — Run contrastive fine-tuning
    - `python run.py coherence-all` — Run full coherence validation pipeline

    Each subcommand follows existing pattern:
    1. set_global_seed()
    2. import module
    3. Load config/data
    4. Run experiment
    5. Save results to data/coherence/
    6. Print summary

coherence/pipeline.py:
    - run_coherence_pipeline(config: dict, start_from: str = "exp0") -> dict:
        Orchestrate the sequential experiment execution with gating:
        - exp0 always runs first
        - exp1 gates on exp0 (metric must be locked)
        - exp2 runs on exp1 data (no gate — reanalysis)
        - exp3 gates on exp1 pass
        - exp4 can run independently after exp1
        - exp5 gates on exp1 pass

        At each gate, print pass/fail and ask whether to continue.
```

**Tests:**
- `test_subcommand_registration`: all new subcommands appear in argparse
- `test_pipeline_gating`: pipeline stops correctly when experiment fails

**Acceptance Criteria:**
- All subcommands are accessible from `python run.py --help`
- Pipeline respects experiment gating (no Experiment 3 without Experiment 1 pass)
- All results saved under `data/coherence/` in a structured directory

**Dependencies:** Steps 5-11 (all experiment modules)

---

## Dependency Graph

```
Step 1 (Ingest) ─────────────────────────────┐
                                               │
Step 2 (Contrastive) ─────────────┐           │
                                   │           │
Step 3 (Metrics) ◄────────────────┤           │
                                   │           │
Step 4 (Baselines) ───────────────│───────────┤
                                   │           │
Step 5 (Exp 0) ◄──────────────────┤◄──────────┘ (needs Steps 2, 3)
                                   │
Step 6 (Reporting) ◄──────────────┤◄── Step 3, Step 5
                                   │
Step 7 (Exp 1) ◄──────────────────┤◄── Steps 1, 2, 3, 4, 5
                                   │
Step 8 (Exp 2) ◄───────────────────── Step 7
                                   │
Step 9 (Exp 3) ◄──────────────────┤◄── Steps 2, 3, 7
                                   │
Step 10 (Exp 4) ◄─────────────────┤◄── Steps 2, 3
                                   │
Step 11 (Exp 5) ◄─────────────────┤◄── Steps 2, 3, 6
                                   │
Step 12 (CLI) ◄────────────────────── Steps 5-11
```

**Parallelizable groups:**
- Group A (no dependencies): Steps 1, 2, 4 — can all proceed in parallel
- Group B (needs Group A): Steps 3, 5 — can proceed once Steps 2+3 are done
- Group C (needs Group B): Steps 6, 7 — can proceed once Step 5 locks the metric
- Group D (needs Group C): Steps 8, 9, 10, 11 — can proceed after Experiment 1 data exists
- Group E (integration): Step 12 — after all modules exist

---

## External Dependencies

| Package | Version | Purpose | Step |
|---------|---------|---------|------|
| `peft` | >= 0.10 | LoRA fine-tuning for contrastive model | 2 |
| `sentence-transformers` | >= 2.6 | BERTScore baseline embeddings | 4 |
| `beautifulsoup4` | >= 4.12 | HTML cleaning in document ingestion | 1 |
| `trafilatura` | >= 1.8 | Web content extraction for document ingestion | 1 |

All other dependencies (torch, transformers, scipy, scikit-learn, h5py, numpy, matplotlib) are already in `pyproject.toml`.

---

## Implementation Risks

### Risk 1: Contrastive fine-tuning fails to separate register from product signal
**Likelihood:** Medium | **Impact:** High
**Mitigation:** Experiment 0 tests multiple correction methods (raw, mean-centered, whitened, contrastive). If contrastive fails, fall back to partial RSA controlling for register (already implemented in `analysis.partial_rsa()`). The pipeline supports both paths.

### Risk 2: Coherence metric does not generalize from calibration to real documents
**Likelihood:** Medium | **Impact:** High
**Mitigation:** Experiment 0 explicitly simulates real-document conditions (length variation, attribute removal) on calibration data before collecting real documents. If the metric breaks under simulation, we know before investing in document collection.

### Risk 3: Document collection is more time-consuming than estimated
**Likelihood:** High | **Impact:** Medium
**Mitigation:** Start with n=5 per group (10 products total) as a quick check before scaling to n=10 per group (20 products). The pipeline supports variable group sizes.

### Risk 4: LoRA fine-tuning requires more VRAM than available (32GB on RTX 5090)
**Likelihood:** Low | **Impact:** Medium
**Mitigation:** Use Qwen2.5-7B (not 32B) as the fine-tuning base. With LoRA rank=16 and 4-bit quantization, 7B model fine-tuning fits comfortably in 32GB VRAM. If needed, reduce batch size to 4 or use gradient accumulation.

### Risk 5: Baseline methods achieve comparable separation to hidden-state method
**Likelihood:** Medium | **Impact:** High
**Mitigation:** This is a valid experimental outcome, not a failure of implementation. If TF-IDF or BERTScore match hidden-state performance, the methodology pivot to simpler approaches is the correct scientific conclusion. The pipeline reports this comparison transparently.

### Risk 6: Model obsolescence (Qwen2.5 superseded)
**Likelihood:** Medium over 6-12 months | **Impact:** Medium
**Mitigation:** The contrastive fine-tuning approach is model-agnostic — the same loss function works with any transformer. The pipeline parameterizes the base model in `ContrastiveConfig`. Switching models requires only rerunning fine-tuning and Experiment 0, not restructuring the pipeline.

---

## File Structure Summary

```
langprod/
├── coherence/
│   ├── __init__.py
│   ├── ingest.py          (Step 1)
│   ├── contrastive.py     (Step 2)
│   ├── losses.py          (Step 2)
│   ├── metrics.py         (Step 3)
│   ├── baselines.py       (Step 4)
│   ├── experiment_0.py    (Step 5)
│   ├── experiment_1.py    (Step 7)
│   ├── experiment_2.py    (Step 8)
│   ├── experiment_3.py    (Step 9)
│   ├── probe_stimuli.py   (Step 9)
│   ├── experiment_4.py    (Step 10)
│   ├── experiment_5.py    (Step 11)
│   ├── reporting.py       (Step 6)
│   ├── viz.py             (Step 6)
│   └── pipeline.py        (Step 12)
├── tests/
│   ├── test_ingest.py     (Step 1)
│   ├── test_coherence_metrics.py  (Step 3)
│   ├── test_baselines.py  (Step 4)
│   └── test_reporting.py  (Step 6)
├── data/
│   ├── coherence/         (all coherence experiment outputs)
│   ├── contrastive_model/ (fine-tuned LoRA weights)
│   └── real_documents/    (real document collection)
│       └── product_manifest.json
├── run.py                 (modified: new subcommands)
└── pyproject.toml         (modified: new dependencies)
```

---

## Estimated Total Effort

| Step | Effort | Cumulative |
|------|--------|------------|
| 1. Ingest | 4-6 hours | 4-6h |
| 2. Contrastive | 8-12 hours | 12-18h |
| 3. Metrics | 3-4 hours | 15-22h |
| 4. Baselines | 2-3 hours | 17-25h |
| 5. Experiment 0 | 6-8 hours | 23-33h |
| 6. Reporting | 4-6 hours | 27-39h |
| 7. Experiment 1 | 8-12 hours (incl. document collection) | 35-51h |
| 8. Experiment 2 | 2-3 hours | 37-54h |
| 9. Experiment 3 | 6-8 hours | 43-62h |
| 10. Experiment 4 | 4-6 hours (incl. historical collection) | 47-68h |
| 11. Experiment 5 | 4-6 hours (excl. expert scheduling) | 51-74h |
| 12. CLI Integration | 2-3 hours | 53-77h |

**Total: approximately 53-77 hours of implementation work.**

The critical path runs through Steps 2 -> 5 -> 7 (contrastive model -> metric lock -> Experiment 1). Steps 1 and 4 can be done in parallel with Step 2 to optimize the timeline.
