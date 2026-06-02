# Architecture Analysis: Brand Message Coherence Validation

## 1. System Overview

The validation plan introduces six experiments (Exp 0 through Exp 5) on top of an existing single-experiment research pipeline. The current system follows a linear five-stage pattern: **generate -> extract -> analyze -> probe -> report**, orchestrated by `run.py` with a global `CONFIG` dict, reading/writing artifacts to `data/`. The new work extends this pipeline in three directions:

1. **Contrastive fine-tuning** of a smaller embedding model to suppress register and amplify product identity -- a new model-training component that sits *upstream* of extraction.
2. **Real-document ingestion** -- a new data source that bypasses the existing LLM-generated stimuli pipeline entirely.
3. **Multi-experiment state management** -- the current system runs one experiment; the new plan requires coordinating six experiments with shared artifacts, sequential gating, and two distinct reporting tiers.

The following diagram shows the high-level component boundaries:

```
                    ┌──────────────────────────┐
                    │   Contrastive Fine-Tuner  │ (NEW)
                    │   - Training data from    │
                    │     calibration set        │
                    │   - Produces: coherence    │
                    │     embedding model        │
                    └───────────┬────────────────┘
                                │ model checkpoint
                                v
┌──────────┐    ┌───────────────────────────┐    ┌─────────────────┐
│ Stimuli  │───>│    Extraction Pipeline    │───>│    Analysis      │
│ (existing│    │   (modified: supports     │    │   (extended:     │
│  + new   │    │    contrastive model OR   │    │    coherence     │
│  document│    │    base Qwen model)       │    │    metrics,      │
│  ingest) │    │                           │    │    two-tier      │
└──────────┘    └───────────────────────────┘    │    reporting)    │
                                                  └─────────────────┘
```

### What Changes vs. What's New

| Component | Status | Rationale |
|-----------|--------|-----------|
| `stimuli.py` | **Extend** | Add document ingestion adapter alongside existing LLM generation |
| `extraction.py` | **Extend** | Support contrastive model + base model extraction; handle variable-length docs |
| `analysis.py` | **Extend** | Add coherence metric computation, pairwise channel analysis, register-corrected RSA |
| `probes.py` | **No change** for Exp 0; minor extension for attribute probes in Exp 3 |
| `viz.py` | **Extend** | Coherence scorecards, channel heatmaps, temporal drift plots |
| `run.py` | **Major refactor** | Experiment-aware orchestration, config per experiment, two-tier reporting |
| `finetune.py` | **NEW** | Contrastive fine-tuning pipeline |
| `documents.py` | **NEW** | Real-document ingestion, cleaning, normalization |
| `coherence.py` | **NEW** | Coherence metric definitions, aggregation, scoring |

---

## 2. Component Design

### 2.1 Contrastive Fine-Tuner (`finetune.py`)

**Purpose:** Train a smaller encoder model (likely a BERT-class or sentence-transformer model, 300M-1B parameters) with contrastive loss that pulls same-product representations together regardless of register, creating a register-invariant embedding space.

**Design decisions:**

- **Training data source:** The existing 800-stimulus calibration dataset provides the positive/negative pairs. Each product has 10 stimuli (5 registers x 2 variants). Positive pairs = same product, any register. Negative pairs = different product.
- **Base model choice:** A sentence-transformer (e.g., `intfloat/e5-large-v2`, `BAAI/bge-large-en-v1.5`, or `sentence-transformers/all-mpnet-base-v2`) is the natural starting point. These already produce good sentence embeddings and are fast to fine-tune. Using Qwen2.5-32B as the fine-tuning target is impractical -- contrastive fine-tuning of a 32B quantized model is neither memory-feasible on a single 32GB GPU nor necessary.
- **Loss function:** `InfoNCE` (contrastive loss) or `MultipleNegativesRankingLoss` from the `sentence-transformers` library. For each anchor stimulus, the positive is another stimulus of the same product (different register), and negatives are stimuli from different products.
- **Output:** A model checkpoint in `data/models/coherence-encoder/` that can be loaded by the extraction pipeline.

**Interface with existing system:**

```python
# finetune.py
def train_contrastive_model(
    stimuli_path: str,          # path to stimuli.json (800 calibration stimuli)
    base_model: str,            # e.g., "intfloat/e5-large-v2"
    output_dir: str,            # e.g., "data/models/coherence-encoder/"
    config: dict,               # training hyperparameters
) -> Path:
    """Fine-tune a sentence encoder with contrastive loss on calibration data.
    Returns path to the saved model checkpoint."""
```

**Key architectural trade-off:** Using a smaller encoder model means abandoning the 64-layer depth analysis that characterized the original experiment. The contrastive model produces a single embedding per document, not per-layer representations. This is acceptable because the validation plan no longer needs layer-by-layer analysis -- it needs a single coherence score. However, this means Experiment 0 (metric exploration on calibration data) should run on *both* the base Qwen model (to explore layer selection) *and* the contrastive model (to validate that register suppression works), producing two candidate metric pipelines that Experiment 1 then tests head-to-head.

### 2.2 Document Ingestion (`documents.py`)

**Purpose:** Ingest, clean, and normalize real-world documents for coherence analysis, replacing the controlled stimuli generation path for Experiments 1-5.

**Design decisions:**

- **Input format:** A JSON manifest file per product portfolio, structured as:
  ```json
  {
    "product_id": "colgate_total",
    "product_name": "Colgate Total",
    "category": "oral_care",
    "tier": "brand",
    "documents": [
      {
        "channel": "regulatory",
        "source_url": "https://...",
        "collected_at": "2026-03-28",
        "raw_text": "...",
        "word_count": 342
      }
    ]
  }
  ```
- **Channel taxonomy:** Fixed vocabulary: `regulatory`, `marketing`, `retail`, `social`, `consumer_review`. The `tier` field on the product distinguishes brand-controlled channels from consumer channels (decision #7: two-tier reporting).
- **Cleaning pipeline:** Strip boilerplate (nav elements, copyright, Amazon template chrome), normalize whitespace, detect and flag non-English content. This is a rule-based preprocessor, not an ML component.
- **Length normalization strategy:** Rather than a single approach, implement three strategies as options tested in Experiment 0:
  1. **Full document** -- use the complete text as-is.
  2. **Chunked** -- split into 100-150 word chunks, embed each, take the mean embedding.
  3. **Truncated** -- truncate to a fixed token count (matching the calibration distribution).

**Interface with existing system:**

The existing stimuli pipeline expects each item to have `stimulus_id`, `product_id`, `category`, `register`, `text`, and `token_count`. The document ingestion adapter produces items with the same schema, mapping `channel` to `register` and generating a `stimulus_id` from `{product_id}_{channel}_{source_hash}`. This allows the extraction pipeline to process real documents without modification to its core loop.

```python
# documents.py
def load_product_portfolio(manifest_path: str) -> list[dict]:
    """Load and validate a product portfolio manifest."""

def clean_document(raw_text: str, channel: str) -> str:
    """Channel-aware document cleaning."""

def portfolio_to_stimuli_format(
    portfolio: list[dict],
    length_strategy: str = "full",
) -> list[dict]:
    """Convert a portfolio of real documents to the stimuli JSON schema
    expected by extraction.py."""
```

### 2.3 Coherence Metrics (`coherence.py`)

**Purpose:** Define, compute, and aggregate coherence scores from embeddings. This is the module that Experiment 0 explores and Experiment 1 tests.

**Design decisions:**

- **Candidate metrics for Experiment 0 exploration:**
  1. **Mean pairwise cosine similarity** -- average cosine similarity across all channel-pair embeddings for a product. Simple, interpretable.
  2. **Minimum pairwise cosine similarity** -- the weakest link. Identifies the most divergent channel pair.
  3. **Centroid distance** -- compute the centroid of all channel embeddings for a product, then measure each channel's distance from the centroid. Products with tight clustering score high.
  4. **Silhouette coefficient** -- treat each product as a cluster of its channel documents. Products with high silhouette scores have coherent messaging.
  5. **Partial RSA controlling for register** -- using the existing `partial_rsa()` function from `analysis.py`, regress out a register RDM from the observed RDM before computing product-identity correlation.

- **Two-tier scoring architecture (decision #7):**
  - **Brand Coherence Score:** Computed on brand-controlled channels only (regulatory, marketing, retail, social).
  - **Market Coherence Score:** Computed on all channels including consumer reviews.
  - Reports present both scores with clear labeling.

- **Baselines (decision #5):**
  - TF-IDF cosine similarity (reuses the BoW infrastructure in `stimuli.py`)
  - BERTScore (pairwise)
  - Run alongside Experiment 1 only; drop if hidden-state method outperforms.

```python
# coherence.py
def compute_pairwise_coherence(
    embeddings: dict[str, np.ndarray],  # channel -> embedding vector
    metric: str = "cosine",
) -> dict[tuple[str, str], float]:
    """Compute pairwise coherence between all channel pairs."""

def compute_coherence_score(
    embeddings: dict[str, np.ndarray],
    method: str = "mean_pairwise",       # one of the candidate metrics
    tier: str = "brand",                  # "brand" or "market"
    brand_channels: set[str] | None = None,
) -> float:
    """Compute an aggregate coherence score for a product."""

def compute_channel_attribution(
    embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Identify outlier channels by distance from centroid. Returns
    per-channel deviation scores (higher = more divergent)."""

def compute_baselines(
    documents: dict[str, str],  # channel -> raw text
) -> dict[str, float]:
    """Compute TF-IDF and BERTScore baselines."""
```

### 2.4 Extended Extraction (`extraction.py` modifications)

**Modifications needed:**

1. **Contrastive model support:** The current extraction pipeline is built around `AutoModelForCausalLM` with forward hooks on transformer layers. The contrastive model is a sentence encoder that produces a single embedding vector per input. Add a second extraction path:

   ```python
   def extract_contrastive_embeddings(
       config: dict,
       stimuli: list[dict],
       model_path: str,
   ) -> Path:
       """Extract embeddings from the contrastive-finetuned encoder.
       Returns path to a simpler HDF5: shape (N, D) instead of (N, L+1, D)."""
   ```

2. **Variable-length handling:** The current pipeline tokenizes each stimulus and processes it individually (batch_size=1), which already handles variable lengths. The key issue is the context window: Qwen2.5-32B supports 32K tokens, so even long regulatory filings fit. For the contrastive encoder (typically 512-token context), implement the chunking strategy from `documents.py`.

3. **Dual extraction mode:** For Experiment 0, need to extract embeddings from *both* the base Qwen model (at selected layers) and the contrastive model. The existing `extract_hidden_states()` function returns an HDF5 with shape `(N, L+1, D)`. The contrastive path returns `(N, D)`. Analysis code must handle both.

### 2.5 Orchestration (`run.py` refactor)

The current `run.py` runs a single pipeline with a global CONFIG. The new plan requires experiment-aware orchestration.

**Design approach:** Add experiment-specific subcommands rather than replacing the existing structure.

```
python run.py generate          # existing: generate stimuli
python run.py extract           # existing: extract from base model
python run.py analyze           # existing: full RSA analysis

python run.py finetune          # NEW: train contrastive model
python run.py exp0              # NEW: metric exploration on calibration data
python run.py exp1              # NEW: real-document sensitivity
python run.py exp2              # NEW: channel attribution (reanalyzes exp1 data)
python run.py exp3              # NEW: attribute-level drill-down
python run.py exp4              # NEW: temporal coherence drift
python run.py exp5              # NEW: competitive benchmarking
python run.py coherence-report  # NEW: two-tier coherence report
```

**Experiment config structure:** Each experiment gets its own config section within CONFIG or a separate config file:

```python
COHERENCE_CONFIG = {
    # Shared
    "coherence_model_path": "data/models/coherence-encoder/",
    "brand_channels": {"regulatory", "marketing", "retail", "social"},
    "consumer_channels": {"consumer_review"},
    "length_strategies": ["full", "chunked", "truncated"],
    "candidate_metrics": ["mean_pairwise", "min_pairwise", "centroid_distance", "silhouette"],

    # Exp 0
    "exp0_calibration_data": "data/stimuli.json",

    # Exp 1
    "exp1_n_per_group": 10,
    "exp1_portfolio_dir": "data/portfolios/exp1/",
    "exp1_pass_criterion": "at_most_2_misclassifications",  # updated for n=20

    # Exp 2 (reuses exp1 data)

    # Exp 3
    "exp3_n_products": 3,
    "exp3_probe_dir": "data/probes/exp3/",

    # Exp 4
    "exp4_n_products": 3,
    "exp4_time_points": 3,

    # Exp 5
    "exp5_category": "toothpaste",
    "exp5_n_brands": 5,
}
```

---

## 3. Data Flow

### 3.1 Experiment 0: Metric Exploration (Calibration Pilot)

```
data/stimuli.json (800 existing stimuli)
    │
    ├──> finetune.py ──> data/models/coherence-encoder/
    │                          │
    │                          v
    │                   extraction.py (contrastive path)
    │                          │
    │                          v
    │                   data/exp0/contrastive_embeddings.h5  (800, D)
    │
    └──> data/Qwen_...hidden_states.h5 (already exists, 800 x 65 x 5120)
                │
                v
         coherence.py (candidate metrics x length strategies x layer choices)
                │
                v
         data/exp0/metric_exploration_results.json
         data/exp0/metric_selection.json  <-- pre-registered choice for Exp 1
```

**Key decision point:** Experiment 0 is purely exploratory. Its output is a *locked metric definition* -- which model (contrastive or base+layer), which aggregation method, which length strategy. This definition is frozen before Experiment 1 data is collected.

### 3.2 Experiments 1-5: Real-Document Pipeline

```
data/portfolios/exp{N}/            <-- raw product portfolios (JSON manifests)
    │
    v
documents.py (clean, normalize)
    │
    v
data/exp{N}/stimuli_adapted.json   <-- documents in stimuli schema format
    │
    v
extraction.py (contrastive model)  <-- uses locked model from Exp 0
    │
    v
data/exp{N}/embeddings.h5          <-- (N_docs, D)
    │
    v
coherence.py (locked metric)       <-- uses locked formula from Exp 0
    │
    v
data/exp{N}/coherence_scores.json
data/exp{N}/channel_attribution.json  (Exp 2+)
data/exp{N}/attribute_analysis.json   (Exp 3)
data/exp{N}/temporal_drift.json       (Exp 4)
data/exp{N}/competitive_ranking.json  (Exp 5)
    │
    v
viz.py (experiment-specific figures)
    │
    v
data/exp{N}/report.md
```

### 3.3 Two-Tier Reporting Data Flow

For each product in Experiments 1-5:

```
All channel embeddings
    │
    ├──> Filter: brand_channels only ──> Brand Coherence Score
    │
    └──> All channels ──> Market Coherence Score

Report includes:
    - Brand coherence section (primary)
    - Market coherence section (separate)
    - Comparison: does consumer perception track brand intent?
```

### 3.4 Artifact Dependency Graph

```
Exp 0 depends on: calibration data (exists), contrastive model (trained in Exp 0)
Exp 1 depends on: locked metric from Exp 0, real document collection
Exp 2 depends on: Exp 1 embeddings and scores (reanalysis, no new data)
Exp 3 depends on: attribute probe design + Exp 1 or new data
Exp 4 depends on: historical document collection (independent of Exp 1-3 data)
Exp 5 depends on: competitive document collection (independent of Exp 1-4 data)
```

The sequential gating (don't proceed to Exp N+1 if Exp N fails) is a *policy decision*, not a technical dependency. Experiments 3, 4, and 5 are technically independent in terms of data flow -- they could run in parallel once the metric is locked.

---

## 4. Technology Choices

### 4.1 Contrastive Fine-Tuning Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | `intfloat/e5-large-v2` (335M params) or `BAAI/bge-large-en-v1.5` (335M) | State-of-art sentence embeddings; fits easily on RTX 5090; well-supported by sentence-transformers |
| Training library | `sentence-transformers` with `MultipleNegativesRankingLoss` | Purpose-built for this exact use case; handles positive/negative pair construction automatically |
| Training compute | Single RTX 5090, ~30 minutes on 800 stimuli | Trivial compute requirements |
| Evaluation | Mean pairwise cosine similarity for same-product-different-register pairs vs. different-product pairs; should show clear separation post-training |

**Alternative considered:** Fine-tuning Qwen2.5 itself with LoRA adapters. Rejected because (a) 32B model on a single 32GB GPU leaves minimal room for gradient computation, (b) we don't need generative capabilities -- only embeddings, (c) smaller encoder models are faster at inference, which matters for a consulting engagement.

### 4.2 Document Processing

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Text extraction | `trafilatura` for web pages, raw text for filings | Trafilatura handles boilerplate removal from web scrapes well |
| Cleaning | Rule-based per-channel regex + heuristics | ML-based cleaning adds unnecessary complexity; channel-specific rules are more reliable |
| Storage | JSON manifests + plain text | Keeps documents human-readable; no need for a database at this scale |

### 4.3 Baseline Metrics

| Baseline | Implementation | Notes |
|----------|---------------|-------|
| TF-IDF cosine | `sklearn.feature_extraction.text.TfidfVectorizer` | Already available via `stimuli.run_bow_baseline()` -- needs minor adaptation |
| BERTScore | `bert_score` library | Standard NLG metric; pairwise computation between channel documents |

### 4.4 Dependencies to Add

```toml
# Additions to pyproject.toml [project.dependencies]
"sentence-transformers>=3.0",   # contrastive fine-tuning
"trafilatura>=1.8",             # web document extraction
"bert-score>=0.3.13",           # BERTScore baseline
```

---

## 5. Integration with Existing System

### 5.1 Components That Remain Unchanged

- **`stimuli.py`** -- The product catalogs (`REAL_PRODUCTS`, `FICTIONAL_PRODUCTS`), register specs, and LLM generation code are untouched. The calibration dataset is a *consumed artifact* of the new plan, not a modified one.
- **`probes.py`** -- Linear probing is not part of the coherence validation (except potentially in Experiment 3 for attribute-level analysis). The existing probing infrastructure remains as-is.
- **Test suite** -- Existing tests (`test_rdm.py`, `test_rsa.py`, `test_anisotropy.py`, `test_pooling.py`, `test_groupkfold.py`) remain valid and passing. New tests are added alongside them.

### 5.2 Components That Need Modification

**`extraction.py`:**
- Add `extract_contrastive_embeddings()` function that loads a `SentenceTransformer` model and calls `model.encode()`.
- Add a `chunked_encode()` helper for documents exceeding the model's context window.
- The existing `extract_hidden_states()` function is unchanged -- it continues to work for Experiment 0's base-model analysis.
- HDF5 schema for contrastive embeddings: simpler than the existing schema. Shape `(N, D)` instead of `(N, L+1, D)`, plus `stimulus_ids`.

**`analysis.py`:**
- Add a `compute_coherence_rdm()` function that builds a product-identity RDM from document-level embeddings (simpler than the current layer-by-layer RDM computation).
- The existing `partial_rsa()` function is directly reusable for register-controlled analysis in Experiment 0.
- The existing `build_register_model_rdm()` and `build_product_model_rdm()` are reusable with the new document metadata.

**`viz.py`:**
- Add coherence-specific plot functions: coherence scorecard (bar chart per product), channel attribution heatmap, temporal drift line chart, competitive ranking.
- Existing plot functions remain for the Protocol Layer Hypothesis report.

**`run.py`:**
- Add new subcommands (`finetune`, `exp0`, `exp1`, ..., `exp5`, `coherence-report`).
- Factor out the main argument parser to accommodate new commands.
- Add `COHERENCE_CONFIG` alongside existing `CONFIG`.
- Consider: should experiment orchestration live in `run.py` or a new `run_coherence.py`?

### 5.3 Recommended Architecture: Separate Orchestrator

Given the scope of the new experiment sequence, I recommend a **separate orchestrator** (`run_coherence.py`) rather than extending `run.py`:

- `run.py` remains the orchestrator for the Protocol Layer Hypothesis experiment. It works, its data is in `data/`, and it should not be modified.
- `run_coherence.py` orchestrates the coherence validation experiments. It has its own `COHERENCE_CONFIG`, its own subcommands, and writes to `data/coherence/`.
- Both orchestrators import from the same shared modules (`extraction.py`, `analysis.py`, `viz.py`) plus new modules (`finetune.py`, `documents.py`, `coherence.py`).

This avoids the risk of breaking the existing pipeline while adding significant new functionality. The shared modules (`extraction.py`, `analysis.py`) are extended via new functions, not by modifying existing ones.

```
run.py              (existing, unchanged)
run_coherence.py    (NEW, coherence validation orchestrator)
    │
    ├── finetune.py      (NEW)
    ├── documents.py     (NEW)
    ├── coherence.py     (NEW)
    ├── extraction.py    (extended)
    ├── analysis.py      (extended)
    └── viz.py           (extended)
```

---

## 6. Architectural Trade-offs

### 6.1 Separate Encoder vs. Base Qwen Model

**Trade-off:** Using a contrastive-finetuned smaller encoder means abandoning the Qwen2.5-32B layer-level analysis that was the core of the original experiment.

- **For separate encoder:** Directly addresses register dominance (the central technical risk). Faster inference (~100x). Easier to deploy in a consulting setting. Model is small enough to version and share.
- **Against separate encoder:** Loses the rich layer-by-layer representational analysis. The coherence metric becomes a single number from a single model, not a multi-layer profile. If the contrastive fine-tuning fails to suppress register, there is no fallback to "pick a different layer."
- **Mitigation:** Experiment 0 tests both approaches side by side. If the contrastive model underperforms base Qwen at a selected layer (after register correction via partial RSA), use the base model. The architecture supports both paths.

### 6.2 Pre-registered Metric vs. Flexible Exploration

**Trade-off:** Decision #4 adds Experiment 0 for metric exploration before committing to a formula. This is methodologically correct but creates a design challenge: the code must support *multiple* metric candidates during Experiment 0, then freeze to *one* for Experiments 1-5.

- **Implementation:** Experiment 0 writes a `metric_selection.json` file that specifies the locked metric. All subsequent experiments read this file and refuse to run if it does not exist. This creates a hard gate between exploration and confirmation.
- **Risk:** If Experiment 0 produces ambiguous results (no single metric clearly dominates), the researcher must make a judgment call about which metric to lock. This is a human decision point, not an automatable one.

### 6.3 Document-Level vs. Passage-Level Coherence

**Trade-off:** The architecture assumes one embedding per document. But real documents may contain multiple topics, only some of which relate to the target product's core message.

- **Document-level approach:** Simpler, faster, matches the original experiment's design. Works well when documents are primarily about one product (regulatory filings, product pages).
- **Passage-level approach:** Split documents into passages, embed each, and compute coherence on the most relevant passages. More accurate for long, multi-topic documents (e.g., a blog post that mentions the product in one paragraph).
- **Architecture decision:** Start with document-level (full and chunked strategies in Experiment 0). If Experiment 1 fails and the failure mode suggests passage-level is needed, add it as an iteration. The `documents.py` module already supports chunking, which is the foundation for passage-level analysis.

### 6.4 Separate Orchestrator vs. Extended `run.py`

**Trade-off:** A separate `run_coherence.py` avoids touching the working experiment pipeline but introduces code duplication (argument parsing, config management, path helpers).

- **For separate orchestrator:** Zero risk of regression to the PLH experiment. Cleaner separation of concerns. The two experiments have different lifecycles.
- **Against separate orchestrator:** Duplicated boilerplate. Shared modules (`extraction.py`, `analysis.py`) may diverge if modified in incompatible ways.
- **Mitigation:** Extract shared utilities (path helpers, config loading, seed management) into a small `utils.py` module imported by both orchestrators.

### 6.5 HDF5 vs. Simpler Storage for Contrastive Embeddings

**Trade-off:** The existing pipeline uses HDF5 for the (800, 65, 5120) tensor -- justified by its size (~3.6 GB). Contrastive embeddings are much smaller: (800, 1024) is ~3.3 MB.

- **Decision:** Use HDF5 for consistency with existing tooling, but keep the schema simple. The analysis code already knows how to read HDF5 files. Alternative: save as `.npy` for simplicity. Either works at this scale.

---

## 7. Open Architectural Questions

### 7.1 Contrastive Training Data Sufficiency

The 800-stimulus calibration set has 80 products x 10 stimuli. For contrastive learning, each product contributes $\binom{10}{2} = 45$ positive pairs and many more negative pairs. Is this sufficient for a 335M-parameter encoder?

- **Assessment:** Likely yes for fine-tuning (not training from scratch). The base model already produces good embeddings; we are adjusting a well-initialized embedding space, not learning from random weights. However, this should be validated in Experiment 0 by checking the train/validation split performance.
- **Fallback:** If 800 stimuli are insufficient, augment by generating additional stimuli for the same 80 products (more register variants, more paraphrase variants). The generation pipeline already supports this.

### 7.2 Context Window Constraints for Real Documents

The contrastive encoder model (e.g., E5-large) has a 512-token context window. Real regulatory filings can exceed 5000 words (~7000 tokens). The chunking strategy (embed 100-150 word chunks, then average) loses document-level structure.

- **Alternative:** Use a long-context encoder (e.g., `jinaai/jina-embeddings-v3` supports 8192 tokens) as the base model for contrastive fine-tuning. This handles most documents without chunking.
- **Decision needed before Experiment 0:** Which base encoder model to use, balancing context window against fine-tuning ease and embedding quality.

### 7.3 Metric Interpretability

The coherence score needs to be interpretable for consulting clients. Raw cosine similarity values (0.0 to 1.0) are not inherently meaningful to a brand manager.

- **Approach:** Use the calibration dataset to establish a reference distribution. "Your coherence score is at the 73rd percentile relative to the CPG calibration set" is more meaningful than "your coherence score is 0.73." This requires computing the coherence score for all 80 calibration products and building a reference distribution in Experiment 0.
- **Risk:** The calibration set uses LLM-generated text. The reference distribution may not transfer well to real documents. Experiment 1's 20 real products become the de facto reference distribution if the calibration baseline proves inadequate.

### 7.4 Experiment Gating Mechanism

How exactly does "Experiment N must pass before proceeding to Experiment N+1" get enforced in the system?

- **Proposed approach:** Each experiment writes a `verdict.json` file with `{"passed": true/false, "timestamp": "...", "details": "..."}`. The next experiment's orchestration checks for the predecessor's verdict file and refuses to run if it is absent or `passed: false`. Manual override with `--force` flag for debugging.
- **Edge case:** Experiment 2 reanalyzes Experiment 1 data. If Experiment 2 fails, should Experiment 3 still be blocked? Decision #10 from the critique suggests these may not be strictly sequential. The gating should be configurable per experiment.

### 7.5 Model Migration Path

The contrastive model is trained on Qwen2.5-32B-generated calibration data. If the methodology is validated and deployed commercially, what happens when the base model changes?

- **Assessment:** The contrastive encoder is independent of the Qwen model once trained. It learns register-invariant embeddings from the *text content* of the stimuli, not from Qwen's hidden states. A new base model for generation would produce slightly different calibration text, requiring retraining the contrastive model -- but this is a 30-minute operation, not a fundamental re-architecture.
- **Greater risk:** Client documents are embedded by the contrastive model, whose embedding space is calibrated against the original 800 stimuli. If the stimuli distribution shifts (new categories, new registers), the calibration becomes less relevant. Plan for periodic recalibration.

### 7.6 Attribute Probe Design (Experiment 3)

Experiment 3 requires creating "probe stimuli" that isolate individual attributes (e.g., "SPF 50 broad-spectrum protection"). These probes need to be embedded in the same space as the real documents.

- **Design question:** Should probes be (a) minimal one-sentence descriptions, (b) short paragraphs with product context, or (c) synthetically generated in each register? The choice affects what the similarity score measures.
- **Architectural implication:** The `documents.py` module needs a `create_attribute_probes()` function that generates probe stimuli from a product's attribute specification. This reuses the product catalog structure from `stimuli.py` (which already defines `core_attributes` and `distinguishing_features` per product).

### 7.7 Test Strategy for New Components

The existing test suite covers mathematical primitives (RDM computation, RSA correlation, anisotropy correction, pooling). New tests needed:

- **`test_documents.py`:** Document cleaning, schema validation, length normalization strategies.
- **`test_coherence.py`:** Coherence metric computation, two-tier scoring, channel attribution, baseline computation.
- **`test_finetune.py`:** Integration test with a tiny model (verify the training loop runs, loss decreases, embeddings change).
- **End-to-end integration test:** Generate a small calibration subset (10 products x 5 registers x 2 variants = 100 stimuli), train a contrastive model, extract embeddings, compute coherence scores, verify the full pipeline produces valid output. This test should run in under 60 seconds using a small base model.
