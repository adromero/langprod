# Open Questions: Brand Coherence Validation

Items that remain unresolved and will be addressed during execution.

---

### 1. Base encoder selection

**Context:** The contrastive fine-tuning needs a base sentence-transformer model. Candidates include `jinaai/jina-embeddings-v3` (8192-token context), `intfloat/e5-large-v2` (512 tokens), and `BAAI/bge-large-en-v1.5` (512 tokens).

**Resolution path:** Benchmark all candidates on calibration data during Step 2 (contrastive fine-tuning). Compare base embedding quality before fine-tuning, then compare post-fine-tuning register suppression. Lock the winner as part of the contrastive model checkpoint.

**When to resolve:** Phase 0, Step 2.

---

### 2. Number of manually rewritten stimuli for the non-LLM transfer test

**Context:** Experiment 0 Part B includes testing on human-rewritten stimuli to check whether the metric transfers beyond LLM-generated text. The original plan left the count unspecified.

**Recommended resolution:** 10 manually rewritten stimuli is sufficient for a directional signal. This is labor-intensive; more than 10 provides diminishing returns at this stage.

**When to resolve:** Phase 1, Step 5 (Experiment 0).

---

### 3. Minimum commercially relevant effect size

**Context:** Cohen's d >= 0.8 is the proposed threshold for Experiment 1, but what coherence score difference translates to an actionable insight for a client? This requires input from potential users of the methodology.

**Recommended resolution:** Defer to post-Experiment 1, when real score distributions are available. The validation plan uses statistical thresholds (AUC, Mann-Whitney); commercial interpretability is a post-validation concern.

**When to resolve:** After Experiment 1 results are available.
