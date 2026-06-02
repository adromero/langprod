# Risk Assessment: Brand Message Coherence Validation

## Risk Matrix

| ID | Risk | Likelihood | Impact | Severity | Mitigation |
|----|------|-----------|--------|----------|------------|
| F1 | Contrastive fine-tuning fails to separate register from product signal | Medium | Critical | **HIGH** | Staged evaluation with fallback to partial RSA |
| F2 | Real documents break extraction pipeline (length, encoding, formatting) | High | High | **HIGH** | Length-adaptive preprocessing with chunking strategy |
| F3 | Calibration data does not transfer to real-world documents | Medium | Critical | **HIGH** | Experiment 0 degradation testing; accept/reject gate |
| F4 | Coherence metric is insensitive, noisy, or confound-dominated | Medium | Critical | **HIGH** | Pre-registered metric + baselines + effect size thresholds |
| F5 | Model obsolescence (Qwen2.5-32B lifecycle) | Medium | Medium | **MEDIUM** | Architecture-agnostic pipeline; version pinning |
| P1 | GPU memory exceeded during contrastive fine-tuning | Medium | Medium | **MEDIUM** | LoRA + gradient checkpointing; memory profiling |
| P2 | Processing time too slow for commercial engagement SLAs | Low | Medium | **MEDIUM** | Batch extraction; layer subsetting |
| P3 | HDF5 scaling with commercial-sized corpora | Low | Low | **LOW** | Chunked reads; dataset-per-engagement architecture |
| S1 | Scaling from 20 to 50+ products per engagement | Low | Medium | **MEDIUM** | Linear scaling verified; parallelization plan |
| S2 | Reproducibility across model versions | Medium | High | **HIGH** | Calibration pinning; cross-model validation protocol |
| S3 | Client data security and confidentiality | Low | Critical | **HIGH** | Data handling protocol; local-only processing |

---

## Failure Mode Analysis

### F1: Contrastive Fine-Tuning Fails to Separate Register from Product Signal

**The core technical risk.** Decision #3 from the decision log commits to contrastive fine-tuning as the register correction strategy. This is the right approach, but it introduces a dependency on an untested training procedure.

**Why it could fail:**
- The 800 calibration stimuli are all LLM-generated with uniform distributional properties. A contrastive model trained on these may learn to factor out "LLM-generated register cues" rather than "register in general." When real documents arrive — with genuinely different vocabulary, sentence structure, and information density per channel — the learned register subspace may not generalize.
- Register and product identity may not be linearly separable in the representation space. Contrastive loss assumes that pulling same-product representations together is compatible with pushing different-register representations apart. If register and product share representational dimensions (plausible given that late-layer register F1 was still 1.0), the contrastive objective may face an irreconcilable trade-off.
- The calibration dataset has exactly 5 registers x 80 products x 2 variants = 800 stimuli. For contrastive fine-tuning (even with LoRA), this is a very small training set. Overfitting to the specific products and categories in the calibration data is a serious risk.

**Concrete indicators of failure:**
- After fine-tuning, register probe accuracy on held-out data remains above 0.9 (register is still encoded).
- Same-product cross-register similarity does not increase meaningfully vs. the base model (less than 0.05 improvement in mean cosine similarity).
- The model collapses: all representations converge regardless of product identity (loss of discriminative power).

**Severity:** Critical. If contrastive fine-tuning fails, the entire methodology reverts to the register-dominance problem identified in the critique. This blocks every downstream experiment.

### F2: Real Documents Break the Extraction Pipeline

**The pipeline was built for 80-150 word LLM-generated stimuli.** Real documents differ in ways that can cause silent failures or garbage outputs.

**Specific failure vectors:**

1. **Token length.** The current extraction pipeline (`extraction.py`, line 511-512) runs tokenization with `truncation=False`. A 5000-word regulatory filing could produce 6000+ tokens. Qwen2.5-32B has a 32K context window, so it will not fail outright, but:
   - Mean pooling across 6000 tokens will produce a representation dominated by document boilerplate, not product-specific content.
   - GPU memory for a single forward pass scales linearly with sequence length. At 6000 tokens with output_hidden_states=True (returning all 65 layer outputs), a single forward pass could consume 6000 x 65 x 5120 x 4 bytes = ~7.5 GB for hidden states alone, plus the GPTQ model itself (~18 GB). This approaches the 32 GB VRAM limit. Documents over ~8000 tokens will likely OOM. The pipeline does handle `torch.cuda.OutOfMemoryError` (line 591), but skipping documents is not acceptable in a commercial engagement.
   - A 30-word tweet will have its representation dominated by just ~15 content tokens after special token exclusion. The signal-to-noise ratio in such a small sample may be too low for meaningful comparison to a 5000-word document.

2. **Encoding and formatting.** Web-scraped documents contain HTML artifacts, Unicode edge cases (smart quotes, em dashes, zero-width characters), and structural boilerplate (navigation bars, cookie notices, legal footers). Amazon listings have structured fields (bullet points, "About this item" headers). Regulatory filings may contain tables, chemical formulas, or non-standard characters. None of this is handled by the current pipeline.

3. **Multi-product documents.** A brand's landing page may describe an entire product line. A consumer review may compare two products. A social media post may mention a product in passing alongside unrelated content. The pipeline assumes one document = one product, with no mechanism for isolating product-relevant content within a multi-topic document.

**Severity:** High. These are not edge cases — they are the normal characteristics of the documents the methodology must handle.

### F3: Calibration Data Does Not Transfer to Real-World Documents

**The calibration dataset is a controlled experiment, not a representative sample.** Decision #6 (pilot on calibration data first) is wise, but the transfer gap is structural, not just a matter of testing.

**Key transfer gaps:**

- **Information completeness.** Every calibration stimulus mentions all core attributes of the product, just in different registers. Real documents are selectively informative: a tweet mentions one benefit, a regulatory filing lists all ingredients, a review discusses one experience. The "coherence" question in real documents is partly about information completeness — which attributes are present in which channels — whereas the calibration data's "coherence" is entirely about register style. The contrastive model will be trained to factor out style differences between complete-information descriptions. Real-world "incoherence" is partly about information gaps.
- **Generative distribution.** All 800 stimuli were generated by an LLM (likely the same model or model family). They share distributional properties that real human-authored documents do not: consistent sentence structure, balanced clause lengths, similar vocabulary diversity. A model fine-tuned on these may learn features specific to LLM-generated text.
- **Category coverage.** The calibration data covers 8 CPG categories with 5 products each. A commercial engagement may involve categories not in the calibration set (automotive, financial services, B2B software). The contrastive model's register correction may not generalize outside the training distribution of categories.

**Severity:** Critical. This is the gap between "it works in the lab" and "it works on client data." Experiment 0 (decision #4) is the right gate, but the degradation tests need to be aggressive.

### F4: Coherence Metric Is Insensitive, Noisy, or Confound-Dominated

**Even if contrastive fine-tuning works and real documents process cleanly, the resulting "coherence score" may not measure what clients care about.**

**Insensitivity scenario:** After register correction, all products cluster tightly (the contrastive model pulled everything together too aggressively) or all products show the same moderate spread (register was the only source of variance, and removing it leaves noise). Result: the metric produces similar scores for all products, with no discriminative power. The n=10 per group design (decision #8) is better than n=5, but with 20 products and a metric that has unknown variance, the statistical power to detect a meaningful effect size is still limited.

**Noise scenario:** The metric is sensitive but unreliable. Running the same product through the pipeline twice with slightly different document samples produces materially different coherence scores. This would be catastrophic for a commercial offering where clients expect stability. Sources of instability include: which specific documents were collected (a different Amazon review changes the score), document preprocessing choices (how much boilerplate was removed), and stochastic elements in the extraction (GPTQ quantization introduces some non-determinism).

**Confound scenario (most likely):** The metric separates the two groups in Experiment 1, but for the wrong reason. The critique identified several confounds: brand size, documentation completeness, category vocabulary narrowness, marketing budget. With n=10 per group, there are not enough observations to statistically control for these. Decision #5 (baselines in Experiment 1) is critical here: if TF-IDF cosine similarity achieves comparable separation, the hidden-state methodology adds no value over a much simpler approach.

**Severity:** Critical. A metric that doesn't discriminate, is unstable, or measures the wrong thing has no commercial value.

### F5: Model Obsolescence

**Qwen2.5-32B-Instruct-GPTQ-Int4 is the foundation.** The specific layer dynamics (product identity peaking at layer 61, register dominant at all layers), the hidden dimension (5120), and the quantization behavior are all model-specific. The Qwen 2.5 series was released in late 2024. By mid-2026, Qwen 3 or a successor will likely be the current generation.

**Impact dimensions:**
- The contrastive fine-tuned model is a LoRA adapter (or similar) on top of Qwen2.5-32B. It cannot be transferred to a different base model.
- The calibration data's "ground truth" geometry (layer 61 peak, RSA values) is model-specific. Recalibrating on a new model requires re-running the full 800-stimulus extraction and re-training the contrastive model.
- If HuggingFace deprecates the GPTQ quantization format or the Qwen2.5 model weights become unavailable, the pipeline breaks entirely. Model weight hosting is not guaranteed indefinitely.
- Clients who receive a report based on Qwen2.5-32B cannot compare their results to a later engagement run on a different model. Longitudinal benchmarking (a likely upsell) requires model consistency.

**Severity:** Medium. This is manageable with version pinning but creates ongoing operational overhead.

---

## Performance Analysis

### P1: GPU Memory for Contrastive Fine-Tuning + Extraction

**Hardware:** RTX 5090 with 32 GB VRAM.

**Extraction (current pipeline):**
- Qwen2.5-32B-GPTQ-Int4 at 4-bit quantization: ~18 GB VRAM for model weights.
- Forward pass with `output_hidden_states=True`: for a 150-token input, hidden states across 65 layers = 150 x 65 x 5120 x 4 bytes = ~200 MB. Manageable.
- For a 5000-token document: 5000 x 65 x 5120 x 4 bytes = ~6.4 GB. Total: 18 + 6.4 = ~24.4 GB. Tight but feasible.
- For a 10000-token document: ~12.8 GB hidden states. Total: ~30.8 GB. At the limit. Attention KV cache adds more. **Documents over ~8000 tokens will likely OOM.**

**Contrastive fine-tuning:**
- LoRA fine-tuning of Qwen2.5-32B-GPTQ-Int4: model weights (18 GB) + LoRA parameters (<1 GB) + optimizer states (2-4 GB with AdamW on LoRA params) + gradient activations. With gradient checkpointing, peak memory ~24-26 GB. **Feasible on RTX 5090 but leaves little headroom.**
- Without gradient checkpointing: likely OOM. Gradient checkpointing is mandatory.
- Contrastive loss requires pairs/triplets in each batch. With batch size 2-4 (two 150-token stimuli per GPU), forward + backward memory is manageable. But mining hard negatives (same-register, different-product pairs that are close in embedding space) requires a forward pass over the full dataset first, which can be done in eval mode.

**Recommendation:** Profile memory usage before committing to fine-tuning parameters. Start with LoRA rank 8-16, gradient checkpointing, batch size 2 with gradient accumulation. If memory is still tight, consider QLoRA (quantized LoRA) which keeps the base model in 4-bit and only trains the adapter in higher precision.

### P2: Processing Time for Commercial Engagements

**Current extraction speed:** Based on the pilot validation function, each stimulus takes ~1-3 seconds on the RTX 5090 (150-token stimuli). The 800-stimulus extraction took approximately 15-40 minutes.

**Commercial engagement sizing (from decision #8 and seed document):**
- 20 products x 5 channels x 1 document = 100 documents minimum for Experiment 1.
- A full commercial engagement: 50 products x 5 channels = 250 documents.
- With variable-length documents (average ~500 tokens, some up to 5000), extraction time per document: ~2-10 seconds.
- Total extraction time: 250 x ~5 seconds = ~20 minutes. **This is fast enough.**

**Contrastive fine-tuning time:**
- Training on 800 stimuli with LoRA, ~10 epochs, batch size 2-4 with contrastive loss: roughly 20-60 minutes per training run on RTX 5090.
- Hyperparameter search (contrastive margin, LoRA rank, learning rate) may require 10-20 runs: 3-20 hours total.
- **This is a one-time cost for calibration, not a per-engagement cost.** But it must be re-run for model updates.

**Analysis time (RSA, permutation testing):**
- RSA computation is O(N^2) where N = number of stimuli. For N=250 (commercial engagement), pairwise distances are 31,125 pairs x 65 layers. Fast — seconds.
- Permutation testing at 10,000 permutations x 65 layers x 31K pairs: minutes, not hours.

**Total estimated turnaround for a commercial engagement:**
- Document collection and preprocessing: 2-5 days (human effort, not compute).
- Extraction: <1 hour.
- Analysis: <1 hour.
- Report generation: 1-2 days (human interpretation and writing).
- **Total: 1-2 weeks. Within the "weeks, not months" requirement.**

### P3: HDF5 Storage Scaling

**Current state:** 800 stimuli x 65 layers x 5120 dimensions x 3 datasets (hidden states, attention, MLP) x 4 bytes/float = ~3.2 GB uncompressed. The actual file is 3.5 GB with gzip compression.

**Per commercial engagement:** 250 stimuli x 65 layers x 5120 dims x 3 x 4 bytes = ~1 GB per engagement.

**Scaling to 10 engagements per year:** ~10 GB total. **Trivial.**

**The real storage concern** is not aggregate size but file access pattern. The current pipeline opens the HDF5 file, reads one layer at a time (`hs[:, layer_idx, :]`), applies anisotropy correction, and computes RDMs. This is efficient. But if a future enhancement needs to access multiple engagements simultaneously (e.g., cross-engagement normalization), the single-file-per-engagement pattern may require coordination. This is a low-priority architectural choice.

---

## Security Concerns

### S3: Client Data Handling

**The methodology requires ingesting actual client documents:** regulatory filings, internal marketing copy, competitive intelligence, product formulations. This is sensitive commercial data.

**Specific concerns:**
1. **Documents may contain trade secrets.** A brand's unreleased product formulation or pending regulatory submission is confidential. The pipeline stores raw text in `stimuli.json` and derived representations in HDF5. Both must be secured.
2. **The LLM sees client data.** Running Qwen2.5-32B locally (not via API) means client documents never leave the machine. This is a significant security advantage over cloud-based approaches. **Maintain local-only processing as a hard requirement.**
3. **Derived representations may be invertible.** Mean-pooled hidden states from late transformer layers contain enough information to reconstruct significant portions of the input text (demonstrated in embedding inversion attacks). HDF5 files containing client representations should be treated as confidential data, not as anonymized derivatives.
4. **Multi-client data isolation.** If running engagements for competing brands (Brand A and Brand B in the same category), their data must be strictly isolated. The current pipeline uses a single `data/` directory. A per-engagement directory structure is needed.
5. **Data retention and deletion.** Clients will expect contractual data deletion after the engagement. The pipeline needs a documented deletion procedure that covers: raw documents, stimuli JSON, HDF5 files, any cached model states, and log files (which may contain document snippets in error messages).

**Recommendations:**
- Per-engagement isolated directories with clear naming: `engagements/<client>-<date>/`
- Encryption at rest for all client data directories
- No cloud storage or external API calls during processing
- Contractual data handling terms with explicit retention and deletion clauses
- Audit logging of data access (who processed what, when)

---

## Scalability Assessment

### S1: Scaling from 20 Products (Validation) to 50+ Products (Commercial)

**Computational scaling:** The pipeline is O(N^2) in the number of stimuli for RSA computation (pairwise distances). With 20 products x 5 channels = 100 stimuli: 4,950 pairs. With 50 products x 5 channels = 250 stimuli: 31,125 pairs. This is a ~6x increase. At current performance, still completes in seconds. **Compute scaling is not a concern up to ~500 stimuli.**

**Document collection scaling:** This is the real bottleneck. Collecting 5 channel documents for 50 products = 250 documents. If each requires 15-30 minutes to find, clean, and validate: 60-125 hours of human labor. At $50-100/hour for a research analyst, that is $3,000-12,500 in document collection cost alone. **This is the dominant cost driver for commercial engagements, not compute.**

**Metric interpretation scaling:** A coherence matrix for 50 products x 5 channels has 250 pairwise comparisons per product and 1,225 cross-product comparisons. Turning this into an interpretable report requires significant analysis effort. The two-tier reporting (decision #7: brand coherence + market coherence) helps, but a 50-product report needs structured visualization and automated narrative generation to be practical.

### S2: Reproducibility Across Model Versions

**The fundamental tension:** The methodology's value comes from the specific geometry of Qwen2.5-32B's representations. But a commercial offering must be maintainable for years, during which the model will become obsolete.

**Reproducibility requirements:**
- **Within-engagement:** Running the same documents through the pipeline twice must produce the same results. GPTQ quantization is deterministic for the same inputs, so this should hold if the random seed and preprocessing are fixed. **Verify this empirically** — run the pilot extraction twice and compare outputs bit-for-bit.
- **Cross-engagement (same model version):** Two engagements run months apart on the same model version should produce comparable scores. This requires version-pinning the model, tokenizer, transformers library, and CUDA toolkit. Use a locked requirements file and containerization (Docker).
- **Cross-model (model version upgrade):** This is the hard problem. When Qwen2.5-32B is replaced, all existing engagement results become incomparable to new ones. Options:
  - **Re-run historical engagements on the new model** for clients who want longitudinal comparison. Feasible if raw documents are retained (conflicts with S3 data retention concerns).
  - **Develop a cross-model calibration transfer function** using the calibration dataset. Run the 800 stimuli through both models and learn a mapping. Untested and may not work well.
  - **Accept model-specific baselines** and report all scores relative to the calibration distribution for that model version. Clients cannot compare across model versions, but within-version comparisons remain valid.

**Recommendation:** Pin the model version and plan for a recalibration event every 12-18 months. Budget 1-2 weeks for recalibration (re-extraction, re-training contrastive model, validating that the coherence metric still discriminates). Communicate to clients that scores are model-version-specific.

---

## Recommended Mitigations

### Critical Priority (must address before Experiment 0)

**M1: Staged contrastive fine-tuning evaluation with fallback.**
- Define explicit pass/fail criteria for the contrastive model before training:
  - Register probe accuracy on held-out data drops below 0.7 (currently 1.0).
  - Same-product cross-register mean cosine similarity increases by at least 0.1 vs. base model.
  - Product probe accuracy on held-out data remains above 0.8 (no representation collapse).
- If contrastive fine-tuning fails these criteria after reasonable hyperparameter search (10-20 runs), fall back to partial RSA with register as nuisance (already implemented in `analysis.py`). This is a weaker correction but does not require training.
- **Timeline:** 1-2 days for training infrastructure, 2-3 days for hyperparameter search, 1 day for evaluation. Total: ~1 week.

**M2: Document preprocessing specification.**
- Before collecting any real documents, define and implement:
  - **Length normalization:** Documents over 4000 tokens are chunked into overlapping windows of 2000 tokens with 500-token overlap. Chunk representations are averaged. Documents under 50 tokens are flagged for review (may be too short for meaningful representation).
  - **Boilerplate removal:** Strip HTML tags, navigation elements, cookie notices, footer/header templates, Amazon listing structure, and social media metadata (hashtags, mentions, timestamps).
  - **Encoding normalization:** Standardize to UTF-8, replace smart quotes with straight quotes, normalize Unicode (NFC), remove zero-width characters.
  - **Multi-product detection:** Flag documents that mention multiple product names. Either split into sections or exclude.
- Implement as a `preprocessing.py` module that sits between document collection and extraction.
- **Timeline:** 2-3 days.

**M3: Pre-registered metric definition.**
- Before running Experiment 0, commit to:
  - Which layer(s) to use (start with layer 61 based on calibration data, but Experiment 0 should explore).
  - What distance metric (cosine distance, Euclidean on whitened representations).
  - How to aggregate pairwise distances into a single coherence score (proposal: mean cosine similarity across all channel pairs for a product, normalized by the calibration dataset mean and standard deviation to produce a z-score).
  - Minimum effect size for commercial relevance (proposal: Cohen's d >= 0.8 between "known consistent" and "known inconsistent" groups).
- Record these choices in the decision log before looking at Experiment 0 results. Experiment 0 is allowed to inform the metric definition, but the metric must be locked before Experiment 1.
- **Timeline:** 1 day.

### High Priority (must address before Experiment 1)

**M4: Calibration transfer validation in Experiment 0.**
- Design degradation tests that simulate the calibration-to-real-world transfer gap:
  - **Length variation:** Take calibration stimuli and truncate to 30 words (simulating tweets) and expand to 500 words (simulating detailed descriptions). Measure whether coherence scores change predictably.
  - **Attribute removal:** Take calibration stimuli and remove 1-3 core attributes (simulating incomplete real documents). Measure whether coherence drops and whether the correct attributes are identified as missing.
  - **Non-LLM text:** Manually rewrite 10-20 calibration stimuli in genuinely human style (or use real human-written product descriptions from the web). Compare representations to LLM-generated equivalents.
- If degradation is catastrophic (coherence scores become random under these perturbations), the methodology needs fundamental rethinking before investing in real-document collection.
- **Timeline:** 2-3 days.

**M5: Baseline battery for Experiment 1.**
- Implement alongside the hidden-state method:
  - TF-IDF cosine similarity across channel documents (simplest baseline).
  - Sentence-BERT (all-MiniLM-L6-v2) mean embedding cosine similarity.
  - BM25 keyword overlap score.
- If any baseline achieves comparable separation to the hidden-state method on Experiment 1's 20 products, the hidden-state approach adds no value. This is a go/no-go signal for the entire methodology.
- **Timeline:** 1 day (these are standard, off-the-shelf methods).

### Medium Priority (address before commercial engagements)

**M6: Reproducibility verification.**
- Run the extraction pipeline twice on the same 5 stimuli and verify bit-identical outputs. If GPTQ or CUDA introduces non-determinism, quantify the variance and determine whether it affects coherence scores at a commercially relevant threshold.
- Create a `requirements.lock` file and a Dockerfile for the pipeline.
- **Timeline:** 1 day.

**M7: Client data handling protocol.**
- Design the per-engagement directory structure.
- Implement a `cleanup.py` script that securely deletes all engagement data (raw documents, HDF5 files, logs).
- Draft a data handling addendum for client contracts.
- **Timeline:** 1-2 days.

**M8: Memory profiling for long documents.**
- Before collecting real documents, run the extraction pipeline on synthetic inputs of 500, 1000, 2000, 5000, and 10000 tokens. Record peak GPU memory for each. Determine the maximum safe document length and implement hard truncation at that limit.
- **Timeline:** 2-3 hours.

---

## Summary of Risk Posture

The project has **three critical risks** (F1, F3, F4) that can all be gated at low cost before significant investment:

1. **Contrastive fine-tuning** (F1) can be tested in ~1 week with a clear pass/fail criterion and a fallback to partial RSA.
2. **Calibration transfer** (F3) can be stress-tested in Experiment 0 using degradation simulations on existing data — no new data collection needed.
3. **Metric discriminative power** (F4) is tested directly by Experiment 1, but the risk is mitigated by pre-registering the metric, including baselines, and increasing to n=10 per group.

The **performance risks** (P1-P3) are manageable on the RTX 5090 hardware, with the single constraint that documents over ~8000 tokens must be chunked. Processing time is well within commercial SLAs.

The **scalability risks** (S1-S3) are addressable with standard engineering practices (containerization, version pinning, per-engagement isolation) but require deliberate implementation before the first commercial engagement.

**Overall assessment:** The risk profile is appropriate for a validation-stage research project. The sequential gating design (Experiment 0 before 1, Experiment 1 before 2-5) naturally limits downside exposure. The highest-value immediate action is implementing and testing the contrastive fine-tuning pipeline (M1), because it is the technical linchpin: if it works, the path to validation is clear; if it fails, the fallback to partial RSA is available but with lower expected performance.
