# The Protocol Layer Hypothesis: An Experimental Test of Register-Invariant Semantic Representations in Transformer Middle Layers

**Date:** March 2026
**Model:** Qwen2.5-32B-Instruct-GPTQ-Int4 (64 layers, 5120 hidden dimensions)
**Stimuli:** 800 product descriptions (80 products x 5 registers x 2 variants)

---

## Abstract

We tested the hypothesis that transformer middle layers encode a format-agnostic semantic substrate where the same product, described across different linguistic registers, converges in representational geometry. Using 800 controlled product descriptions spanning five registers (marketing, regulatory, casual social, patent, journalistic) for 80 consumer products, we extracted hidden-state representations from all 64 layers of Qwen2.5-32B and analyzed their geometry via Representational Similarity Analysis (RSA) and linear probes. The protocol layer hypothesis was not supported: register identity, not product identity, was the dominant organizing principle at every layer (register RSA r=0.670 vs. product RSA r=0.371). Product identity peaked in late layers (layer 61/64), not the predicted middle layers. Category identity saturated early (probe F1=0.99 by layer 20), and register was trivially classifiable at all layers including the embedding layer (F1=1.000). These findings suggest that transformer representations are organized primarily by communication style rather than semantic content, with fine-grained product identity emerging only in the final processing stages.

---

## 1. Introduction

### 1.1 Motivation

Evidence from layer-duplication experiments (Ng 2026), cross-lingual alignment studies (Liu & Niehues, ACL 2025), and bell-shaped anisotropy profiles (Tyshchuk et al. 2024) suggests that transformer middle layers learn representations that are more abstract and less tied to surface form than early or late layers. This has been interpreted as evidence for a "protocol layer" --- a processing stage where the model constructs a format-agnostic semantic substrate, analogous to a network protocol that separates content from serialization format.

However, this interpretation has not been systematically tested for within-language register variation using controlled stimulus sets with known semantic equivalence. The present experiment provides that test using product descriptions --- a domain where the same factual content naturally appears across multiple text registers, enabling controlled manipulation of surface form while holding semantic content constant.

### 1.2 Hypotheses

We tested three nested hypotheses:

**H1 (Phase Structure):** Hidden-state representations of product descriptions exhibit a three-phase pattern (encoding, convergence, decoding divergence) across text registers, with product-identity RSA peaking in the middle 60% of layers.

- Falsification criteria: Peak product-identity RSA falls outside the middle 60% of layers, OR peak RSA r < 0.1, OR Cohen's d < 0.3 for early-vs-middle layer comparison.

**H2 (Content Dominance):** In the middle layers, semantic content (product/category identity) is more geometrically prominent than surface register.

- Falsification criteria: Register-identity RSA exceeds product-identity RSA in the middle layers.

**H3 (Protocol Layer Advantage):** Middle-layer representations outperform early, late, and output layers for product classification.

- Falsification criteria: Best probe layer falls outside the middle 60%, OR middle-layer advantage over output layer is < 2 percentage points.

### 1.3 Alternative Explanations

The experiment was designed to discriminate the protocol-layer interpretation from two alternatives:

1. **Topic modeling:** Middle layers merely encode coarse category membership (oral care, pet food, etc.), which is trivially register-invariant and does not require fine-grained semantic encoding.
2. **Lexical abstraction:** Middle layers map synonym sets to similar representations without deeper "protocol" structure.

The 40-class product probe (vs. 8-class category probe) and within-category RSA were the primary tools for this discrimination.

---

## 2. Methods

### 2.1 Stimuli

We generated 800 product descriptions spanning the full cross of:

- **80 products:** 40 real consumer products (e.g., Colgate Total, Blue Buffalo Life Protection) and 40 fictional products (e.g., AeroMint ProShield, TerraHound Ancestral Blend) with plausible but invented names and novel feature combinations
- **8 categories:** Oral Care, Pet Food, Home Cleaning, Sports Nutrition, Baby Care, Coffee/Beverage, Skincare, Smart Home (5 real + 5 fictional products per category)
- **5 registers:** Marketing copy, regulatory/technical, casual social, patent/IP, journalistic
- **2 paraphrase variants** per product-register combination

Each product was defined with 3--5 specific quantitative core attributes (e.g., fluoride_ppm: 1450, tube_size_oz: 4.8) and 2--3 distinguishing features. All stimuli were constrained to 80--150 words and required to convey all core attributes.

Stimuli were generated using the Claude CLI (`claude -p`) with register-specific prompts specifying voice, tone, structure, and vocabulary constraints. Generation achieved 800/800 stimuli with zero errors. Mean word count was 119.7 (SD=14.1, range 80--158).

The 40 fictional products served as a memorization control: because they do not exist in the model's training data, any representational clustering for fictional products cannot be attributed to memorized associations.

### 2.2 Quality Gates

Two quality checks were applied to the stimulus set before extraction:

**Bag-of-words baseline.** TF-IDF + logistic regression classifiers were trained on the three probe tasks (40-class product, 8-class category, 5-class register). All achieved 100% accuracy, indicating that surface lexical features (particularly product-specific numeric attributes and register-specific vocabulary) are sufficient for classification. This is expected given that each product has unique quantitative attributes that appear in every description, but it means any neural probe results must be interpreted against this high BoW ceiling.

**Register distinctiveness.** Mean inter-register TF-IDF cosine distance was 1.76x the mean intra-register distance (threshold: 1.5x), confirming that the five registers are lexically distinct.

### 2.3 Model

**Primary model:** Qwen2.5-32B-Instruct-GPTQ-Int4 (Qwen2ForCausalLM architecture, 64 transformer layers, hidden dimension 5120, 40 attention heads, 8 KV heads with GQA, GPTQ 4-bit symmetric quantization with group_size=128). Loaded via native transformers GPTQ integration with `device_map="auto"` on a single NVIDIA RTX 5090 (32GB VRAM). Model occupied approximately 19.4 GB VRAM.

### 2.4 Hidden State Extraction

For each of the 800 stimuli, we performed a forward pass through the model and captured:

- **Residual stream** at each of 65 positions (embedding layer + 64 transformer layers), yielding representations of shape (800, 65, 5120)
- **Attention output** at each of 64 layers, shape (800, 64, 5120)
- **MLP output** at each of 64 layers, shape (800, 64, 5120)

Representations were mean-pooled across token positions, excluding BOS, EOS, and PAD tokens. All representations were checked for NaN/Inf values (none found). Extraction completed in approximately 28 minutes (approximately 2.1 seconds per stimulus on average) and was saved incrementally to HDF5 with gzip compression (3.7 GB total).

### 2.5 Representational Similarity Analysis (RSA)

We computed Representational Dissimilarity Matrices (RDMs) at each layer using cosine distance (scipy `pdist` + `squareform`), yielding 800x800 distance matrices.

Three theoretical model RDMs were constructed:

1. **Product-identity model:** same-product pairs = 0, same-category pairs = 0.5, different-category pairs = 1.0
2. **Register-identity model:** same-register pairs = 0, different-register pairs = 1.0
3. **Within-category discrimination model:** same-product pairs = 0, different-product-same-category pairs = 1.0 (cross-category pairs excluded via NaN masking)

RSA correlations were computed as Spearman rank correlations between the upper triangles of observed and model RDMs.

### 2.6 Permutation Testing

We employed a tiered permutation testing strategy:

- **Tier 1 (screening):** 200 permutations at all 65 layers, with Benjamini-Hochberg FDR correction. All layers showed significant product-identity RSA (all p < 0.005, FDR-corrected).
- **Tier 2 (full test):** 10,000 permutations at the top-5 layers by RSA magnitude (layers 57, 58, 59, 60, 61). All achieved p < 0.0001.

### 2.7 Condition Similarity Analysis

We classified all stimulus pairs into three conditions:

- **SP-DR (Same Product, Different Register):** 3,200 pairs
- **DP-SC (Different Product, Same Category):** 36,000 pairs
- **DC (Different Category):** 280,000 pairs

Mean cosine similarity was computed per condition per layer.

### 2.8 Linear Probes

L2-regularized logistic regression probes were trained at each of the 65 layer positions for three classification tasks:

- **Product:** 80-class (40 real + 40 fictional products)
- **Category:** 8-class
- **Register:** 5-class

To manage the computational cost of probing 5120-dimensional representations, PCA dimensionality reduction to 200 components was applied before probe training (standard practice in neural probing; the first 200 principal components captured the dominant variance structure at each layer).

Cross-validation used 5-fold GroupKFold with product_id as the grouping variable, ensuring all variants of one product remained in the same fold. This prevents data leakage but means the 80-class product probe is effectively zero-shot for held-out products (each test fold contains products never seen during training), yielding 0.000 F1 across all layers. This is a design limitation, not a model failure --- the product probe result should be interpreted as "the probe cannot generalize product identity to unseen products" rather than "product identity is absent."

**Control probes** (Hewitt & Manning 2019) were trained with one random label permutation per task, preserving class size distributions. Selectivity (real F1 minus control F1) was computed at each layer.

**Zone probes** were trained on mean-pooled representations within four zones: early (layers 0--6, 10th percentile), protocol (layers 7--44, 10th--70th percentile), late (layers 45--63, 70th--99th percentile), and output (layer 64, final layer).

---

## 3. Results

### 3.1 RSA: Register Dominates Product Identity at Every Layer

Register-identity RSA was substantially and consistently stronger than product-identity RSA across the entire layer stack:

| Zone | Product RSA (mean) | Register RSA (mean) | Within-Category RSA (mean) |
|---|---|---|---|
| Early (layers 0--6) | 0.092 | 0.634 | 0.050 |
| Protocol (layers 7--44) | 0.224 | 0.581 | 0.127 |
| Late (layers 45--63) | 0.276 | 0.555 | 0.157 |
| Output (layer 64) | 0.139 | 0.617 | 0.068 |

**Product-identity RSA** peaked at layer 61 (r = 0.371, p < 0.0001), in the late zone --- not the predicted protocol (middle) zone. Product RSA increased monotonically from early layers through the late zone, then declined sharply at the output layer.

**Register-identity RSA** peaked at layer 47 (r = 0.670), within the protocol zone, and showed the opposite trajectory: strongest in early layers, declining through middle and late layers, then recovering slightly at the output layer. Register was the dominant geometric feature at every layer without exception.

**Within-category discrimination RSA** (the critical test distinguishing the protocol-layer hypothesis from topic modeling) peaked at layer 61 (r = 0.198), tracking the product-identity curve. The model can distinguish between products within the same category, but only in late layers.

### 3.2 Condition Similarities

Mean cosine similarities across conditions were uniformly high (all > 0.80) and showed relatively small differentiation:

| Condition | Mean Cosine Similarity | Peak Layer |
|---|---|---|
| SP-DR (Same Product, Different Register) | 0.962 | 6 |
| DP-SC (Different Product, Same Category) | 0.958 | 6 |
| DC (Different Category) | 0.945 | 6 |

All conditions peaked at layer 6 (early), suggesting that raw cosine similarity is dominated by the overall anisotropic structure of early-layer representations rather than by condition-specific geometry. The SP-DR vs. DP-SC gap (0.004) is small in absolute terms, consistent with register being the dominant organizing dimension.

### 3.3 Linear Probes

| Task | Early Zone F1 | Protocol Zone F1 | Late Zone F1 | Output Zone F1 |
|---|---|---|---|---|
| Product (80-class) | 0.000 | 0.000 | 0.000 | 0.000 |
| Category (8-class) | 0.947 | 0.990 | 0.990 | 0.989 |
| Register (5-class) | 1.000 | 1.000 | 1.000 | 1.000 |

**Register** was perfectly classifiable (F1 = 1.000) at every layer, including the embedding layer (layer 0: F1 = 0.996). This confirms that register cues are encoded in the surface-level token distribution and are never abstracted away by any layer of processing.

**Category** reached near-ceiling accuracy by layer 20 (F1 = 0.991) and remained stable through all subsequent layers, with a slight decline at the output layer (F1 = 0.985). The best single layer was layer 33 (F1 = 0.992), within the protocol zone, but the advantage over the output layer was only 0.7 percentage points --- below the pre-registered 2 percentage point threshold for H3.

**Product** probe returned F1 = 0.000 at all layers. As noted in Methods, this is an artifact of GroupKFold: holding out entire products from the training set makes 80-class classification impossible because the test classes were never seen during training.

### 3.4 Permutation Tests

All 65 layers showed significant product-identity RSA in the 200-permutation screen (all p < 0.005, FDR-corrected). The top-5 layers (57--61) all achieved p < 0.0001 in the 10,000-permutation full test. The null distribution at the peak layer (61) ranged from r = -0.003 to r = 0.003, compared to the observed r = 0.371, confirming that the product-identity signal is robust even though it peaks in the wrong location for the hypothesis.

---

## 4. Hypothesis Evaluation

### 4.1 H1 (Phase Structure): NOT SUPPORTED

The prediction was that product-identity RSA would peak in the middle 60% of layers (the "protocol zone," layers 7--44). Product-identity RSA instead peaked at layer 61, in the late zone. While RSA did increase from the early zone (mean r = 0.092) to the protocol zone (mean r = 0.224), the peak was clearly in the late zone (mean r = 0.276), and the single highest value (r = 0.371 at layer 61) was far outside the predicted region.

The effect was real and highly significant (p < 0.0001), but the phase structure predicted by the hypothesis --- encoding, convergence, decoding divergence --- was not observed. Instead, product identity accumulated monotonically through the network and peaked just before the output layer.

### 4.2 H2 (Content Dominance): NOT SUPPORTED

The prediction was that product identity would be more geometrically prominent than register identity in the protocol zone. The opposite was true: register RSA (mean r = 0.581) was 2.6x stronger than product RSA (mean r = 0.224) in the protocol zone. Register identity was the dominant geometric feature at every layer.

### 4.3 H3 (Protocol Layer Advantage): NOT SUPPORTED

The prediction was that the best-performing layer for product classification would fall in the protocol zone and outperform the output layer by at least 2 percentage points. The best category probe layer (33) was in the protocol zone, but the advantage over the output layer was only 0.7 percentage points, below the 2pp threshold. The product probe was uninformative due to the GroupKFold design limitation.

### 4.4 Overall Verdict: NO-GO

All three hypotheses were falsified. The protocol layer hypothesis is not supported by this data.

---

## 5. Discussion

### 5.1 What the Data Does Show

While the specific protocol-layer hypothesis was not supported, the experiment produced several substantive findings about how large language models internally represent product descriptions across registers:

**Register is the dominant organizing principle of representational geometry.** At every layer, the model's internal map of 800 product descriptions is organized primarily by how things are said (marketing vs. regulatory vs. casual, etc.), not what they are about. Register-identity RSA (mean r = 0.579 across all layers) consistently dwarfed product-identity RSA (mean r = 0.223). This is not merely a surface-level effect --- it persists deep into the network, suggesting that register cues shape processing at every stage.

**Product identity is a late-layer phenomenon.** Fine-grained semantic identity (which specific product is being described) only became geometrically prominent in layers 55--63, peaking at layer 61. This is consistent with a decoding interpretation: the model only needs to resolve specific product identity when preparing to generate product-specific tokens. During intermediate processing, coarse category membership suffices.

**Category identity saturates early and is maintained.** The 8-class category probe reached F1 = 0.99 by layer 20 and remained stable. This supports the "topic modeling" alternative: the model quickly determines the general domain (oral care, pet food, etc.) and this coarse semantic signal persists unchanged through subsequent processing.

**Register is encoded before any processing occurs.** Register probe F1 = 0.996 at the embedding layer means that the token distribution alone contains sufficient information to identify the register. The model never constructs a representation where register is stripped away --- it is baked in from the start and maintained throughout.

### 5.2 Implications for the Protocol Layer Interpretation

The strongest interpretation of our results is that **there is no protocol layer for within-language register variation.** The model does not construct a format-agnostic semantic substrate at any processing stage. Instead, it processes each register as a distinct communication mode, maintaining register-specific representations throughout the layer stack.

This does not rule out the possibility that a protocol-layer effect exists for cross-lingual variation (where the surface form differs far more dramatically than across registers within a single language) or for other types of content-form separation. But for the specific case of product descriptions across English-language registers, the representational geometry tells a clear story: form dominates content.

### 5.3 The Anisotropy Caveat

A significant limitation of this analysis is that anisotropy correction was not applied to the RSA computations. Transformer middle layers are known to be highly anisotropic (Tyshchuk et al. 2024), meaning raw cosine distances may be dominated by the global geometric structure rather than condition-specific variation. PCA-whitening or mean centering could change the relative prominence of product vs. register signals.

The condition similarity analysis supports this concern: all three conditions (SP-DR, DP-SC, DC) showed very high cosine similarities (all > 0.80) with small absolute differences between them, and all peaked at layer 6 --- consistent with anisotropy-dominated geometry. Corrected analyses could reveal content structure that is masked by the dominant anisotropic axes.

### 5.4 The BoW Ceiling and Stimulus Design

The 100% bag-of-words accuracy on all tasks indicates that surface lexical features are sufficient for classification. Each product has unique numeric attributes (e.g., "1450 ppm fluoride") that appear in every description, making products trivially distinguishable by keyword matching. This means we cannot cleanly attribute neural probe or RSA results to "deep semantic processing" versus "surface feature encoding."

A stronger test would use stimuli where the same product is described with deliberately non-overlapping vocabulary across registers --- forcing the model to rely on relational or structural similarity rather than shared keywords. This would require a fundamentally different stimulus generation approach but would more cleanly isolate the "protocol layer" phenomenon if it exists.

### 5.5 The GroupKFold Limitation

The product probe (80-class, GroupKFold by product_id) returned F1 = 0.000 at all layers because held-out products were never seen during training. This is the correct behavior for GroupKFold --- it prevents leakage --- but it renders the product probe uninformative for the hypothesis. A stratified k-fold design (allowing some variants of each product in training and others in test) would have tested whether the model can generalize across variants of the same product, which is the scientifically relevant question. This is a design flaw that should be corrected in any replication.

### 5.6 Broader Implications

The finding that LLMs organize representations by register has practical implications beyond this hypothesis test:

1. **Embedding-based search** across document types (e.g., matching a regulatory filing to a marketing claim) will underperform if register is the dominant geometric axis, because documents in different registers will be far apart in embedding space even when they describe the same thing.
2. **Style transfer quality** could be measured by how much an operation moves text along the register dimension while preserving position on the product/content dimension.
3. **Brand messaging coherence** could be quantified by measuring how tightly the same product clusters across channels, using the representational geometry as a measurement instrument.

---

## 6. Methods: Technical Details

### 6.1 Hardware

- **GPU:** NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **CPU:** AMD Ryzen 9 9950X3D (16-core)
- **RAM:** 32 GB
- **OS:** Ubuntu 24.04 (WSL2)

### 6.2 Software

- Python 3.12.3
- PyTorch 2.12.0.dev20260327+cu128 (nightly, CUDA 12.8 for Blackwell architecture support)
- Transformers 4.57.6
- auto-gptq 0.7.1 + optimum 1.27.0 (for GPTQ model loading)
- scikit-learn 1.8.0, scipy 1.17.1, numpy 2.4.3, h5py 3.16.0

### 6.3 Reproducibility

- Global random seed: 42 (Python random, numpy, PyTorch CPU+CUDA)
- All 800 stimuli archived in `data/stimuli.json` with generation metadata
- Hidden states archived in HDF5 with gzip compression
- Exact prompt templates and register specifications preserved in `stimuli.py`
- Full prompt/response pairs for stimulus generation via `claude` CLI

### 6.4 Compute Time

| Stage | Duration |
|---|---|
| Stimulus generation (800 via claude CLI) | ~40 minutes |
| Hidden state extraction (800 stimuli, 64 layers) | ~28 minutes |
| RSA computation (65 layers, 800x800 RDMs) | ~1 minute |
| Permutation testing (200 screen + 10,000 full) | ~18 minutes |
| Condition similarity computation | ~1 minute |
| Probe training (65 layers x 3 tasks, PCA-200) | ~193 minutes |
| **Total** | **~280 minutes** |

---

## 7. Conclusion

The protocol layer hypothesis --- that transformer middle layers encode a format-agnostic semantic substrate where the same product converges across registers --- was not supported. Register identity, not product identity, was the dominant organizing principle of representational geometry at every layer. Product identity peaked in late layers (layer 61/64), not the predicted middle layers. Category identity saturated early and was maintained, consistent with the topic-modeling alternative.

These negative results are informative: they tell us that within-language register variation is not "stripped away" at any processing stage. The model processes marketing text as marketing text and patent text as patent text, end-to-end. This has practical implications for any application that requires matching content across communication styles --- standard LLM embeddings will organize documents by style before content, and any cross-register search or comparison system must explicitly account for this.

---

## References

- Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. NAACL.
- Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis. Frontiers in Systems Neuroscience.
- Liu, D., & Niehues, J. (2025). Cross-lingual alignment in transformer middle layers. ACL.
- Ng, T. (2026). Layer duplication experiments in transformer architectures. Preprint.
- Tyshchuk, Y., et al. (2024). Bell-shaped anisotropy profiles in transformer representations.
- Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency.
