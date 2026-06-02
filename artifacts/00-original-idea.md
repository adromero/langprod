# Original Idea: The Protocol Layer Hypothesis

## Research Experiment Design

Do Transformer Middle Layers Encode a Format-Agnostic Semantic Substrate Useful for Product Classification?

### Experiment Protocol v1.0 — March 2026

**Hardware**: PowerSpec 5090 System
**Built on**: Ng (2026) RYS / Godey et al. (2024) Anisotropy / Liu & Niehues (2025) Cross-Lingual Alignment

## 1. Background and Motivation

### 1.1 The Three-Phase Architecture
Recent work on transformer internals has converged on a structural observation: decoder-only LLMs organize their layer stacks into three functional phases. Early layers encode surface-form inputs into an internal representation. Middle layers perform format-agnostic reasoning in a compressed representational space. Late layers decode back into language-specific token predictions.

This structure was demonstrated empirically by Ng (2026) in the RYS (Repeat Your Self) experiments, where duplicating contiguous blocks of middle layers in Qwen2-72B and Qwen3.5-27B produced measurable performance gains on math and emotional intelligence benchmarks without any weight changes or retraining. The key insight: middle-layer duplication works precisely because those layers operate in a self-consistent representational space where input and output distributions are compatible, allowing re-traversal without distribution mismatch.

Independent confirmation came from Liu & Niehues (ACL 2025), who analyzed internal representations across 1,000+ language pairs in Llama 3 and Qwen 2.5 and found that middle layers exhibit the strongest cross-lingual alignment and the best translation retrieval accuracy. LinguaMap (2026) further corroborated a three-phase internal structure across multiple multilingual LLM families.

### 1.2 The Hyper-Cone Effect and Anisotropy
The representational phenomenon underlying these observations is well-documented. Gao et al. (2019) identified the "representation degeneration problem": hidden states in transformers concentrate in a narrow hypercone rather than occupying the full representational space. This anisotropy — the tendency for all hidden states at a given layer to be unexpectedly close in angular distance — was originally attributed to cross-entropy optimization on long-tailed token distributions.

However, Godey et al. (EACL 2024) demonstrated that anisotropy is inherent to self-attention itself, observable even in character-level models, vision transformers, and speech models that should not suffer from token-frequency effects. Critically, Tyshchuk et al. (2024) showed that in decoder architectures specifically, the anisotropy profile follows a bell-shaped curve peaking in the middle layers — precisely the layers where Ng's RYS technique produces gains.

A nuanced finding complicates the "universal language" narrative: Lim, Aji, & Cohn (2025) found that larger models actually maintain more language-specific processing in their middle layers, with hidden states more likely to dissociate from a shared semantic space. Yet these larger models still perform better overall. This suggests the middle-layer representation is not a lossless universal encoding but rather a compressed, partially lossy substrate that retains enough shared structure for cross-lingual generalization while preserving some language-specific signal.

### 1.3 The Information Protocol Hypothesis
We propose a specific framing of the middle-layer representational space: all natural languages perform the same three fundamental operations — instantiate information (encode a mental state), disseminate it (transmit via a shared channel), and record it (persist in recoverable form). The grammar, syntax, and morphology of individual languages are serialization formats for a shared underlying protocol. What the middle layers learn is not a "universal language" per se, but an approximation of this protocol-level representation — a lossy compression of semantic content stripped of serialization-format overhead.

If this is correct, the protocol layer should exhibit a specific, testable property: representations of semantically identical content should cluster by meaning rather than by surface form, even when the surface forms are radically different. Previous work (Ng 2026, Maunder 2024) tested this across natural languages. We propose a stronger test: across text registers and document types within a single language.

**Core claim**: If the protocol layer is real, product identity should be recoverable from middle-layer representations regardless of whether the product is described in a marketing brochure, a regulatory filing, a tweet, or a patent abstract.

### 1.4 Why Product Classification Across Registers
We choose consumer product classification across radically different text registers as the experimental domain for three reasons:

1. **Ecological validity.** Market research routinely requires identifying and categorizing products from heterogeneous sources: Amazon listings, SEC filings, social media posts, patent databases, news articles. A format-agnostic product classifier would have direct commercial value.

2. **Controlled semantic content.** Unlike cross-lingual experiments (where translation quality introduces noise), we can precisely control the semantic content: the same product described in five registers, with each description verified to convey the same core attributes.

3. **Orthogonal to existing benchmarks.** No existing LLM benchmark tests register-invariant semantic classification using hidden-state geometry. This ensures our results cannot be explained by training data contamination or benchmark overfitting.

## 2. Research Questions

This experiment tests three nested hypotheses, ordered from weakest to strongest:

| Hypothesis | Description |
|-----------|-------------|
| **H1 — Phase Structure** | Hidden-state representations of product descriptions exhibit the three-phase pattern (encoding → convergence → decoding divergence) across text registers, consistent with the architecture observed for natural languages. |
| **H2 — Content Dominance** | In the middle layers (the hypothesized protocol zone), a linear probe can predict product category more accurately than it can predict text register. That is, semantic content dominates surface form in the representational geometry. |
| **H3 — Protocol Layer Advantage** | A simple classifier trained on middle-layer (protocol zone) representations outperforms classifiers trained on early-layer, late-layer, or final-output representations for register-invariant product classification. |

### 2.1 Falsification Criteria
- **H1 falsified** if per-layer cosine similarity between same-product different-register pairs does not show a statistically significant increase from layers 0–5 to the middle of the stack (p > 0.05, paired t-test).
- **H2 falsified** if, at any layer in the hypothesized protocol zone, register-prediction accuracy exceeds category-prediction accuracy by more than 5 percentage points.
- **H3 falsified** if the best-performing layer for product classification falls outside the middle 60% of the layer stack, or if the best middle-layer classifier does not outperform the output-layer classifier by at least 2 percentage points.

## 3. Experimental Materials

### 3.1 Product Categories (8 categories)
| # | Category | Representative Products |
|---|----------|----------------------|
| 1 | Oral Care | Electric toothbrush, whitening strips, mouthwash, floss pick system |
| 2 | Pet Food | Grain-free dog kibble, freeze-dried cat treats, puppy nutrition formula |
| 3 | Home Cleaning | Robot vacuum, enzyme-based stain remover, microfiber mop system |
| 4 | Sports Nutrition | Whey protein isolate, electrolyte powder, pre-workout supplement |
| 5 | Baby Care | Organic diaper cream, bottle sterilizer, infant probiotic drops |
| 6 | Coffee/Beverage | Single-origin espresso pods, cold-brew concentrate, oat milk creamer |
| 7 | Skincare | Retinol serum, mineral sunscreen SPF 50, hyaluronic acid moisturizer |
| 8 | Smart Home | Wi-Fi thermostat, video doorbell, smart plug with energy monitoring |

### 3.2 Text Registers (5 registers)
| # | Register | Characteristics | Example Source Analog |
|---|----------|----------------|---------------------|
| R1 | Marketing Copy | Aspirational, benefit-led, 2nd person, emotional appeals | Amazon listing, brand website |
| R2 | Regulatory / Technical | Passive voice, formal diction, precise measurements, compliance | FDA filing, MSDS sheet |
| R3 | Casual Social | 1st person, informal, slang, abbreviations, sentence fragments | Reddit review, tweet thread |
| R4 | Patent / IP | Dense nominal phrases, claim structure, technical jargon, legalese | Patent abstract |
| R5 | Journalistic | 3rd person, balanced tone, quotes from experts, inverted pyramid | Trade publication |

### 3.3 Stimulus Construction
For each of the 8 categories, select 5 specific products, yielding 40 unique products. Each product described in all 5 registers = 200 text stimuli total. Target length: 80–150 tokens per description.

**Critical constraint — semantic anchoring**: every description of a given product must convey the same core factual claims, merely expressed in the conventions of each register.

### 3.4 Design Matrix
- Product categories: 8
- Products per category: 5 → 40 products
- Registers per product: 5 → 200 stimuli
- Same-product pairs: C(5,2) × 40 = 400 pairs
- Cross-product same-category: ~400 pairs (sampled)
- Cross-category pairs: ~400 pairs (sampled)

## 4. Experimental Procedure

### 4.1 Hardware and Software
- **Hardware**: PowerSpec 5090 system (RTX 5090, 32GB VRAM)
- **Model (Primary)**: Qwen3.5-27B (4-bit GPTQ/AWQ quantization for comfortable fit)
- **Model (Validation)**: Llama-3.1-8B-Instruct (full precision)
- **Framework**: PyTorch + HuggingFace Transformers (forward hooks for hidden state extraction)
- **Analysis**: scikit-learn (logistic regression probes), NumPy/SciPy, matplotlib
- **Stimulus Generation**: Anthropic API (Claude Sonnet) with structured prompts

### 4.2 Step-by-Step Protocol

**Step 1: Generate Stimuli** (Day 1, ~2 hours)
- Define 40 products with 3–5 core factual attributes each
- Generate 5 register variants per product via Claude API
- Manually verify 10% sample for semantic equivalence
- Save as stimuli.json

**Step 2: Extract Hidden States** (Day 1, ~3–4 hours)
- Forward pass with output_hidden_states=True
- Mean pooling across token dimension (excluding BOS/EOS)
- Result tensor: (200, num_layers, hidden_dim)
- Save as hidden_states.npy

**Step 3: Compute Per-Layer Similarity Matrices** (Day 1–2, ~1 hour)
- Pairwise cosine similarity across all 200 stimuli per layer
- Three conditions: SP-DR, DP-SC, DC
- Centered cosine similarity (following Ng 2026)
- Plot three condition curves across layers → tests H1

**Step 4: Train Linear Probes** (Day 2, ~2 hours)
- Category probe (8-class) and Register probe (5-class) at each layer
- L2-regularized logistic regression, 5-fold CV
- Plot both accuracies across layers → tests H2
- Compare peak layer positions → tests H3

**Step 5: Classification Comparison** (Day 2, ~1 hour)
- Early zone (layers 0–5), Protocol zone (middle 60%), Late zone (final 10%), Output layer
- All logistic regression with identical hyperparameters
- 5-fold CV stratified by category and register
- Macro F1 comparison → tests H3

## 5. Expected Outcomes

| Outcome | H1 (Phase Structure) | H2 (Content Dominance) | H3 (Protocol Advantage) |
|---------|---------------------|----------------------|------------------------|
| Full Support | Clear 3-phase curve with SP-DR similarity peaking mid-stack | Category probe > register probe from layer ~10–50, gap > 10pp | Protocol zone F1 > output F1 by ≥5pp |
| Partial Support | 3-phase present but noisier | Category > register in narrow window only | Protocol zone F1 > output by 2–5pp |
| Falsified | No convergence pattern | Register probe dominates throughout | Output ≥ protocol zone |

### 5.1 Secondary Observations
- Quantization sensitivity
- Category-specific convergence patterns
- Register difficulty gradient
- Anisotropy correlation with classification utility

## 6. Timeline
~15 hours across 3–4 evenings

## 7. Risks and Mitigations
- **Stimulus leakage**: BoW baseline check; strip top-10 register-predictive tokens if needed
- **VRAM constraints**: 4-bit quantization as primary run
- **Small N**: Report CIs; expand to 400 stimuli if noisy
- **Mean pooling**: Secondary analysis using last-token hidden state
- **Ceiling effect**: Switch to 40-class product prediction if categories trivially separable

## 8. Deliverables
- Stimulus dataset (200 product descriptions, JSON)
- Hidden state archive (pre-extracted tensors)
- Phase structure plots
- Probe accuracy curves
- Zone classifier comparison table
- Go/no-go assessment for protocol-layer feature extraction
