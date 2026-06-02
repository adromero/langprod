# Domain Analysis: Transformer Interpretability Research

## Domain Context

This experiment sits at the intersection of three active subfields: (1) **mechanistic interpretability** of transformer language models, (2) **probing classifiers** as a methodology for understanding learned representations, and (3) **representational geometry** analysis using similarity-based methods. The "protocol layer hypothesis" — that transformer middle layers encode a format-agnostic semantic substrate — is a specific claim about the functional role of a particular architectural zone, tested through the lens of product classification across text registers.

The experiment is motivated by convergent evidence from several lines of work: Ng (2026) RYS layer-duplication experiments, Liu & Niehues (ACL 2025) cross-lingual alignment findings, Godey et al. (EACL 2024) on intrinsic anisotropy in self-attention, and Tyshchuk et al. (2024) on bell-shaped anisotropy profiles in decoder models. The theoretical framing — that middle layers learn an approximation of a "protocol-level" semantic representation stripped of surface-form serialization — extends existing "three-phase" accounts of transformer processing.

### Where This Fits in the Literature

The three-phase model of transformer processing (encoding, abstract processing, decoding) is increasingly well-supported. Tenney et al. (2019, "BERT Rediscovers the Classical NLP Pipeline") showed that linguistic features are recoverable at predictable layers, with syntactic information peaking in middle layers and semantic information distributed more broadly. Jawahar et al. (2019) found similar layered structure in BERT. For decoder-only models, the evidence from Ng (2026) and Liu & Niehues (2025) extends this to the autoregressive setting, with the critical addition that middle-layer representations appear to be self-consistent enough for re-traversal (the RYS finding).

The proposed experiment would contribute evidence about whether this three-phase structure holds not just across languages (as in Liu & Niehues) but across **text registers within a single language**. This is a less dramatic axis of variation than cross-lingual transfer but one that has not been systematically tested in the hidden-state geometry literature.

---

## Methodological Assessment

### RSA as Primary Method: Strong Choice

The decision to use Representational Similarity Analysis (RSA) as the primary method is well-grounded. RSA was developed in computational neuroscience (Kriegeskorte et al., 2008) precisely for situations where stimulus counts per condition are small but the number of pairwise comparisons is large. With ~800 stimuli (after the expansion to ~10 per product), you have ~320,000 pairwise distances — enormous statistical power for detecting geometric structure.

RSA has been applied to transformer representations by several groups:
- **Abnar et al. (2020)** used RSA to compare representational similarity across BERT layers.
- **Merchant et al. (2020)** applied RSA to compare representations across models and layers.
- **Wu et al. (2020)** used centered kernel alignment (CKA), a related method, to study layer similarity.

The critical advantage of RSA here is that it tests the **geometric prediction** directly: same-product pairs should have lower dissimilarity than same-category pairs, which should have lower dissimilarity than cross-category pairs. This three-level structure maps cleanly onto the model RDM (Representational Dissimilarity Matrix). A Spearman or Kendall rank correlation between model and observed RDMs at each layer provides a single, interpretable statistic per layer with clear statistical testing via permutation tests.

**Recommendation**: Use Spearman rank correlation between model and observed RDMs, with significance testing via permutation of stimulus labels (10,000 permutations). This is standard practice in the RSA literature and avoids parametric assumptions about the distance distribution.

### Probing Methodology: Adequate but Needs Refinement

The choice of L2-regularized logistic regression (C=1.0) as the probe architecture is reasonable and follows the "simple probe" school of thought articulated by Hewitt & Manning (2019) and Belinkov (2022). The argument: if a simple linear probe can recover the information, it is likely "encoded" in the representation rather than constructed by the probe itself.

However, several concerns:

1. **Fixed C=1.0 is suboptimal.** The regularization strength should be tuned via inner cross-validation or at least justified. C=1.0 is a default that may be too weak (allowing the probe to overfit on small per-class samples) or too strong (preventing the probe from fitting the available signal). The probing literature increasingly recommends cross-validated C selection (Belinkov, 2022) or minimum description length probing (Voita & Titov, 2020) to ensure probe complexity is appropriate.

   **Recommendation**: Use nested cross-validation — within each outer fold, tune C over {0.01, 0.1, 1.0, 10.0, 100.0} via 3-fold inner CV. This adds minimal computational cost and prevents the regularization strength from being a hidden confound.

2. **40-class probe with ~10 samples per class.** Even with expansion to ~400 stimuli, this is a challenging regime. The probe will work better if you:
   - Use stratified sampling to ensure each fold has the same product/register distribution.
   - Report both accuracy and macro-F1 (the current plan reports macro-F1, which is correct for imbalanced or small-class settings).
   - Report per-fold variance to characterize stability.
   - Consider a dimensionality reduction step (PCA to top-k components) before probing, which can stabilize small-sample classifiers without adding probe complexity.

3. **No control task (Hewitt & Manning 2019).** This is the most significant methodological gap in the probing design. Hewitt & Manning (2019, "A Structural Probe for Finding Syntax in Word Representations") introduced the concept of **selectivity**: a probe's accuracy on the task of interest minus its accuracy on a control task where labels are assigned randomly but with the same marginal distribution. If a probe achieves 60% on the real task but 40% on the control task, the selectivity is only 20% — meaning much of the apparent "encoding" is due to the probe's capacity to fit noise in the representation space.

   **Recommendation**: For each probe (40-class product, 8-class category, 5-class register), also train a matched control probe where product/category/register labels are randomly permuted but class sizes are preserved. Report selectivity = real accuracy - control accuracy at each layer. This is critical for distinguishing genuine encoding from probe artifact, especially with the relatively high-dimensional representations (~4096 for Qwen3.5-27B) and small per-class sample sizes.

4. **Consider MDL probing (Voita & Titov, 2020).** The Minimum Description Length probing framework measures not just whether information is recoverable but how *easily* recoverable it is (measured in bits required to transmit the labels given the representations). MDL probing is more robust to the "complex probe can extract anything" critique than accuracy-based probing. However, this is optional — the combination of a simple linear probe + control task is the minimum standard from the literature.

### Falsification Criteria: Mostly Reasonable but Need Calibration

The current falsification criteria are:

- **H1**: Per-layer cosine similarity between same-product different-register pairs must show a statistically significant increase from early to middle layers (p < 0.05, paired t-test).
- **H2**: Register-prediction accuracy must not exceed category-prediction accuracy by more than 5pp at any middle-layer point.
- **H3**: Best-performing layer for product classification must fall in the middle 60% of the layer stack, and must outperform the output-layer classifier by at least 2pp.

Concerns:

- **H1 criterion is too weak.** A p < 0.05 paired t-test with 19,900+ pairwise distances will be significant for almost any nonzero effect. The effect size matters more than significance. **Recommendation**: Add a minimum effect-size criterion — e.g., Cohen's d > 0.3 (small-to-medium effect) or the RSA correlation must exceed r = 0.1 at the peak layer.

- **H2 criterion is reasonable** but should be stated in terms of the RSA primary analysis as well. In RSA terms: the model RDM based on product identity should correlate more strongly with observed RDMs in middle layers than a model RDM based on register identity.

- **H3 criterion of 2pp advantage is possibly too strict or too lenient** depending on the variance. With ~400 stimuli and 40 classes, a 2pp difference in macro-F1 might not be reliably detectable. **Recommendation**: Frame H3 in RSA terms — the peak RSA correlation with the product-identity model RDM should occur in the middle 60% of layers and should be significantly higher than the RSA correlation at the output layer (permutation test, p < 0.05).

- **Multiple comparisons**: Testing three hypotheses at each of ~60 layers creates a multiplicity problem. **Recommendation**: Apply Bonferroni or FDR correction across layers when identifying peak regions, or use cluster-based permutation testing (standard in neuroimaging RSA) to identify contiguous layer ranges with significant effects.

### Statistical Testing Plan

The 5-fold stratified CV is standard. Additional recommendations:

- **Permutation testing for RSA**: 10,000 permutations of stimulus labels, computing the null distribution of RSA correlations. This is preferable to parametric significance testing because the pairwise distances are not independent (they share stimuli).
- **Bootstrap confidence intervals**: For probe accuracy curves across layers, use bootstrap resampling (1,000 replicates) to generate 95% CIs at each layer. This provides visual uncertainty quantification in the layer-by-layer plots.
- **Correction for dependent observations in RSA**: The pairwise distances in the RDM are not independent — each stimulus appears in multiple pairs. The permutation test handles this correctly, but parametric tests (e.g., Mantel test) can be liberal. Use permutation-based p-values throughout.

### Hidden State Decomposition: Valuable Addition

Decomposing hidden states into attention output, MLP output, and residual stream contributions is aligned with best practices from mechanistic interpretability (Elhage et al., 2021; Olsson et al., 2022). This decomposition can reveal whether the "protocol layer" effect (if present) is driven by:
- **Attention heads** (suggesting the effect is about contextual aggregation across token positions)
- **MLP layers** (suggesting the effect is about per-token feature computation)
- **Residual stream accumulation** (suggesting the effect is an emergent property of the full computation)

**Recommendation**: For the decomposition, extract the residual stream at each layer boundary, plus the isolated attention output and MLP output at each layer. Compute RSA on each component separately. This adds ~3x computational cost for the RSA computation (trivial) but provides substantial interpretive value.

---

## Literature Positioning

### Novelty Assessment

The "protocol layer" framing is **partially novel** but draws heavily on established ideas:

1. **The three-phase structure** is well-documented (Tenney et al. 2019, Jawahar et al. 2019, Liu & Niehues 2025). The experiment does not discover this but tests a specific prediction of it.

2. **Register-invariant representation** is less studied than cross-lingual representation. Most probing work tests for specific linguistic features (syntax, semantics, coreference) rather than invariance across text registers. The closest work is probably:
   - **Aharoni & Goldberg (2020)**: Showed that multilingual BERT representations cluster by language in early layers but by meaning in later layers — analogous to the predicted register-invariance here.
   - **Conneau et al. (2020)**: Showed cross-lingual transfer is possible from shared representations, implying some format-agnosticism.

3. **The "protocol" metaphor** is novel as a theoretical frame but is essentially a restatement of the "learned lingua franca" or "interlingual representation" hypothesis from the multilingual NLP literature, applied to within-language register variation rather than cross-lingual variation. This is a weaker test of the same idea — if middle layers can align across languages with completely different morphology, syntax, and vocabulary, aligning across English text registers (which share grammar and most vocabulary) is a less dramatic prediction.

4. **Product classification as the probing task** is novel as a choice of downstream domain. Most probing studies use linguistic tasks (POS tagging, dependency parsing, NER, semantic role labeling). Using an applied classification task (product categorization) is an interesting choice that increases ecological validity but moves away from the linguistic interpretability tradition.

### Key Papers to Cite and Engage With

The experiment should explicitly engage with:

- **Hewitt & Manning (2019)**: Control tasks for probes — the experiment needs this.
- **Belinkov (2022)**: "Probing Classifiers: Promises, Shortcomings, and Advances" — the definitive survey, which warns about probe complexity, selectivity, and the difficulty of distinguishing "encoded" from "extractable."
- **Voita & Titov (2020)**: MDL probing as an alternative to accuracy-based probing.
- **Pimentel et al. (2020)**: Information-theoretic probing with MDL — formalizes what it means for information to be "encoded" in a representation.
- **Conneau et al. (2018)**: "What You Can Cram into a Single $&!#* Vector" — early work on probing sentence representations, relevant methodology.
- **Kriegeskorte et al. (2008)**: Foundational RSA paper from neuroscience, for methodological grounding.
- **Kornblith et al. (2019)**: CKA (Centered Kernel Alignment) as an alternative to RSA for comparing representations. Consider using CKA as a robustness check alongside RSA.

### The "What Are Probes Really Measuring?" Debate

The experiment should address the well-known critique that probing classifiers may not measure what they claim to measure:

1. **Probes can extract information that is not "used" by the model** (Ravichander et al., 2021). A representation may contain product-identity information in a subspace that the model never accesses during its actual computation. The hidden-state decomposition helps here — if the signal is in the MLP output but not in the attention output, this suggests it is being actively computed, not just passively present.

2. **Linear probes are not as "simple" as claimed** when the representation is high-dimensional. A linear classifier in 4096 dimensions has enormous capacity. The control task (Hewitt & Manning 2019) is essential to calibrate this.

3. **RSA partially sidesteps the probe-complexity critique** because it measures geometric structure directly rather than training a classifier. However, the choice of distance metric (cosine, Euclidean, etc.) still imposes assumptions about what structure is "there."

---

## Stimulus Design Assessment

### Register Taxonomy

The five-register taxonomy (Marketing, Regulatory, Patent, Casual Social, Journalistic) is reasonable but has some linguistic issues:

1. **These are not "registers" in the strict sociolinguistic sense.** In register theory (Biber 1988, Biber & Conrad 2009), registers are defined by their situational characteristics (mode, field, tenor). The five categories here mix register features (formal/informal, spoken/written) with genre features (patent, news article) and functional features (marketing, regulatory). This is fine for the experiment's purposes — the goal is surface-form variation, not a contribution to register theory — but the write-up should acknowledge that "register" is used loosely.

2. **The registers are not equally distant from each other.** Marketing copy and journalistic writing share more surface features (both are edited, public-facing English prose) than either shares with patent language. Patent and regulatory language share formal, technical features. Casual social is the most distinct from all others. This asymmetry in register distances could confound the register probe — it might predict patent vs. casual easily but struggle with marketing vs. journalistic.

   **Recommendation**: Report the full register confusion matrix for the register probe at each layer, not just aggregate accuracy. This reveals which register pairs are confusable and at which layers, providing insight into what surface features the model strips away at each processing stage.

3. **Register-specific lexical markers are strong.** Patent text uses "comprising," "wherein," "embodiment." Regulatory text uses "shall," "compliance," "pursuant to." Marketing uses "you'll love," "revolutionary," "transform your." A simple bag-of-words classifier will distinguish registers with near-perfect accuracy. This is not a problem for the experiment per se (the hypothesis predicts that middle layers strip away these markers), but it means the register probe will have very high accuracy at early layers, establishing a strong baseline against which the middle-layer "content dominance" prediction is tested.

### Semantic Anchoring: Achievable but Challenging

The requirement that all descriptions of a given product convey the same core factual claims is critical and achievable for the main claims (product name, key features, specifications) but not for the pragmatic framing (a patent claim cannot include "you'll love how this feels" and a tweet cannot include "the apparatus comprising a handle portion"). The anchoring should be defined as:

- **Hard anchors** (must be present in every register variant): Product name/brand, 3-5 key factual attributes (ingredients, dimensions, specifications, primary function).
- **Soft anchors** (may vary by register): Tone, evaluative language, pragmatic framing, audience assumptions.

**Recommendation**: Create a "fact sheet" for each product listing the hard-anchor attributes. After generating all register variants, verify programmatically (or via LLM-as-judge) that each variant mentions all hard-anchor attributes. This creates a verifiable semantic-equivalence criterion.

### Stimulus Length

The 80-150 token target creates a genuine tension:

- Natural patent claims run 150-300 tokens.
- Natural tweets run 20-50 tokens.
- Enforcing 80-150 tokens makes all registers converge toward a similar "medium-length paragraph" format, partially defeating the purpose of register variation.

**Recommendation**: Allow length variation within a wider range (50-200 tokens) and include stimulus length as a covariate in the analysis. Specifically:
- In the RSA analysis, include a "length-difference model RDM" as a nuisance regressor (partial RSA, following Kriegeskorte & Kievit 2013). This controls for the possibility that similarity in the observed RDM is driven by length similarity rather than semantic similarity.
- In the probing analysis, report probe accuracy on length-matched subsets as a robustness check.

### Token Count: Mean Pooling Concerns

Mean pooling across token positions is a reasonable default but has known issues:

1. **Last-token pooling** is often better for decoder-only models because the last token has attended to the entire sequence. For causal (decoder-only) models like Qwen and Llama, each token only attends to previous tokens, so the last token has the richest representation.

2. **Mean pooling is sensitive to sequence length**: a 50-token mean and a 200-token mean have different noise profiles, and the 200-token mean is closer to the population mean of the representation, potentially inflating similarity for longer sequences.

**Recommendation**: Run the primary analysis with both mean pooling and last-token pooling. If results are consistent, report mean pooling as primary (more standard). If they diverge, report both and discuss. The current plan mentions last-token as a secondary analysis, which is appropriate.

---

## Interpretability of Results

### If All Hypotheses Are Supported

If H1, H2, and H3 are all supported, the experiment establishes:

1. **A geometric fact**: In the hidden-state space of Qwen3.5-27B and Llama-3.1-8B, product descriptions cluster by semantic content rather than by surface register in the middle layers. This is a **correlational** finding about representational geometry.

2. **A practical fact**: Middle-layer representations are more useful for register-invariant product classification than early-layer, late-layer, or output-layer representations.

3. **A suggestive fact about architecture**: The three-phase structure (encoding → abstract processing → decoding) holds for within-language register variation, not just cross-lingual variation.

What it does **not** establish:

- **Causality**: The experiment shows that product-identity information is geometrically accessible in middle layers, not that the model *uses* this representation during inference. Causal evidence would require interventions — e.g., activation patching (Meng et al., 2022) or distributed alignment search (Geiger et al., 2024) — showing that perturbing middle-layer representations disrupts product-relevant downstream behavior.

- **The "protocol" interpretation**: The finding that middle-layer representations are register-invariant does not uniquely support the "information protocol" theory. Alternative explanations include:
  - **Lexical abstraction**: Middle layers have simply learned to map synonym sets and paraphrases to similar representations. This is useful but does not require a "protocol layer."
  - **Topic modeling**: Middle layers encode topic/domain information (oral care, pet food, etc.) which is naturally register-invariant. This is the pre-registered null hypothesis, and the 40-class probe is designed to discriminate it from the protocol-layer hypothesis — but the discrimination is imperfect (see below).
  - **Training distribution statistics**: Products frequently appear in multiple registers in training data. Middle layers may have learned register-invariant product representations because register-invariant associations are statistically dominant in the training distribution, not because of any architectural "protocol" property.

- **Generalization beyond product classification**: The experiment tests one semantic domain (consumer products). The "protocol layer" claim implies generality across all semantic content. The experiment cannot establish this without additional domains.

### If H1 Is Supported but H2/H3 Are Not

This would suggest that middle layers show convergent representational geometry (the three-phase structure is real) but that the convergence does not preferentially encode semantic content over surface form. This is actually an interesting finding — it would suggest the middle-layer "compression" is not primarily semantic but might be a more generic dimensionality reduction. This outcome would be consistent with the anisotropy literature (the bell-shaped anisotropy curve reflects generic compression, not semantic abstraction).

### If All Hypotheses Are Falsified

This would mean either: (a) the three-phase structure does not hold for within-language register variation (possible — English registers are more similar to each other than different languages, so the "encoding" phase may not need to do much normalization), or (b) the experimental methodology is insufficiently sensitive to detect the effect. The fictional product condition and anisotropy correction make (b) less likely, but it cannot be fully ruled out.

### Discriminating the Protocol-Layer Hypothesis from Topic Modeling

The pre-registered discriminating criterion is: if the 40-class probe shows a protocol-layer advantage but the 8-class probe does not (or vice versa), this distinguishes fine-grained semantic identity from coarse topic clustering. The Devil's Advocate critique correctly notes that the more likely outcome is that **both** show some advantage. A sharper criterion:

**Recommendation**: Compute a "within-category product discrimination index" — the RSA correlation when the model RDM only includes within-category product pairs (i.e., the ability to distinguish Crest from Colgate from Sensodyne in middle layers). If the protocol-layer hypothesis is correct, this within-category discrimination should peak in the same middle layers as the overall product-identity RSA. If only topic modeling is at work, within-category discrimination should be flat or absent (because all toothpaste products map to the same "oral care" topic).

---

## Domain-Specific Recommendations

### From Mechanistic Interpretability

1. **Report layer-normalized RSA curves, not just raw curves.** Raw per-layer metrics are confounded by the baseline representation quality at each layer. Normalize by the within-layer variance or report rank-order statistics.

2. **Include a random-baseline model RDM.** In addition to the product-identity and register-identity model RDMs, include a random model RDM (random pairwise distances) to establish the null distribution of RSA correlations.

3. **Consider logit lens or tuned lens analysis (nostalgebraist 2020, Belrose et al. 2023).** The logit lens projects each layer's hidden state through the final unembedding matrix to see what tokens are predicted at each layer. If middle layers predict product-related tokens (brand names, category words) more strongly than register-related tokens, this provides converging evidence for the content-dominance hypothesis from a different analytical angle.

4. **Activation patching for causal evidence (optional but powerful).** If the correlational results are positive, a follow-up experiment could use activation patching (Meng et al., 2022): replace the middle-layer representation of product A described in register X with the middle-layer representation of product A described in register Y, and see if the model's subsequent behavior is unchanged. If the protocol layer truly encodes register-invariant product identity, patching across registers for the same product should have minimal effect on downstream predictions, while patching across products should have a large effect.

### From the Probing Literature

5. **Always report selectivity (Hewitt & Manning 2019).** Real accuracy minus control-task accuracy at each layer. Non-negotiable for publishable probing results.

6. **Consider probe-training dynamics (Saphra & Lopez 2019).** How quickly the probe converges (in training epochs) at each layer provides additional information — faster convergence suggests the information is more "linearly accessible." This is a lightweight addition to the standard probing pipeline.

7. **Report confidence intervals, not just point estimates.** For both RSA correlations and probe accuracies, bootstrap 95% CIs provide essential uncertainty quantification.

8. **Avoid over-interpreting small accuracy differences across layers.** With 40 classes and ~400 stimuli, a 2-3 percentage point difference in probe accuracy between layers may not be reliable. Use statistical tests (paired bootstrap, permutation tests) to assess significance, not just visual inspection of curves.

### From Representational Geometry

9. **Use partial RSA to control for confounds.** Following the partial correlation approach in the RSA literature (Kriegeskorte & Kievit 2013), include nuisance model RDMs for:
   - Stimulus length difference
   - Lexical overlap (Jaccard distance on token sets)
   - Generator identity (for the multi-source subset)

   This isolates the unique contribution of product identity to representational geometry, above and beyond these surface-level confounds.

10. **Consider CKA (Centered Kernel Alignment) as a robustness check.** CKA (Kornblith et al. 2019) is invariant to orthogonal transformation and isotropic scaling of representations, making it potentially more robust than cosine-similarity-based RSA to the anisotropy issues discussed in the experiment. Running CKA alongside RSA provides a built-in robustness check.

---

## Standards & Best Practices

### Minimum Reporting Standards for Probing Studies

Based on Belinkov (2022) and the broader probing literature, the experiment should report:

| Item | Required | Notes |
|------|----------|-------|
| Probe architecture and hyperparameters | Yes | Logistic regression, C (cross-validated), solver, max_iter |
| Control task accuracy (selectivity) | Yes | Random label permutation with preserved class sizes |
| Per-layer accuracy curves with confidence intervals | Yes | Bootstrap or CV-based CIs |
| Number of probe parameters vs. training examples | Yes | Ensures probe capacity is reasonable relative to data |
| Stratification strategy for CV | Yes | Stratified by product and register |
| Macro-F1 (not just accuracy) | Yes | Already planned |
| Per-class accuracy breakdown | Recommended | Reveals which products/categories are easy vs. hard |
| Training convergence | Recommended | Confirms probe has converged |
| Dimensionality reduction before probing | Optional | PCA to top-k can stabilize small-sample regimes |

### Minimum Reporting Standards for RSA

| Item | Required | Notes |
|------|----------|-------|
| Distance metric used | Yes | Cosine, Euclidean, or correlation distance |
| Model RDM construction | Yes | Explicit definition of predicted dissimilarity structure |
| Significance testing method | Yes | Permutation test with number of permutations stated |
| Multiple comparison correction | Yes | Across layers — FDR or cluster-based correction |
| Partial RSA for confound control | Recommended | Length, lexical overlap as nuisance regressors |
| RDM visualization at selected layers | Recommended | Aids interpretability of the geometric structure |

### Minimum Reporting Standards for Anisotropy Analysis

| Item | Required | Notes |
|------|----------|-------|
| Both corrected and uncorrected results | Yes | Already planned |
| Correction method specified | Yes | Mean centering, whitening, or both |
| Intrinsic dimensionality estimate per layer | Recommended | Participation ratio or effective rank |
| Isotropy score per layer | Recommended | Following Mu & Viswanath (2018) |

### Reproducibility Requirements

| Item | Required | Notes |
|------|----------|-------|
| All stimuli released | Yes | JSON with product attributes and register variants |
| Model versions and quantization details | Yes | Exact model IDs from HuggingFace |
| Random seeds for all stochastic processes | Yes | CV splits, permutation tests, any sampling |
| Code for hidden-state extraction | Yes | Forward hooks, pooling strategy |
| Pre-extracted hidden states (optional) | Recommended | Large files, but aids reproducibility |

---

## Summary of Critical Recommendations

Ordered by priority:

1. **Add control tasks (Hewitt & Manning 2019) for all probes.** This is the single most important methodological addition. Without it, probe accuracy is uninterpretable.

2. **Use partial RSA to control for lexical overlap and stimulus length.** These are the two most likely surface-level confounds that could mimic the protocol-layer effect.

3. **Tune regularization strength (C) via inner cross-validation** rather than fixing C=1.0.

4. **Add effect-size criteria to falsification thresholds**, not just statistical significance. With 19,900+ pairwise distances, even trivial effects will be significant.

5. **Compute within-category product discrimination** as the key test distinguishing the protocol-layer hypothesis from topic modeling.

6. **Run both mean pooling and last-token pooling** and report convergence/divergence.

7. **Consider logit lens analysis** as converging evidence from a different analytical angle.

8. **Report full confusion matrices** for register and category probes at selected layers.

9. **Apply multiple-comparison correction** across layers for both RSA and probe analyses.

10. **Include a nuisance RDM for generator identity** in the multi-source subset analysis.

---

## Conclusion

The experiment is methodologically sound in its overall design. The combination of RSA (primary) with probing classifiers (secondary), fictional products as memorization control, anisotropy correction, quantization control, and hidden-state decomposition addresses the major potential confounds. The research question — whether transformer middle layers encode register-invariant semantic content — is well-motivated by the convergent evidence from the RYS, cross-lingual alignment, and anisotropy literatures.

The main risks are:
- **Overstating the theoretical contribution.** The "protocol layer" framing suggests a deeper architectural principle than the experiment can establish. The experiment tests a correlational geometric property of hidden states. The write-up should be careful about the gap between "middle layers encode register-invariant product identity" (a geometric fact) and "middle layers implement a universal information protocol" (a functional-architectural claim).
- **Probe interpretability.** Without control tasks, probe accuracy is a poor measure of what information is "encoded." This is fixable by adding the Hewitt & Manning control.
- **Lexical confounds in RSA.** Without partial RSA controlling for lexical overlap, positive RSA results could reflect word-level similarity rather than abstract semantic structure.

With the recommended additions (control tasks, partial RSA, within-category discrimination, tuned regularization), the experiment would meet the methodological standards of the probing and representational geometry literatures and produce interpretable results regardless of which hypotheses are supported or falsified.
