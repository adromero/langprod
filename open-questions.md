# Open Questions: Protocol Layer Hypothesis

Items that must be resolved at implementation time. None block planning, but each affects implementation details.

## 1. Token Range Prompt Policy

**Status**: Minor decision, can be made during prompt engineering

The plan allows 50-200 tokens with length as a covariate (partial RSA). Generation prompts target 80-150 as the preferred range. The question is whether to:
- (a) Keep prompts targeting 80-150 and accept wider range post-hoc, or
- (b) Update prompts to explicitly allow register-natural lengths (shorter for social, longer for patent)

Option (b) produces more authentic register variation but increases length confound magnitude. Partial RSA handles the confound, but the correction assumes linearity.

**Action**: Start with option (a). If register distinctiveness check (Step 2.2) shows registers are too similar, switch to option (b) to increase surface-form variation.

## 2. Human Stimuli for Cross-Generator Subset

**Status**: Stretch goal, not blocking

The pipeline structurally supports human-written stimuli (just add entries to stimuli.json with generator="human"). Claude + GPT-4 provide sufficient generator diversity for the control analysis. Human stimuli would strengthen the paper but require significant manual effort (writing 100 product descriptions across registers).

**Action**: Skip unless core results are strong and paper submission is planned.

## 3. Logit Lens / Tuned Lens Analysis

**Status**: Follow-up if core results are positive

Recommended by the domain analysis as converging evidence. Project each layer's hidden state through the final unembedding matrix to see if middle layers predict product-related tokens more strongly than register-related tokens. Not in the core pipeline.

**Action**: Consider as Phase 7 extension if H1 and H2 are supported.

## 4. Causal Evidence (Activation Patching)

**Status**: Separate follow-up experiment

If all hypotheses are supported, the correlational RSA evidence does not establish that the model uses these representations during inference. Activation patching (Meng et al. 2022) would provide causal evidence: replace middle-layer representations across registers for the same product and observe downstream behavior.

**Action**: Plan as a separate experiment if the protocol-layer effect is confirmed.

## 5. GroupKFold Metric Interpretation

**Status**: Document in results, not blocking

With 80 products in 5 folds, each test fold has ~16 products. For the 40-class real-product probe, each test fold has only ~8 real products. Macro-F1 is computed over classes present in the test set, which varies across folds. This makes fold-to-fold comparison noisy.

**Action**: Report micro-F1 alongside macro-F1. Document the limitation explicitly in results. RSA is primary evidence and does not have this issue.
