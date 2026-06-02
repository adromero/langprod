# Decision Log: brand-coherence-validation

| # | Phase | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | 0 | Tier: feature | Extends existing research pipeline with validation experiments, not a new platform |
| 2 | 0 | Goal: validate & refine | Stress-test the 5-experiment sequence, sharpen pass/fail criteria |
| 3 | 1 | Register correction: contrastive fine-tuning | Fine-tune a smaller model with contrastive loss to pull same-product representations together regardless of register. Addresses the root cause of register dominance (RSA r=0.670) rather than working around it. |
| 4 | 1 | Add Experiment 0: metric exploration | Exploratory phase on calibration data before committing to a metric formula. Avoids p-hacking by separating exploration from confirmation. |
| 5 | 1 | Baselines in Experiment 1 only | Run TF-IDF, BERTScore alongside Experiment 1 as a sanity check. Drop if hidden-state method clearly outperforms. |
| 6 | 1 | Pilot on calibration data (not new docs) | Simulate real-document conditions using existing 800 stimuli — vary length, remove attributes, test register correction — before collecting real documents. |
| 7 | 1 | Consumer reviews: separate tier | Analyze brand-controlled channels for "brand coherence" and include consumer reviews as a separate "market coherence" tier. Reports both. |
| 8 | 1 | Increase to n=10 per group | 20 products total for Experiment 1. Stronger statistical power at the cost of more document collection effort. |
| 9 | 5 | Ground-truth: hybrid rater protocol | The researcher assigns, one additional rater independently rates same products, compute agreement. 3-5h. Balances rigor with practicality. |
| 10 | 5 | Register probe threshold: relative drop >= 50% | Probe accuracy must drop from 1.0 to <= 0.5 after contrastive fine-tuning. Balances suppression with avoiding representation collapse. |
| 11 | 5 | Add wrong-product control to Experiment 3 | Probe each attribute against a different product in same category. ~1h extra. Catches lexical overlap confound. |
| 12 | 5 | Accept full scope (52-75h + 40-80h collection, 4-6 weeks) | All 6 experiments (0-5). No scope reduction. |
| 13 | 0 | Smoke test PASSED — mean centering is primary approach | Mean centering at layer ~30 produces within-category coherence differentiation (skincare +0.215, supplements +0.211) without contrastive training. Contrastive fine-tuning moved from critical path to escalation. Whitening removed (degenerate at N<<D). Layer selection is model-specific. Vocabulary narrowness control added. Effort revised to 40-63h. |
