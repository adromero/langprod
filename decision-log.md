# Decision Log: protocol-layer-hypothesis

| # | Phase | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | 1 | 40-class individual product probe becomes primary analysis; 8-class category probe becomes secondary | DA showed 8 categories are lexically separable by BoW — too easy to distinguish protocol-layer effect from topic clustering |
| 2 | 1 | Add fictional products (invented brand names, novel specifications) as memorization control condition | Real products appear across registers in pretraining data; clustering could reflect memorization rather than protocol-layer processing |
| 3 | 1 | Apply rigorous anisotropy correction (mean centering + whitening) before cosine similarity computation | Middle layers are most anisotropic — raw cosine similarity would inflate H1 curve as measurement artifact |
| 4 | 1 | Run Qwen3.5-27B at full precision on subset (with CPU offloading) as quantization control | 4-bit quant vs full-precision 8B model differ on too many dimensions; disagreement would be uninterpretable |
| 5 | 1 | Pre-register topic-modeling null hypothesis alongside protocol-layer hypothesis | Topic clustering makes identical predictions for 8-class categories; need explicit discriminating criteria |
| 6 | 1 | Address stimulus contamination risk by diversifying generation sources or adding human-written controls | Claude-generated stimuli may embed systematic patterns detectable by another transformer |
| 7 | 1 | RSA (Representational Similarity Analysis) becomes primary analytical method; 40-class probe is secondary | 40-class probe with 5 samples/class is underpowered; RSA operates on 19,900 pairwise distances with far more statistical power |
| 8 | 1 | Expand stimulus count to ~400 (10 per product: additional registers or paraphrases) | Belt-and-suspenders: RSA primary + sufficient per-class samples for 40-class probe as secondary confirmation |
| 9 | 1 | Run cosine similarity analysis both with and without anisotropy correction | Anisotropy might BE the signal; correction could flatten the effect being measured; need both to interpret |
| 10 | 1 | Fictional products: 40 fictional (5 per category), parallel condition analyzed separately from real products | Clean memorization control; separate analysis avoids contaminating main results; if fictional products show same phase structure, memorization is ruled out |
| 11 | 1 | Multi-source generation: fully crossed on 10-product subset (Claude + GPT-4 + human). If generator effect negligible, main dataset uses Claude only | Tests for generator confound without requiring full crossing of all 40 products; practical compromise |
| 12 | 1 | Quantization control: Spearman correlation > 0.9 between full-precision and 4-bit per-layer RSA scores | Pre-registered quantitative threshold prevents subjective "looks similar enough" comparisons |
| 13 | 3 | Target hardware: PowerSpec 5090 system (RTX 5090, 32GB VRAM), not WSL desktop | Experiment runs on dedicated GPU system, resolving Open Question #3 |
| 14 | 3 | Draft plan approved, proceed to red-team | User satisfied with plan structure, conflicts resolution, and execution schedule |
| 15 | 5 | Register distinctiveness: quantitative soft gate (TF-IDF distance between registers, warning threshold) | Cheap (~30 min), gives publishable number, catches shallow register variation early |
| 16 | 5 | Fictional-vs-real threshold: split-half reliability baseline replaces arbitrary r > 0.7 | Empirically grounded; experiment not yet pre-registered so no commitment to break |
| 17 | 5 | Llama role: exploratory cross-model comparison, not confirmatory | Models differ on too many dimensions; primary claims rest on Qwen only |
| 18 | 6 | Primary model: `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` (64 layers, hidden_dim=5120, Qwen2ForCausalLM) | Qwen3.5-27B is a hybrid DeltaNet/attention VLM — non-homogeneous layers break probing assumptions. Qwen2.5-32B is standard decoder-only with 64 identical layers, ideal for interpretability |
| 19 | 6 | Quantization: GPTQ 4-bit via native transformers (not auto-gptq) | Official GPTQ-Int4 variant available, desc_act=false for hook compatibility, ~22-25 GB VRAM fits RTX 5090 |
