# Scope Audit

## Scope Alignment

### In Scope (Correctly Included)

These items align with either the original idea or explicit user decisions in the decision log:

1. **Three nested hypotheses (H1, H2, H3)**: Core of the original idea. Correctly preserved.
2. **RSA as primary method, probes as secondary**: User Decision #7. Correctly implemented as the analytical backbone.
3. **40 real + 40 fictional products**: User Decisions #1, #2, #10. Correctly expanded from the original 40 real products.
4. **5 registers x 2 variants = ~800 primary stimuli**: User Decision #8. Correctly scaled from original 200.
5. **Multi-source generation (Claude + GPT-4 on 10-product subset)**: User Decision #11. Correctly scoped as a subset test rather than full crossing.
6. **Hidden state decomposition (attention/MLP/residual via forward hooks)**: User Decision #5. Correctly included as Tier 3 deferrable.
7. **Hewitt & Manning control tasks for probes**: User Decision #6 (implicit via domain analysis acceptance). Correctly implemented in Step 5.2.
8. **Partial RSA with nuisance regressors**: User Decision #7 (part of RSA adoption). Correctly handles lexical overlap and stimulus length confounds.
9. **Anisotropy correction (both corrected and uncorrected)**: User Decision #3 and #9. Correctly implements the "analyze both" approach.
10. **Qwen 4-bit primary + Llama-3.1-8B validation**: Original idea. Correctly preserved.
11. **Qwen FP16 subset as Tier 4 deferrable**: User Decision #4 + Risk Analyst tiering. Correctly deprioritized.
12. **Fictional products as separate analysis condition**: User Decision #10. Correctly isolated from main results.
13. **Spearman > 0.9 quantization control threshold**: User Decision #12. Correctly pre-registered.
14. **GroupKFold stratification by product_id**: Domain Expert recommendation accepted by user. Correctly prevents data leakage.
15. **Nested CV for regularization tuning**: Domain Expert recommendation accepted. Correctly implemented.
16. **Within-category product discrimination**: Domain Expert critical recommendation #5. Correctly addresses topic-modeling alternative explanation.
17. **Effect-size criteria (Cohen's d > 0.3)**: Domain Expert recommendation #4. Correctly avoids over-reliance on significance with large N.
18. **Permutation-based statistical testing with FDR correction**: Domain Expert and Architect convergent recommendation. Correctly handles multiple comparisons.
19. **Tiered execution plan with MVR fallback**: Risk Analyst recommendation. Correctly preserves a minimum viable path.

### Scope Creep (Not Requested)

1. **Pydantic configuration system with YAML + CLI override support (Step 1.2)**: The original idea has no configuration system. A research experiment with a known, fixed design does not need a validated config loader with nested Pydantic models and CLI overrides. Severity: **moderate**. This adds 1-2 hours of development time for infrastructure that will be used exactly once.

2. **Checkpoint/crash recovery system (Step 1.4, `checkpoint.py`)**: Pickle-based checkpoint save/load with `StageCheckpoint` class. The longest single-stage runtime is ~30 minutes for extraction. Rerunning from scratch costs less than building crash recovery. Severity: **minor**. Quick to implement but unnecessary for runs under 1 hour.

3. **HDF5 validation module (`io.py` with NaN/Inf detection)**: A few assert statements inline would accomplish the same thing. A dedicated validation module is over-scoped. Severity: **minor**.

4. **`debug.yaml` with tiny model config (Step 1.2)**: Using Qwen2.5-1.5B as a debug model requires downloading an additional model. A simpler approach: use a smaller subset of stimuli with the real model, or use random tensors for code testing. Severity: **minor**.

5. **`validate_data.py` inter-stage data integrity script (Step 6.5)**: For a 5-stage pipeline run by one person, inter-stage validation is overkill. A few assertions at stage entry points suffice. Severity: **minor**.

6. **`run_all.py` orchestrator script (Step 6.5)**: Five scripts run sequentially is `bash -c "python run_stage1.py && python run_stage2.py && ..."`. A Python orchestrator adds no value. Severity: **minor**.

7. **Last-token pooling as robustness check (Step 3.4, Domain Requirement #7)**: Listed as "recommended but optional" in the plan. Not in the original idea or any user decision. However, it is a single function and low cost. Severity: **minor**.

8. **Register confusion matrices (Domain Requirement #10)**: Not requested by the user. Optional visualization. Severity: **minor** (a few lines of code if desired).

9. **Full reporting standards checklist (lines 364-381)**: While good practice, the original idea's deliverables are: stimulus dataset, hidden state archive, phase structure plots, probe accuracy curves, zone comparison table, go/no-go assessment. The plan adds 12+ reporting requirements. Severity: **minor** -- most are just "remember to include X" rather than additional code.

10. **Nine visualization types (Step 6.4)**: The original idea calls for phase structure plots, probe accuracy curves, and a zone comparison table (3 visualization types). The plan specifies 9 distinct visualization types across 4 visualization modules. Severity: **minor to moderate**. Several are genuinely useful (RSA heatmaps, decomposition panels), but 9 types with a dedicated `style.py` module is more than a research prototype needs.

### Missing (Requested but Not Covered)

1. **BoW baseline check**: The original idea explicitly mentions "BoW baseline check; strip top-10 register-predictive tokens if needed" as a risk mitigation for stimulus leakage (Section 7). The draft plan does not include a bag-of-words baseline analysis. The concern (8 categories being lexically separable) was partly addressed by Decision #1 (switching to 40-class primary), but the original leakage check was never explicitly implemented. Severity: **minor** -- partially addressed by the 40-class switch and within-category discrimination, but the explicit BoW baseline from the original idea was dropped.

2. **10% manual verification of stimuli**: The original idea calls for "manually verify 10% sample for semantic equivalence" (20 stimuli). The plan reduced this to "5% manual spot-check" (Step 2.4). This is a minor change but worth noting: 5% of 800 = 40 stimuli, which is actually more than 10% of 200. So the absolute count increased. Severity: **negligible** (the plan is actually more rigorous in absolute terms).

## Over-Engineering Findings

### 1. Enterprise-Grade Project Structure for a Research Script

**What's over-engineered**: The plan specifies a `src/plh/` package with 7 subpackages, a `scripts/` directory with 6 entry point scripts, a `config/` directory with YAML files, Pydantic config validation, and a `utils/` package with 3 modules. This is the directory structure of a reusable library, not a research experiment.

**Simpler alternative**: Two or three Jupyter notebooks (or Python scripts) totaling ~1500-2500 lines:
- `01_generate_stimuli.py` (stimulus generation + validation)
- `02_extract_and_analyze.py` (extraction, RSA, probes, all analysis)
- `03_visualize.py` (all plots)

Or equivalently, a single well-organized notebook with clear section headers. The "pipeline stages" are linear and will never be run by anyone other than the author.

**Impact**: The current structure requires writing ~15-20 boilerplate files (`__init__.py`, config loaders, entry point scripts, checkpoint utilities) that contribute zero scientific value. Estimated overhead: 3-5 hours.

### 2. Pydantic Configuration System

**What's over-engineered**: Step 1.2 describes a Pydantic-validated config system with nested models, YAML parsing, CLI override support, and a debug config variant. For an experiment with known, fixed parameters that will be run perhaps 5-10 times total.

**Simpler alternative**: A Python file with constants at the top, or a simple dict. If YAML is desired, `yaml.safe_load()` into a plain dict takes 3 lines.

**Impact**: 1-2 hours of development time for infrastructure used once.

### 3. Checkpoint/Crash Recovery System

**What's over-engineered**: A `StageCheckpoint` class with pickle-based save/load for crash recovery. The longest individual stage (extraction) runs in ~30 minutes on the target hardware. The stimulus generation stage already has checkpoint support built into Step 2.4 (save every 20 stimuli), which is reasonable for API calls. But a generalized checkpoint system is not needed.

**Simpler alternative**: For stimulus generation: save after each API call (trivial). For extraction: HDF5 already supports incremental writes -- the plan correctly describes this in Step 3.5 (resume by checking existing stimulus_ids). The dedicated checkpoint utility is redundant.

**Impact**: 30-60 minutes of unnecessary development.

### 4. Excessive Test Infrastructure

**What's over-engineered**: The plan calls for 6 test directories with ~15 test files covering every stage. This is appropriate for a production system maintained by a team. For a research prototype run by one person, the test suite will take longer to write than to debug the code manually.

**Simpler alternative**: One test file (`tests/test_core.py`) with ~10-15 tests covering the critical numerical operations: RDM computation, RSA correlation, anisotropy correction, pooling, and probe stratification. These are the functions where bugs would be silent and invalidate results. Skip testing config loading, schema validation, prompt generation, visualization, and entry point scripts.

**Impact**: 2-3 hours saved by testing only the mathematically critical path.

### 5. Six Entry Point Scripts

**What's over-engineered**: Six separate scripts in `scripts/` (`run_stage1.py` through `run_stage5.py` + `run_all.py` + `validate_data.py`). Each requires argparse setup, config loading, logging configuration.

**Simpler alternative**: Run stages directly: `python -m plh.stage1_stimuli.generate` or just call functions from a notebook/REPL. A single `run.py` with a stage argument is sufficient if scripts are desired.

**Impact**: 1 hour of boilerplate.

### 6. Separate Modules for Trivial Functions

**What's over-engineered**: Dedicated files for seed setting (`seeds.py`), I/O validation (`io.py`), pooling strategies (`pooling.py`), zone classification (`zone_classifier.py`), condition similarities (`cosine.py`), and go/no-go logic (`go_no_go.py`). Several of these are 10-30 line functions.

**Simpler alternative**: Group related functions into fewer files. The entire analysis pipeline (RSA + probes + hypothesis tests + controls) could live in 2-3 files totaling ~800-1200 lines.

**Impact**: 30-60 minutes of file management overhead. More importantly, increases cognitive load when navigating the codebase.

## Requirement Coverage

| Requirement (from idea + decisions) | Covered in Plan? | Plan Section | Sufficient? |
|-------------------------------------|-----------------|--------------|-------------|
| H1: Phase Structure hypothesis test | Yes | Steps 6.1, 4.3 | Yes -- RSA + cosine similarity |
| H2: Content Dominance hypothesis test | Yes | Steps 6.1, 5.1 | Yes -- RSA + probe comparison |
| H3: Protocol Layer Advantage test | Yes | Steps 6.1, 5.3 | Yes -- zone probes + RSA peak location |
| 40 real products, 8 categories x 5 | Yes | Step 1.3 | Yes |
| 40 fictional products (memorization control) | Yes | Step 1.3, 6.2 | Yes |
| 5 registers per product | Yes | Step 1.3 | Yes |
| 2 variants per product-register (800 stimuli) | Yes | Step 2.4 | Yes |
| RSA as primary analysis | Yes | Steps 4.2-4.3 | Yes |
| Linear probes as secondary | Yes | Steps 5.1-5.3 | Yes |
| Multi-source generation (Claude + GPT-4 subset) | Yes | Step 2.4 | Yes |
| Hewitt & Manning control tasks | Yes | Step 5.2 | Yes |
| Partial RSA with nuisance regressors | Yes | Step 4.3 | Yes |
| Anisotropy correction (both methods) | Yes | Step 4.1 | Yes |
| Hidden state decomposition (attn/MLP/residual) | Yes | Step 3.3 | Yes -- correctly tiered as deferrable |
| Qwen 4-bit primary model | Yes | Steps 3.1-3.2 | Yes |
| Llama-3.1-8B validation model | Yes | Step 3.2 | Yes |
| Qwen FP16 subset (quantization control) | Yes | Step 3.5 | Yes -- correctly tiered as Tier 4 |
| Spearman > 0.9 quant control threshold | Yes | Step 6.2 | Yes |
| GroupKFold by product_id | Yes | Step 5.1 | Yes |
| Nested CV for regularization | Yes | Step 5.1 | Yes |
| Within-category product discrimination | Yes | Step 6.2 | Yes |
| Effect-size criteria (Cohen's d > 0.3) | Yes | Step 6.1 | Yes |
| FDR multiple comparison correction | Yes | Step 6.1 | Yes |
| Permutation-based testing | Yes | Step 4.3 | Yes |
| Stimulus dataset deliverable (JSON) | Yes | Step 2.4 | Yes |
| Hidden state archive | Yes | Step 3.5 | Yes |
| Phase structure plots | Yes | Step 6.4 | Yes |
| Probe accuracy curves | Yes | Step 6.4 | Yes |
| Zone comparison table | Yes | Step 6.3 | Yes |
| Go/no-go assessment | Yes | Step 6.3 | Yes |
| BoW baseline leakage check | **Partial** | Decision #1 partially addresses | 40-class switch mitigates but explicit BoW check dropped |
| Tiered execution with MVR | Yes | Execution Schedule | Yes |

## Effort Assessment

### Current Plan Complexity

The plan describes **27 implementation steps** across **6 phases**, producing approximately:
- 25-30 Python source files
- 15 test files
- 6 entry point scripts
- 2 config files
- ~3,000-5,000 lines of code (estimated)

This is the complexity profile of a small software product, not a research experiment. The original idea described a 5-step protocol that could be implemented in ~1,500 lines.

### Time Budget Reality Check

| Activity | Plan Estimate | Realistic (with over-engineering) | Simplified Estimate |
|----------|--------------|----------------------------------|-------------------|
| Project scaffolding + config + tests | Evening 1 (partial) | 3-4 hours | 30 min |
| Product catalog (80 products) | Included in scaffolding | 2-3 hours | 1-2 hours |
| Stimulus generation + validation | Evening 1 (partial) | 3-4 hours | 2-3 hours |
| Extraction code + VRAM gate | Evening 1-2 | 3-4 hours | 2 hours |
| RSA + anisotropy analysis | Evening 2 | 3-4 hours | 2-3 hours |
| Linear probes + control tasks | Evening 3 | 3-4 hours | 2 hours |
| Statistical testing + viz | Evening 4 | 3-4 hours | 2-3 hours |
| FP16 control + final report | Evening 5 | 3-4 hours | 1-2 hours |
| **Total** | **~20-25 hours** | **22-31 hours** | **12-18 hours** |

The original estimate of ~15 hours was for the original 200-stimulus design. The scope expansions (user-approved) legitimately add time. However, the implementation plan's engineering overhead adds an additional 5-10 hours that could be eliminated.

### Minimum Viable Approach

The minimum viable experiment that tests all three hypotheses with all user-approved methodological controls:

1. **One script** for stimulus generation (~200 lines)
2. **One script** for extraction (~300 lines, including hooks and pooling)
3. **One script** for analysis (~500 lines, including RSA, probes, control tasks, hypothesis tests)
4. **One script** for visualization (~200 lines)
5. **One test file** for critical numerical functions (~150 lines)

Total: ~1,350 lines across 5 files. No Pydantic, no YAML, no checkpoint system, no package structure. Constants at the top of each file or in a shared `constants.py`.

This approach sacrifices: reusability, crash recovery (for runs under 1 hour), configuration flexibility (for an experiment with fixed parameters), and comprehensive test coverage (for code run by its author).

This approach preserves: all scientific rigor, all user-approved scope expansions, all methodological controls, all statistical tests.

### Recommended Simplifications

1. **Eliminate the Pydantic config system**. Use a Python dict or dataclass at the top of the main script. Save 1-2 hours.

2. **Eliminate the checkpoint system**. HDF5 incremental writes for extraction (already in the plan) and simple JSON appending for stimulus generation are sufficient. Save 30-60 minutes.

3. **Reduce test infrastructure to critical-path-only**. Test: RDM computation, RSA correlation, anisotropy correction, pooling math, GroupKFold stratification. Skip: config loading, schema validation, prompt strings, visualization, entry points. Save 2-3 hours.

4. **Merge entry point scripts into one**. `python run.py --stage {1-5|all}` or just run functions directly. Save 30-60 minutes.

5. **Flatten the package structure**. Instead of 7 subpackages with `__init__.py` files, use 4-5 flat modules: `stimuli.py`, `extraction.py`, `analysis.py`, `probes.py`, `viz.py`. Save 30 minutes + ongoing navigation time.

6. **Drop `debug.yaml` and tiny model config**. Test with real model on 5 stimuli, or use random tensors. Save 30 minutes.

7. **Defer last-token pooling, register confusion matrices, and CKA** unless core results call for them. These are "nice to have" robustness checks not in any user decision. Save 1-2 hours.

**Total savings from recommended simplifications: 6-10 hours.**

## Verdict

**The plan is scientifically sound but architecturally over-engineered for a solo research experiment.**

Every user decision and critical methodological requirement is correctly reflected. The analytical design (RSA primary, probes secondary, control tasks, partial RSA, within-category discrimination, permutation testing with FDR, tiered execution) is thorough and appropriate for publishable results.

The biggest scope concern is not any single addition but the **cumulative engineering overhead**: Pydantic configs, YAML files, checkpoint systems, 6 entry point scripts, 7 subpackages, 15 test files. None of these individually are large, but together they transform a "3-4 evenings of research" plan into a "5+ evenings of software development that happens to do research" plan.

The scientifically expanded scope (RSA, 800 stimuli, fictional products, control tasks, partial RSA) was user-approved and adds genuine value. The engineering infrastructure was not requested and adds time without improving the quality of the research output.

**Recommendation**: Keep the full analytical design. Simplify the implementation structure. A researcher who can design partial RSA with nuisance regressors does not need Pydantic to validate a YAML config file. Target the 12-18 hour simplified estimate, which preserves all scientific rigor while respecting the original "3-4 evenings" timeline.
