# Brainstorm: brand-coherence-validation

## Status
- **Phase**: completed
- **Tier**: feature
- **Seed Document**: plan-coherence.md
- **Created**: 2026-03-28
- **Last Updated**: 2026-03-28 (completed)
- **Plan Directory**: plans/brand-coherence-validation
- **Resume Command**: `/brainstorm --resume plans/brand-coherence-validation/status.md`

## Original Idea
Validate and refine a 5-experiment research plan for a Brand Message Coherence methodology — a quantitative approach to measuring how consistently a product's core message survives translation across communication channels (regulatory, marketing, retail, social, consumer) using LLM hidden-state representations (RSA on extracted embeddings). The plan builds on a completed Protocol Layer Hypothesis experiment that produced a working measurement instrument and 800-stimulus calibration dataset. Goal: stress-test the experiment sequence, identify gaps in methodology and pass/fail criteria, and produce a refined plan ready for execution.

## Seed Document
See artifacts/00-seed-document.md. The document is a detailed research plan describing 5 sequential validation experiments (Real-Document Sensitivity, Channel Attribution, Attribute-Level Drill-Down, Temporal Coherence Drift, Competitive Benchmarking) with pass/fail criteria, failure modes, and a decision framework for interpreting outcomes.

## Project Context
Python research pipeline for the Protocol Layer Hypothesis experiment. Modules: `stimuli.py` (product catalogs, register specs, LLM generation, BoW baseline), `extraction.py` (hidden-state extraction from Qwen2.5-32B via forward hooks, HDF5 storage), `analysis.py` (RDM computation, RSA, permutation testing, anisotropy correction, partial RSA), `viz.py` (visualization), `run.py` (CLI orchestrator with generate/extract/analyze/probe/report stages). Uses scipy, scikit-learn, h5py, torch, transformers. Has test suite covering RDM, RSA, anisotropy, pooling, and GroupKFold CV.

## Phase Progress
| Phase | Name | Status |
|-------|------|--------|
| 0 | Setup & Triage | completed |
| 0.5 | Auto-Context Selection | skipped |
| 1 | Idea Challenge | completed |
| 2 | Synthesis | completed |
| 3 | User Checkpoint | completed |
| 4 | Red-Team | completed |
| 5 | Reconciliation | completed |
| 6 | Final Output | completed |
