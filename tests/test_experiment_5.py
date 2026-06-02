"""Tests for coherence.experiment_5 -- uses synthetic data, no GPU required."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from coherence.experiment_5 import (
    ACCEPTABLE_VERDICTS,
    MIN_BRANDS,
    MIN_PAIRWISE_AGREEMENT,
    N_EXPERTS,
    BrandScore,
    ExpertRanking,
    Experiment5Results,
    PairwiseComparison,
    build_results,
    check_gates,
    compute_brand_scores,
    compute_inter_expert_taus,
    compute_metric_expert_taus,
    compute_pairwise_comparisons,
    generate_expert_form,
    load_competitive_manifest,
    load_expert_rankings,
    results_to_dict,
)
from coherence.metrics import CoherenceResult, compute_coherence_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D = 64  # reduced hidden dim for tests
CHANNELS = ("regulatory", "marketing", "retail", "social", "consumer_review")
BRAND_IDS = ["brand_a", "brand_b", "brand_c", "brand_d", "brand_e"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verdict_file(tmp_dir: Path, overall: str = "PASS") -> Path:
    """Create a mock verdict.json."""
    verdict_path = tmp_dir / "verdict.json"
    verdict = {
        "verdict": {"overall": overall},
        "roc": {"auc": 0.92},
    }
    with open(verdict_path, "w") as f:
        json.dump(verdict, f)
    return verdict_path


def _make_metric_selection(tmp_dir: Path) -> Path:
    """Create a mock metric_selection.json."""
    ms_path = tmp_dir / "metric_selection.json"
    ms = {
        "aggregation": "mean_pairwise",
        "layer_hdf5_index": 31,
        "layer_label": "layer_30",
    }
    with open(ms_path, "w") as f:
        json.dump(ms, f)
    return ms_path


def _make_global_mean(tmp_dir: Path, dim: int = D) -> Path:
    """Create a mock global_mean.npy."""
    gm_path = tmp_dir / "global_mean.npy"
    np.save(gm_path, np.zeros(dim))
    return gm_path


def _make_competitive_manifest(
    tmp_dir: Path,
    brand_ids: List[str] = BRAND_IDS,
) -> Path:
    """Create a competitive experiment manifest."""
    manifest = {
        "category": "oral_care",
        "brands": [
            {"brand_id": bid, "brand_name": bid.replace("_", " ").title()}
            for bid in brand_ids
        ],
    }
    manifest_path = tmp_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def _make_synthetic_brand_embeddings(
    brand_ids: List[str],
    channels: tuple[str, ...] = CHANNELS,
    dim: int = D,
    rng: Optional[np.random.Generator] = None,
    coherence_levels: Optional[Dict[str, float]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Create synthetic brand embeddings with controlled coherence levels.

    Parameters
    ----------
    coherence_levels :
        Optional mapping of brand_id -> spread (lower = more coherent).
        Default assigns linearly increasing spread.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if coherence_levels is None:
        coherence_levels = {
            bid: 0.05 + i * 0.1
            for i, bid in enumerate(brand_ids)
        }

    result: dict[str, dict[str, np.ndarray]] = {}
    for brand_id in brand_ids:
        spread = coherence_levels.get(brand_id, 0.2)
        base = rng.standard_normal(dim)
        base /= np.linalg.norm(base)

        channel_embs = {}
        for ch in channels:
            noise = rng.standard_normal(dim) * spread
            channel_embs[ch] = base + noise

        result[brand_id] = channel_embs

    return result


def _make_expert_rankings(
    exp5_dir: Path,
    brand_ids: List[str],
    n_experts: int = N_EXPERTS,
    rng: Optional[np.random.Generator] = None,
    agreement_level: str = "high",
) -> None:
    """Create expert ranking files.

    Parameters
    ----------
    agreement_level : "high" | "low" | "perfect"
        Controls how similar expert rankings are.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    exp5_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_experts):
        if agreement_level == "perfect":
            ranking = list(brand_ids)
        elif agreement_level == "high":
            ranking = list(brand_ids)
            # Swap at most one adjacent pair
            if i > 0 and len(ranking) > 1:
                swap_idx = i % (len(ranking) - 1)
                ranking[swap_idx], ranking[swap_idx + 1] = (
                    ranking[swap_idx + 1],
                    ranking[swap_idx],
                )
        else:  # low
            ranking = list(brand_ids)
            rng.shuffle(ranking)

        data = {
            "expert_id": f"expert_{i + 1}",
            "ranking": ranking,
        }
        path = exp5_dir / f"expert_ranking_{i + 1}.json"
        with open(path, "w") as f:
            json.dump(data, f)


# ---------------------------------------------------------------------------
# Gate check tests
# ---------------------------------------------------------------------------


class TestCheckGates:
    """Tests for the check_gates function."""

    def test_missing_metric_selection(self, tmp_path: Path):
        """check_gates exits when metric_selection.json is missing."""
        with patch("coherence.experiment_5.METRIC_SELECTION_PATH", tmp_path / "missing.json"), \
             pytest.raises(SystemExit):
            check_gates()

    def test_missing_global_mean(self, tmp_path: Path):
        """check_gates exits when global_mean.npy is missing."""
        ms_path = _make_metric_selection(tmp_path)
        with patch("coherence.experiment_5.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_5.GLOBAL_MEAN_PATH", tmp_path / "missing.npy"), \
             pytest.raises(SystemExit):
            check_gates()

    def test_missing_verdict(self, tmp_path: Path):
        """check_gates exits when verdict.json is missing."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        with patch("coherence.experiment_5.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_5.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_5.VERDICT_PATH", tmp_path / "missing.json"), \
             pytest.raises(SystemExit):
            check_gates()

    def test_failing_verdict(self, tmp_path: Path):
        """check_gates exits when verdict is FAIL."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        verdict_path = _make_verdict_file(tmp_path, "FAIL")
        with patch("coherence.experiment_5.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_5.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_5.VERDICT_PATH", verdict_path), \
             pytest.raises(SystemExit):
            check_gates()

    def test_passing_gates(self, tmp_path: Path):
        """check_gates succeeds with valid files."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        verdict_path = _make_verdict_file(tmp_path, "PASS")
        with patch("coherence.experiment_5.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_5.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_5.VERDICT_PATH", verdict_path):
            metric_sel, global_mean, verdict = check_gates()

        assert metric_sel["aggregation"] == "mean_pairwise"
        assert global_mean.shape == (D,)
        assert verdict["verdict"]["overall"] == "PASS"

    def test_pass_no_value_added_accepted(self, tmp_path: Path):
        """check_gates also accepts PASS_NO_VALUE_ADDED."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        verdict_path = _make_verdict_file(tmp_path, "PASS_NO_VALUE_ADDED")
        with patch("coherence.experiment_5.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_5.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_5.VERDICT_PATH", verdict_path):
            _, _, verdict = check_gates()
        assert verdict["verdict"]["overall"] == "PASS_NO_VALUE_ADDED"


# ---------------------------------------------------------------------------
# Manifest loading tests
# ---------------------------------------------------------------------------


class TestLoadCompetitiveManifest:
    """Tests for load_competitive_manifest."""

    def test_valid_manifest(self, tmp_path: Path):
        """Loads a valid manifest successfully."""
        _make_competitive_manifest(tmp_path)
        manifest = load_competitive_manifest(tmp_path)
        assert len(manifest["brands"]) == 5

    def test_missing_manifest(self, tmp_path: Path):
        """Exits if manifest.json is missing."""
        with pytest.raises(SystemExit):
            load_competitive_manifest(tmp_path)

    def test_insufficient_brands(self, tmp_path: Path):
        """Exits if fewer than MIN_BRANDS brands."""
        _make_competitive_manifest(tmp_path, brand_ids=["a", "b", "c"])
        with pytest.raises(SystemExit):
            load_competitive_manifest(tmp_path)


# ---------------------------------------------------------------------------
# Expert form generation tests
# ---------------------------------------------------------------------------


class TestGenerateExpertForm:
    """Tests for generate_expert_form."""

    def test_form_generation(self, tmp_path: Path):
        """Generates an expert form with correct structure."""
        _make_competitive_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        form_path = generate_expert_form(manifest, tmp_path / "exp5")

        assert form_path.exists()
        with open(form_path) as f:
            form = json.load(f)

        assert form["pre_registered"] is True
        assert len(form["brands"]) == 5
        assert form["expert_id"] is None
        assert form["category"] == "oral_care"
        assert "instructions" in form

    def test_form_brands_have_null_rank(self, tmp_path: Path):
        """All brands in the form have null rank (to be filled by expert)."""
        _make_competitive_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        form_path = generate_expert_form(manifest, tmp_path / "exp5")
        with open(form_path) as f:
            form = json.load(f)

        for brand in form["brands"]:
            assert brand["rank"] is None


# ---------------------------------------------------------------------------
# Expert ranking loading tests
# ---------------------------------------------------------------------------


class TestLoadExpertRankings:
    """Tests for load_expert_rankings."""

    def test_load_valid_rankings(self, tmp_path: Path):
        """Loads valid expert rankings."""
        _make_expert_rankings(tmp_path, BRAND_IDS)
        rankings = load_expert_rankings(tmp_path)

        assert len(rankings) == N_EXPERTS
        for ranking in rankings:
            assert len(ranking.ranking) == 5
            assert len(ranking.rank_map) == 5

    def test_missing_ranking_file(self, tmp_path: Path):
        """Exits if a ranking file is missing."""
        # Create only 2 out of 3 required rankings
        for i in range(1, 3):
            data = {"expert_id": f"expert_{i}", "ranking": BRAND_IDS}
            with open(tmp_path / f"expert_ranking_{i}.json", "w") as f:
                json.dump(data, f)

        with pytest.raises(SystemExit):
            load_expert_rankings(tmp_path)

    def test_rank_map_correctness(self, tmp_path: Path):
        """Rank map correctly maps brand_id to 1-based rank."""
        _make_expert_rankings(tmp_path, BRAND_IDS, agreement_level="perfect")
        rankings = load_expert_rankings(tmp_path)

        for ranking in rankings:
            for i, brand_id in enumerate(ranking.ranking):
                assert ranking.rank_map[brand_id] == i + 1


# ---------------------------------------------------------------------------
# Brand scoring tests
# ---------------------------------------------------------------------------


class TestComputeBrandScores:
    """Tests for compute_brand_scores."""

    def test_basic_scoring(self, tmp_path: Path):
        """Computes scores and ranks for all brands."""
        _make_competitive_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        brand_embeddings = _make_synthetic_brand_embeddings(BRAND_IDS)
        scores = compute_brand_scores(brand_embeddings, manifest, "mean_pairwise")

        assert len(scores) == 5
        # Ranks should be 1-5
        ranks = [s.metric_rank for s in scores]
        assert sorted(ranks) == [1, 2, 3, 4, 5]

        # Scores should be sorted descending
        coherences = [s.market_coherence for s in scores]
        assert coherences == sorted(coherences, reverse=True)

    def test_brand_with_insufficient_channels(self, tmp_path: Path):
        """Brand with < 2 channels gets None coherence."""
        _make_competitive_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        brand_embeddings = _make_synthetic_brand_embeddings(BRAND_IDS)
        # Reduce one brand to 1 channel
        brand_embeddings["brand_e"] = {"regulatory": brand_embeddings["brand_e"]["regulatory"]}

        scores = compute_brand_scores(brand_embeddings, manifest, "mean_pairwise")

        # brand_e should have None coherence and be last
        brand_e_score = next(s for s in scores if s.brand_id == "brand_e")
        assert brand_e_score.market_coherence is None

    def test_coherence_order_follows_spread(self):
        """Brands with lower spread (more coherent) should rank higher."""
        rng = np.random.default_rng(42)
        # Explicitly set coherence levels
        coherence_levels = {
            "brand_a": 0.01,  # most coherent
            "brand_b": 0.05,
            "brand_c": 0.10,
            "brand_d": 0.20,
            "brand_e": 0.50,  # least coherent
        }
        brand_embeddings = _make_synthetic_brand_embeddings(
            BRAND_IDS, coherence_levels=coherence_levels, rng=rng
        )
        manifest = {
            "brands": [
                {"brand_id": bid, "brand_name": bid} for bid in BRAND_IDS
            ]
        }

        scores = compute_brand_scores(brand_embeddings, manifest, "mean_pairwise")

        # brand_a should be rank 1 (most coherent)
        brand_a_score = next(s for s in scores if s.brand_id == "brand_a")
        assert brand_a_score.metric_rank == 1


# ---------------------------------------------------------------------------
# Agreement computation tests
# ---------------------------------------------------------------------------


class TestComputeMetricExpertTaus:
    """Tests for compute_metric_expert_taus."""

    def test_perfect_agreement(self):
        """Perfect agreement between metric and experts -> tau = 1.0."""
        metric_ranking = BRAND_IDS
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            )
            for i in range(3)
        ]

        taus = compute_metric_expert_taus(metric_ranking, expert_rankings)

        assert len(taus) == 3
        for tau in taus:
            assert abs(tau - 1.0) < 1e-10

    def test_reversed_agreement(self):
        """Reversed ranking -> tau = -1.0."""
        metric_ranking = BRAND_IDS
        reversed_ids = list(reversed(BRAND_IDS))
        expert_rankings = [
            ExpertRanking(
                expert_id="expert_1",
                ranking=reversed_ids,
                rank_map={bid: rank + 1 for rank, bid in enumerate(reversed_ids)},
            ),
        ]

        taus = compute_metric_expert_taus(metric_ranking, expert_rankings)

        assert len(taus) == 1
        assert abs(taus[0] - (-1.0)) < 1e-10


class TestComputeInterExpertTaus:
    """Tests for compute_inter_expert_taus."""

    def test_perfect_inter_expert_agreement(self):
        """All experts agree -> all inter-expert taus = 1.0."""
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            )
            for i in range(3)
        ]

        taus = compute_inter_expert_taus(expert_rankings)

        # C(3,2) = 3 pairs
        assert len(taus) == 3
        for tau in taus:
            assert abs(tau - 1.0) < 1e-10

    def test_inter_expert_count(self):
        """Correct number of inter-expert pairs."""
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            )
            for i in range(3)
        ]

        taus = compute_inter_expert_taus(expert_rankings)

        # C(3,2) = 3
        assert len(taus) == 3


# ---------------------------------------------------------------------------
# Pairwise comparison tests
# ---------------------------------------------------------------------------


class TestComputePairwiseComparisons:
    """Tests for compute_pairwise_comparisons."""

    def test_perfect_agreement(self):
        """Perfect agreement -> all comparisons agree."""
        metric_ranking = BRAND_IDS
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            )
            for i in range(3)
        ]

        comparisons = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        # C(5,2) = 10 pairs
        assert len(comparisons) == 10
        assert all(c.agrees_with_majority for c in comparisons)

    def test_reversed_agreement(self):
        """Fully reversed -> no comparisons agree."""
        metric_ranking = BRAND_IDS
        reversed_ids = list(reversed(BRAND_IDS))
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=reversed_ids,
                rank_map={bid: rank + 1 for rank, bid in enumerate(reversed_ids)},
            )
            for i in range(3)
        ]

        comparisons = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        assert len(comparisons) == 10
        assert all(not c.agrees_with_majority for c in comparisons)

    def test_pairwise_count_for_5_brands(self):
        """5 brands -> C(5,2) = 10 pairwise comparisons."""
        metric_ranking = BRAND_IDS
        expert_rankings = [
            ExpertRanking(
                expert_id="expert_1",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            ),
        ]

        comparisons = compute_pairwise_comparisons(metric_ranking, expert_rankings)
        assert len(comparisons) == 10


# ---------------------------------------------------------------------------
# Results building tests
# ---------------------------------------------------------------------------


class TestBuildResults:
    """Tests for build_results."""

    def _make_scores(self) -> list[BrandScore]:
        """Create mock brand scores."""
        return [
            BrandScore(
                brand_id=bid,
                brand_name=bid,
                market_coherence=1.0 - i * 0.1,
                brand_coherence=0.9 - i * 0.1,
                metric_rank=i + 1,
            )
            for i, bid in enumerate(BRAND_IDS)
        ]

    def test_pass_verdict(self):
        """Both criteria pass -> PASS."""
        scores = self._make_scores()
        metric_ranking = BRAND_IDS
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            )
            for i in range(3)
        ]
        metric_expert_taus = [1.0, 1.0, 1.0]
        inter_expert_taus = [1.0, 1.0, 1.0]
        pairwise = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        results = build_results(
            scores, metric_ranking, expert_rankings,
            metric_expert_taus, inter_expert_taus, pairwise,
        )

        assert results.verdict == "PASS"
        assert results.tau_criterion_pass is True
        assert results.pairwise_criterion_pass is True

    def test_fail_verdict(self):
        """Both criteria fail -> FAIL."""
        scores = self._make_scores()
        metric_ranking = BRAND_IDS
        reversed_ids = list(reversed(BRAND_IDS))
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=reversed_ids,
                rank_map={bid: rank + 1 for rank, bid in enumerate(reversed_ids)},
            )
            for i in range(3)
        ]
        metric_expert_taus = [-1.0, -1.0, -1.0]
        inter_expert_taus = [1.0, 1.0, 1.0]
        pairwise = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        results = build_results(
            scores, metric_ranking, expert_rankings,
            metric_expert_taus, inter_expert_taus, pairwise,
        )

        assert results.verdict == "FAIL"
        assert results.tau_criterion_pass is False
        assert results.pairwise_criterion_pass is False

    def test_partial_pass_verdict(self):
        """One criterion passes, one fails -> PARTIAL_PASS."""
        scores = self._make_scores()
        metric_ranking = BRAND_IDS

        # Experts mostly agree but with some disagreement
        expert_rankings = [
            ExpertRanking(
                expert_id="expert_1",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            ),
            ExpertRanking(
                expert_id="expert_2",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            ),
            ExpertRanking(
                expert_id="expert_3",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            ),
        ]

        # Force tau criterion to fail but pairwise to pass
        metric_expert_taus = [0.3, 0.3, 0.3]
        inter_expert_taus = [0.9, 0.9, 0.9]
        pairwise = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        results = build_results(
            scores, metric_ranking, expert_rankings,
            metric_expert_taus, inter_expert_taus, pairwise,
        )

        assert results.verdict == "PARTIAL_PASS"


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestResultsToDict:
    """Tests for results_to_dict."""

    def test_serialization_roundtrip(self):
        """Results can be serialized to JSON."""
        scores = [
            BrandScore(
                brand_id=bid,
                brand_name=bid,
                market_coherence=1.0 - i * 0.1,
                brand_coherence=0.9 - i * 0.1,
                metric_rank=i + 1,
            )
            for i, bid in enumerate(BRAND_IDS)
        ]
        metric_ranking = BRAND_IDS
        expert_rankings = [
            ExpertRanking(
                expert_id=f"expert_{i}",
                ranking=list(BRAND_IDS),
                rank_map={bid: rank + 1 for rank, bid in enumerate(BRAND_IDS)},
            )
            for i in range(3)
        ]
        metric_expert_taus = [1.0, 1.0, 1.0]
        inter_expert_taus = [1.0, 1.0, 1.0]
        pairwise = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        results = build_results(
            scores, metric_ranking, expert_rankings,
            metric_expert_taus, inter_expert_taus, pairwise,
        )
        d = results_to_dict(results)

        json_str = json.dumps(d)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert "brand_scores" in parsed
        assert "metric_ranking" in parsed
        assert "expert_rankings" in parsed
        assert "agreement" in parsed
        assert "pairwise_comparisons" in parsed
        assert "verdict" in parsed

    def test_numpy_types_serialized(self):
        """Numpy types are handled by _json_default."""
        scores = [
            BrandScore(
                brand_id="brand_a",
                brand_name="Brand A",
                market_coherence=np.float64(0.9),
                brand_coherence=np.float64(0.85),
                metric_rank=np.int64(1),
            ),
        ]
        metric_ranking = ["brand_a"]
        expert_rankings = [
            ExpertRanking(
                expert_id="expert_1",
                ranking=["brand_a"],
                rank_map={"brand_a": 1},
            ),
        ]

        results = build_results(
            scores, metric_ranking, expert_rankings,
            [np.float64(1.0)], [], [],
        )
        d = results_to_dict(results)
        from coherence.experiment_5 import _json_default
        json_str = json.dumps(d, default=_json_default)
        assert json_str is not None


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests using synthetic data end-to-end."""

    def test_full_pipeline_pass(self, tmp_path: Path):
        """Full pipeline with perfect agreement -> PASS."""
        _make_competitive_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        # Create embeddings with clear ordering
        coherence_levels = {
            "brand_a": 0.01,
            "brand_b": 0.05,
            "brand_c": 0.10,
            "brand_d": 0.20,
            "brand_e": 0.50,
        }
        brand_embeddings = _make_synthetic_brand_embeddings(
            BRAND_IDS, coherence_levels=coherence_levels
        )

        # Compute scores
        scores = compute_brand_scores(brand_embeddings, manifest, "mean_pairwise")
        metric_ranking = [s.brand_id for s in scores]

        # Generate form
        form_path = generate_expert_form(manifest, tmp_path / "exp5")
        assert form_path.exists()

        # Create expert rankings matching metric order
        _make_expert_rankings(
            tmp_path / "exp5", metric_ranking, agreement_level="perfect"
        )
        expert_rankings = load_expert_rankings(tmp_path / "exp5")

        # Compute agreements
        metric_expert_taus = compute_metric_expert_taus(
            metric_ranking, expert_rankings
        )
        inter_expert_taus = compute_inter_expert_taus(expert_rankings)
        pairwise = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        # Build and verify
        results = build_results(
            scores, metric_ranking, expert_rankings,
            metric_expert_taus, inter_expert_taus, pairwise,
        )

        assert results.verdict == "PASS"
        assert results.tau_criterion_pass is True
        assert results.pairwise_criterion_pass is True
        assert results.n_pairwise_agree == 10
        assert results.n_pairwise_total == 10

    def test_full_pipeline_results_serializable(self, tmp_path: Path):
        """Full pipeline results can be serialized to JSON."""
        _make_competitive_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        brand_embeddings = _make_synthetic_brand_embeddings(BRAND_IDS)
        scores = compute_brand_scores(brand_embeddings, manifest, "mean_pairwise")
        metric_ranking = [s.brand_id for s in scores]

        _make_expert_rankings(tmp_path / "exp5", metric_ranking)
        expert_rankings = load_expert_rankings(tmp_path / "exp5")

        metric_expert_taus = compute_metric_expert_taus(metric_ranking, expert_rankings)
        inter_expert_taus = compute_inter_expert_taus(expert_rankings)
        pairwise = compute_pairwise_comparisons(metric_ranking, expert_rankings)

        results = build_results(
            scores, metric_ranking, expert_rankings,
            metric_expert_taus, inter_expert_taus, pairwise,
        )
        d = results_to_dict(results)
        from coherence.experiment_5 import _json_default
        json_str = json.dumps(d, default=_json_default)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert "verdict" in parsed
        assert parsed["verdict"]["overall"] in ("PASS", "PARTIAL_PASS", "FAIL")
