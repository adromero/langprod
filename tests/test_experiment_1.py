"""Tests for coherence.experiment_1 -- uses mock data, no GPU required."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy import stats

from coherence.experiment_1 import (
    AUC_FUNDAMENTAL_FAILURE,
    AUC_MARGINAL,
    COHENS_D_PASS,
    COHENS_D_SUGGESTIVE,
    DELONG_P_THRESHOLD,
    MANN_WHITNEY_P_THRESHOLD,
    MAX_MISCLASSIFICATIONS,
    ROC_AUC_THRESHOLD,
    _auc_variance_components,
    _compute_midrank,
    _interpret_d,
    _json_default,
    build_verdict,
    check_gates,
    collect_channel_texts,
    compute_baseline_scores,
    compute_cohens_d,
    compute_product_coherences,
    compute_roc_auc,
    delong_test,
    load_and_prepare_documents,
    rank_biserial_correlation,
    roc_auc_score_safe,
    run_analysis,
)
from coherence.ingest import RealDocument
from coherence.metrics import CoherenceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 64  # reduced hidden dim for tests


def _make_metric_selection(tmp_dir: Path) -> dict:
    """Create and save a mock metric_selection.json."""
    metric = {
        "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "layer_hdf5_index": 31,
        "layer_transformer": 30,
        "correction": "mean_centering",
        "aggregation": "mean_pairwise",
        "distance": "cosine",
        "vocab_narrowness_flag": False,
        "effect_size_d": 0.85,
        "p_value": 0.001,
        "status": "locked",
    }
    metric_path = tmp_dir / "metric_selection.json"
    with open(metric_path, "w") as f:
        json.dump(metric, f)
    return metric


def _make_global_mean(tmp_dir: Path, dim: int = D) -> np.ndarray:
    """Create and save a mock global_mean.npy."""
    rng = np.random.default_rng(42)
    gm = rng.standard_normal(dim)
    np.save(tmp_dir / "global_mean.npy", gm)
    return gm


def _make_mock_documents(
    n_consistent: int = 10,
    n_inconsistent: int = 10,
    channels: tuple[str, ...] = ("regulatory", "marketing", "retail"),
) -> tuple[dict[str, list[RealDocument]], dict[str, bool]]:
    """Create mock product documents and labels."""
    product_docs: dict[str, list[RealDocument]] = {}
    product_labels: dict[str, bool] = {}

    for i in range(n_consistent):
        pid = f"consistent_{i:03d}"
        product_labels[pid] = True
        docs = []
        for ch in channels:
            docs.append(
                RealDocument(
                    product_id=pid,
                    channel=ch,
                    text=f"This is {ch} text for consistent product {i}. "
                    f"The product is well-designed and has many features. "
                    f"Quality is excellent across all channels.",
                )
            )
        product_docs[pid] = docs

    for i in range(n_inconsistent):
        pid = f"inconsistent_{i:03d}"
        product_labels[pid] = False
        docs = []
        for ch in channels:
            docs.append(
                RealDocument(
                    product_id=pid,
                    channel=ch,
                    text=f"This is {ch} text for inconsistent product {i}. "
                    f"Different channels say very different things about this "
                    f"product and there is no unified messaging.",
                )
            )
        product_docs[pid] = docs

    return product_docs, product_labels


def _make_mock_coherence_results(
    product_labels: dict[str, bool],
    rng: np.random.Generator,
    separation: float = 2.0,
) -> dict[str, CoherenceResult]:
    """Create mock coherence results with controlled separation between groups."""
    results: dict[str, CoherenceResult] = {}
    for pid, is_consistent in product_labels.items():
        if is_consistent:
            score = 0.85 + rng.normal(0, 0.05)
        else:
            score = 0.85 - separation * 0.10 + rng.normal(0, 0.05)
        results[pid] = CoherenceResult(
            brand_coherence=score * 0.95,
            market_coherence=float(np.clip(score, 0.0, 1.0)),
            method="mean_pairwise",
        )
    return results


# ---------------------------------------------------------------------------
# Tests: Gate checks
# ---------------------------------------------------------------------------


class TestCheckGates:
    """Gate checks for Experiment 0 dependencies."""

    def test_both_files_present(self, tmp_path: Path) -> None:
        exp0_dir = tmp_path / "exp0"
        exp0_dir.mkdir(parents=True)
        _make_metric_selection(exp0_dir)
        _make_global_mean(exp0_dir)

        with (
            patch("coherence.experiment_1.METRIC_SELECTION_PATH", exp0_dir / "metric_selection.json"),
            patch("coherence.experiment_1.GLOBAL_MEAN_PATH", exp0_dir / "global_mean.npy"),
        ):
            metric, gm = check_gates()

        assert metric["aggregation"] == "mean_pairwise"
        assert metric["layer_hdf5_index"] == 31
        assert gm.shape == (D,)

    def test_missing_metric_selection(self, tmp_path: Path) -> None:
        exp0_dir = tmp_path / "exp0"
        exp0_dir.mkdir(parents=True)
        _make_global_mean(exp0_dir)

        with (
            patch("coherence.experiment_1.METRIC_SELECTION_PATH", exp0_dir / "metric_selection.json"),
            patch("coherence.experiment_1.GLOBAL_MEAN_PATH", exp0_dir / "global_mean.npy"),
            pytest.raises(SystemExit),
        ):
            check_gates()

    def test_missing_global_mean(self, tmp_path: Path) -> None:
        exp0_dir = tmp_path / "exp0"
        exp0_dir.mkdir(parents=True)
        _make_metric_selection(exp0_dir)

        with (
            patch("coherence.experiment_1.METRIC_SELECTION_PATH", exp0_dir / "metric_selection.json"),
            patch("coherence.experiment_1.GLOBAL_MEAN_PATH", exp0_dir / "global_mean.npy"),
            pytest.raises(SystemExit),
        ):
            check_gates()


# ---------------------------------------------------------------------------
# Tests: Cohen's d
# ---------------------------------------------------------------------------


class TestCohensD:
    """Cohen's d computation."""

    def test_identical_distributions(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = compute_cohens_d(a, a.copy())
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_separated_distributions(self) -> None:
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([0.0, 1.0, 2.0])
        d = compute_cohens_d(a, b)
        assert d > 0
        assert d == pytest.approx(10.0, abs=0.01)

    def test_negative_d(self) -> None:
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([10.0, 11.0, 12.0])
        d = compute_cohens_d(a, b)
        assert d < 0

    def test_zero_variance(self) -> None:
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        d = compute_cohens_d(a, b)
        assert d == 0.0


# ---------------------------------------------------------------------------
# Tests: rank-biserial correlation
# ---------------------------------------------------------------------------


class TestRankBiserial:
    """Rank-biserial correlation from Mann-Whitney U."""

    def test_perfect_separation(self) -> None:
        # When all of group A > all of group B, U = 0, r = 1
        r = rank_biserial_correlation(0.0, 5, 5)
        assert r == pytest.approx(1.0)

    def test_no_separation(self) -> None:
        # When U = n1*n2/2 (completely mixed), r = 0
        r = rank_biserial_correlation(12.5, 5, 5)
        assert r == pytest.approx(0.0)

    def test_inverse_separation(self) -> None:
        # U = n1*n2, r = -1
        r = rank_biserial_correlation(25.0, 5, 5)
        assert r == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Tests: ROC AUC
# ---------------------------------------------------------------------------


class TestROCAUC:
    """ROC AUC and misclassification computation."""

    def test_perfect_classification(self) -> None:
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        auc, misclass, threshold = compute_roc_auc(scores, labels)
        assert auc == pytest.approx(1.0)
        assert misclass == 0

    def test_random_classification(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, size=100)
        labels = rng.choice([0, 1], size=100)
        auc, misclass, threshold = compute_roc_auc(scores, labels)
        # Random: AUC should be around 0.5
        assert 0.3 < auc < 0.7

    def test_misclassification_count(self) -> None:
        # One error
        scores = np.array([0.9, 0.8, 0.3, 0.7, 0.1, 0.2, 0.4, 0.05])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        auc, misclass, threshold = compute_roc_auc(scores, labels)
        assert misclass <= 2  # at most 2 misclassified at optimal threshold


# ---------------------------------------------------------------------------
# Tests: safe AUC
# ---------------------------------------------------------------------------


class TestROCAUCSafe:
    """Safe ROC AUC handles edge cases."""

    def test_single_class(self) -> None:
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([1, 1, 1])
        assert roc_auc_score_safe(scores, labels) == 0.5

    def test_two_classes(self) -> None:
        scores = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        assert roc_auc_score_safe(scores, labels) == 1.0


# ---------------------------------------------------------------------------
# Tests: mid-rank computation
# ---------------------------------------------------------------------------


class TestComputeMidrank:
    """Mid-rank computation for DeLong test."""

    def test_no_ties(self) -> None:
        x = np.array([1.0, 3.0, 2.0])
        ranks = _compute_midrank(x)
        # sorted: 1,2,3 -> ranks: 1,2,3
        assert ranks[0] == pytest.approx(1.0)  # value 1.0 -> rank 1
        assert ranks[1] == pytest.approx(3.0)  # value 3.0 -> rank 3
        assert ranks[2] == pytest.approx(2.0)  # value 2.0 -> rank 2

    def test_with_ties(self) -> None:
        x = np.array([1.0, 2.0, 2.0, 3.0])
        ranks = _compute_midrank(x)
        assert ranks[0] == pytest.approx(1.0)  # value 1.0 -> rank 1
        assert ranks[1] == pytest.approx(2.5)  # value 2.0 -> avg of ranks 2,3
        assert ranks[2] == pytest.approx(2.5)  # value 2.0 -> avg of ranks 2,3
        assert ranks[3] == pytest.approx(4.0)  # value 3.0 -> rank 4

    def test_all_tied(self) -> None:
        x = np.array([5.0, 5.0, 5.0])
        ranks = _compute_midrank(x)
        for r in ranks:
            assert r == pytest.approx(2.0)  # avg of 1,2,3


# ---------------------------------------------------------------------------
# Tests: DeLong test
# ---------------------------------------------------------------------------


class TestDeLongTest:
    """DeLong test for comparing AUCs."""

    def test_identical_scores(self) -> None:
        """Two identical methods should give z=0, p=1."""
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        z, p_two, p_one = delong_test(scores, scores, labels)
        assert z == pytest.approx(0.0, abs=1e-10)
        assert p_two == pytest.approx(1.0, abs=0.01)

    def test_clearly_better_method(self) -> None:
        """Method A is clearly better than random baseline."""
        rng = np.random.default_rng(42)
        n = 50
        labels = np.array([1] * (n // 2) + [0] * (n // 2))
        # Good predictor
        scores_a = np.where(labels == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
        # Near-random predictor
        scores_b = rng.uniform(0.3, 0.7, n)

        z, p_two, p_one = delong_test(scores_a, scores_b, labels)
        # A should be significantly better
        assert z > 0
        assert p_one < 0.05

    def test_degenerate_case(self) -> None:
        """Scores with zero variance difference should not crash."""
        scores_a = np.array([1.0, 0.0])
        scores_b = np.array([1.0, 0.0])
        labels = np.array([1, 0])
        z, p_two, p_one = delong_test(scores_a, scores_b, labels)
        # With only 1 pos and 1 neg, cannot estimate variance; returns defaults
        assert z == pytest.approx(0.0)
        assert p_two == pytest.approx(1.0)
        assert p_one == pytest.approx(0.5)

    def test_minimum_samples(self) -> None:
        """Works with minimum sample sizes (1 pos, 1 neg) without crashing."""
        scores_a = np.array([0.9, 0.1])
        scores_b = np.array([0.6, 0.4])
        labels = np.array([1, 0])
        z, p_two, p_one = delong_test(scores_a, scores_b, labels)
        # With only 1 pos and 1 neg, cannot estimate variance; returns defaults
        assert z == pytest.approx(0.0)
        assert p_one == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: Cohen's d interpretation
# ---------------------------------------------------------------------------


class TestInterpretD:
    """Cohen's d interpretation with power caveats."""

    def test_pass(self) -> None:
        assert _interpret_d(1.2, 0.01) == "pass"

    def test_suggestive_significant(self) -> None:
        assert _interpret_d(0.9, 0.03) == "suggestive_but_significant"

    def test_suggestive_underpowered(self) -> None:
        assert _interpret_d(0.9, 0.10) == "suggestive_underpowered"

    def test_below_threshold(self) -> None:
        assert _interpret_d(0.5, 0.30) == "below_threshold"

    def test_exact_boundary_pass(self) -> None:
        assert _interpret_d(1.0, 0.01) == "pass"

    def test_exact_boundary_suggestive(self) -> None:
        assert _interpret_d(0.8, 0.06) == "suggestive_underpowered"

    def test_exact_boundary_suggestive_sig(self) -> None:
        assert _interpret_d(0.8, 0.04) == "suggestive_but_significant"


# ---------------------------------------------------------------------------
# Tests: build_verdict
# ---------------------------------------------------------------------------


class TestBuildVerdict:
    """Verdict construction logic."""

    def test_all_pass(self) -> None:
        verdict = build_verdict(
            auc=0.92,
            mw_p=0.001,
            d=1.2,
            misclass=1,
            n_total=20,
            baseline_results={"tfidf": {"auc": 0.65}},
            delong_results={"tfidf": {"p_one_tailed": 0.03}},
        )
        assert verdict["overall"] == "PASS"
        assert verdict["criteria"]["auc"]["pass"] is True
        assert verdict["criteria"]["mann_whitney_p"]["pass"] is True
        assert verdict["criteria"]["misclassifications"]["pass"] is True
        assert verdict["criteria"]["delong_value_added"]["pass"] is True

    def test_fail_auc(self) -> None:
        verdict = build_verdict(
            auc=0.60,
            mw_p=0.001,
            d=1.2,
            misclass=1,
            n_total=20,
            baseline_results={},
            delong_results={"tfidf": {"p_one_tailed": 0.03}},
        )
        assert verdict["overall"] == "FAIL"
        assert verdict["criteria"]["auc"]["pass"] is False
        assert any("FUNDAMENTAL FAILURE" in n for n in verdict["notes"])

    def test_fail_marginal_auc(self) -> None:
        verdict = build_verdict(
            auc=0.75,
            mw_p=0.001,
            d=1.2,
            misclass=1,
            n_total=20,
            baseline_results={},
            delong_results={"tfidf": {"p_one_tailed": 0.03}},
        )
        assert verdict["overall"] == "FAIL"
        assert any("MARGINAL" in n for n in verdict["notes"])

    def test_fail_mann_whitney(self) -> None:
        verdict = build_verdict(
            auc=0.90,
            mw_p=0.10,
            d=0.5,
            misclass=1,
            n_total=20,
            baseline_results={},
            delong_results={"tfidf": {"p_one_tailed": 0.03}},
        )
        assert verdict["overall"] == "FAIL"
        assert verdict["criteria"]["mann_whitney_p"]["pass"] is False

    def test_fail_misclassifications(self) -> None:
        verdict = build_verdict(
            auc=0.90,
            mw_p=0.001,
            d=1.2,
            misclass=5,
            n_total=20,
            baseline_results={},
            delong_results={"tfidf": {"p_one_tailed": 0.03}},
        )
        assert verdict["overall"] == "FAIL"
        assert verdict["criteria"]["misclassifications"]["pass"] is False

    def test_pass_no_value_added(self) -> None:
        """Primary criteria pass but DeLong fails."""
        verdict = build_verdict(
            auc=0.92,
            mw_p=0.001,
            d=1.2,
            misclass=1,
            n_total=20,
            baseline_results={"tfidf": {"auc": 0.90}},
            delong_results={"tfidf": {"p_one_tailed": 0.30}},
        )
        assert verdict["overall"] == "PASS_NO_VALUE_ADDED"
        assert any("does not significantly outperform" in n for n in verdict["notes"])

    def test_suggestive_note(self) -> None:
        """Suggestive d with underpowered p produces informative note."""
        verdict = build_verdict(
            auc=0.90,
            mw_p=0.08,  # > 0.05 so MW fails
            d=0.9,
            misclass=1,
            n_total=20,
            baseline_results={},
            delong_results={"tfidf": {"p_one_tailed": 0.03}},
        )
        assert any("underpowered" in n for n in verdict["notes"])

    def test_baseline_matches_auc_note(self) -> None:
        """When baseline matches our AUC and DeLong fails."""
        verdict = build_verdict(
            auc=0.92,
            mw_p=0.001,
            d=1.2,
            misclass=1,
            n_total=20,
            baseline_results={"tfidf": {"auc": 0.91}},
            delong_results={"tfidf": {"p_one_tailed": 0.30}},
        )
        assert any("Consider adopting baseline tfidf" in n for n in verdict["notes"])


# ---------------------------------------------------------------------------
# Tests: compute_product_coherences
# ---------------------------------------------------------------------------


class TestComputeProductCoherences:
    """Coherence score computation from embeddings."""

    def test_basic_computation(self) -> None:
        rng = np.random.default_rng(42)
        product_embeddings = {
            "p1": {
                "regulatory": rng.standard_normal(D),
                "marketing": rng.standard_normal(D),
                "retail": rng.standard_normal(D),
            },
            "p2": {
                "regulatory": rng.standard_normal(D),
                "marketing": rng.standard_normal(D),
            },
        }
        results = compute_product_coherences(product_embeddings, "mean_pairwise")
        assert "p1" in results
        assert "p2" in results
        assert results["p1"].method == "mean_pairwise"
        assert 0.0 <= results["p1"].market_coherence <= 1.0 or results["p1"].market_coherence < 0

    def test_single_channel_skipped(self) -> None:
        rng = np.random.default_rng(42)
        product_embeddings = {
            "p1": {"regulatory": rng.standard_normal(D)},
        }
        results = compute_product_coherences(product_embeddings, "mean_pairwise")
        assert "p1" not in results


# ---------------------------------------------------------------------------
# Tests: collect_channel_texts
# ---------------------------------------------------------------------------


class TestCollectChannelTexts:
    """Channel text collection for baselines."""

    def test_basic(self) -> None:
        docs = {
            "p1": [
                RealDocument(product_id="p1", channel="regulatory", text="reg text"),
                RealDocument(product_id="p1", channel="marketing", text="mkt text"),
            ],
        }
        result = collect_channel_texts(docs)
        assert "p1" in result
        assert result["p1"]["regulatory"] == "reg text"
        assert result["p1"]["marketing"] == "mkt text"

    def test_multiple_docs_per_channel(self) -> None:
        docs = {
            "p1": [
                RealDocument(product_id="p1", channel="regulatory", text="first"),
                RealDocument(product_id="p1", channel="regulatory", text="second"),
            ],
        }
        result = collect_channel_texts(docs)
        assert result["p1"]["regulatory"] == "first second"


# ---------------------------------------------------------------------------
# Tests: run_analysis (integration)
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    """Integration test of the full analysis pipeline with synthetic data."""

    def test_well_separated_groups(self) -> None:
        """With well-separated scores, should produce clear pass."""
        rng = np.random.default_rng(42)
        product_labels = {}
        for i in range(10):
            product_labels[f"consistent_{i:03d}"] = True
        for i in range(10):
            product_labels[f"inconsistent_{i:03d}"] = False

        coherence_results = _make_mock_coherence_results(
            product_labels, rng, separation=3.0
        )

        # Baseline scores (weaker separation)
        tfidf_scores = {}
        sbert_scores = {}
        for pid, is_consistent in product_labels.items():
            if is_consistent:
                tfidf_scores[pid] = 0.70 + rng.normal(0, 0.10)
                sbert_scores[pid] = 0.72 + rng.normal(0, 0.10)
            else:
                tfidf_scores[pid] = 0.55 + rng.normal(0, 0.10)
                sbert_scores[pid] = 0.58 + rng.normal(0, 0.10)

        results = run_analysis(coherence_results, product_labels, tfidf_scores, sbert_scores)

        assert "mann_whitney" in results
        assert "cohens_d" in results
        assert "roc" in results
        assert "baselines" in results
        assert "delong" in results
        assert "spearman_rank_order" in results
        assert "product_details" in results
        assert "verdict" in results

        # With strong separation, AUC should be high
        assert results["roc"]["auc"] > 0.7
        assert results["mann_whitney"]["p_value"] < 0.05
        assert results["cohens_d"]["d"] > 0

    def test_product_details_complete(self) -> None:
        """Every product has a detail entry."""
        rng = np.random.default_rng(42)
        product_labels = {"c1": True, "c2": True, "i1": False, "i2": False}
        coherence_results = _make_mock_coherence_results(
            product_labels, rng, separation=2.0
        )
        tfidf_scores = {pid: 0.5 for pid in product_labels}
        sbert_scores = {pid: 0.5 for pid in product_labels}

        results = run_analysis(
            coherence_results, product_labels, tfidf_scores, sbert_scores
        )

        assert len(results["product_details"]) == 4
        for detail in results["product_details"]:
            assert "product_id" in detail
            assert "label" in detail
            assert "market_coherence" in detail

    def test_missing_baselines_handled(self) -> None:
        """Analysis works when baselines are partially missing."""
        rng = np.random.default_rng(42)
        product_labels = {"c1": True, "c2": True, "i1": False, "i2": False}
        coherence_results = _make_mock_coherence_results(
            product_labels, rng, separation=2.0
        )
        # Empty baselines
        results = run_analysis(coherence_results, product_labels, {}, {})
        assert results["baselines"]["tfidf"]["auc"] is None
        assert results["baselines"]["sbert"]["auc"] is None


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestJsonDefault:
    """JSON serialization helper handles numpy types."""

    def test_numpy_int(self) -> None:
        assert _json_default(np.int64(42)) == 42

    def test_numpy_float(self) -> None:
        assert _json_default(np.float64(3.14)) == pytest.approx(3.14)

    def test_numpy_array(self) -> None:
        result = _json_default(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_numpy_bool(self) -> None:
        assert _json_default(np.bool_(True)) is True

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _json_default(set())


# ---------------------------------------------------------------------------
# Tests: DeLong variance components
# ---------------------------------------------------------------------------


class TestAUCVarianceComponents:
    """AUC variance component extraction for DeLong test."""

    def test_perfect_separation(self) -> None:
        """Perfect separation: mean of positive placement values equals AUC (1.0)."""
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        labels = np.array([1, 1, 1, 0, 0, 0])
        v_pos, v_neg = _auc_variance_components(scores, labels)
        # With perfect separation, mean placement value for positives = AUC = 1.0
        assert np.mean(v_pos) == pytest.approx(1.0, abs=1e-10)
        # Mean placement for negatives = 0.0 (they are all ranked below positives)
        assert np.mean(v_neg) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self) -> None:
        """With n_pos == n_neg, component lengths match."""
        scores = np.array([0.9, 0.6, 0.4, 0.1])
        labels = np.array([1, 1, 0, 0])
        v_pos, v_neg = _auc_variance_components(scores, labels)
        assert len(v_pos) == 2
        assert len(v_neg) == 2


# ---------------------------------------------------------------------------
# Tests: load_and_prepare_documents (mocked filesystem)
# ---------------------------------------------------------------------------


def _noop_truncate(doc):
    """Mock truncate_document that returns the document unchanged."""
    return doc


class TestLoadAndPrepareDocuments:
    """Document loading with mocked filesystem."""

    def test_missing_portfolio_dir(self, tmp_path: Path) -> None:
        with (
            patch("coherence.experiment_1.PORTFOLIO_DIR", tmp_path / "nonexistent"),
            pytest.raises(SystemExit),
        ):
            load_and_prepare_documents()

    def test_missing_manifest(self, tmp_path: Path) -> None:
        portfolio = tmp_path / "portfolio"
        portfolio.mkdir()
        # Create a minimal product directory
        prod_dir = portfolio / "product_001"
        prod_dir.mkdir()
        (prod_dir / "regulatory.txt").write_text("Some regulatory text.")
        (prod_dir / "marketing.txt").write_text("Some marketing text.")
        (prod_dir / "retail.txt").write_text("Some retail text.")

        with (
            patch("coherence.experiment_1.PORTFOLIO_DIR", portfolio),
            patch("coherence.experiment_1.truncate_document", _noop_truncate),
            pytest.raises(SystemExit),
        ):
            load_and_prepare_documents()

    def test_successful_load(self, tmp_path: Path) -> None:
        portfolio = tmp_path / "portfolio"
        portfolio.mkdir()

        # Create two products
        for pid in ["prod_a", "prod_b"]:
            prod_dir = portfolio / pid
            prod_dir.mkdir()
            (prod_dir / "regulatory.txt").write_text(f"Regulatory text for {pid}")
            (prod_dir / "marketing.txt").write_text(f"Marketing text for {pid}")
            (prod_dir / "retail.txt").write_text(f"Retail text for {pid}")

        # Create manifest
        manifest = {"consistent": ["prod_a"], "inconsistent": ["prod_b"]}
        with open(portfolio / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with (
            patch("coherence.experiment_1.PORTFOLIO_DIR", portfolio),
            patch("coherence.experiment_1.truncate_document", _noop_truncate),
        ):
            product_docs, product_labels = load_and_prepare_documents()

        assert len(product_docs) == 2
        assert product_labels["prod_a"] is True
        assert product_labels["prod_b"] is False
        # Each product should have 3 documents (regulatory, marketing, retail)
        assert len(product_docs["prod_a"]) == 3
        assert len(product_docs["prod_b"]) == 3

    def test_unlabeled_product_skipped(self, tmp_path: Path) -> None:
        portfolio = tmp_path / "portfolio"
        portfolio.mkdir()

        for pid in ["prod_a", "prod_b", "prod_c"]:
            prod_dir = portfolio / pid
            prod_dir.mkdir()
            (prod_dir / "regulatory.txt").write_text(f"Text {pid}")
            (prod_dir / "marketing.txt").write_text(f"Text {pid}")
            (prod_dir / "retail.txt").write_text(f"Text {pid}")

        # Only label two of three
        manifest = {"consistent": ["prod_a"], "inconsistent": ["prod_b"]}
        with open(portfolio / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with (
            patch("coherence.experiment_1.PORTFOLIO_DIR", portfolio),
            patch("coherence.experiment_1.truncate_document", _noop_truncate),
        ):
            product_docs, product_labels = load_and_prepare_documents()

        assert "prod_c" not in product_docs
        assert "prod_c" not in product_labels
        assert len(product_docs) == 2
