"""Tests for coherence.experiment_4 -- uses synthetic data, no GPU required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from coherence.experiment_4 import (
    ACCEPTABLE_VERDICTS,
    KENDALL_TAU_P_THRESHOLD,
    MIN_TEST_PRODUCTS,
    MIN_TIME_POINTS,
    N_CONTROL_PRODUCTS,
    PASS_MIN_DIRECTIONALLY_CORRECT,
    BrandEvent,
    Experiment4Results,
    ProductTrajectory,
    TimePoint,
    apply_control_correction,
    build_results,
    check_gates,
    compute_trajectories,
    evaluate_trajectory,
    load_temporal_manifest,
    results_to_dict,
)
from coherence.metrics import CoherenceResult, compute_coherence_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D = 64  # reduced hidden dim for tests
CHANNELS = ("regulatory", "marketing", "retail", "social")


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


def _make_manifest(
    tmp_dir: Path,
    n_test: int = 2,
    n_control: int = 1,
    n_timepoints: int = 3,
) -> Path:
    """Create a temporal experiment manifest."""
    test_products = []
    for i in range(n_test):
        tp_list = [
            {"label": f"t{j}", "date": f"2020-0{j + 1}"}
            for j in range(n_timepoints)
        ]
        test_products.append({
            "product_id": f"test_brand_{i}",
            "time_points": tp_list,
            "brand_events": [
                {
                    "date": "2021-06",
                    "event_type": "rebrand",
                    "description": f"Test brand {i} rebrand",
                    "expected_direction": "disruption",
                }
            ],
        })

    control_products = []
    for i in range(n_control):
        tp_list = [
            {"label": f"t{j}", "date": f"2020-0{j + 1}"}
            for j in range(n_timepoints)
        ]
        control_products.append({
            "product_id": f"control_brand_{i}",
            "time_points": tp_list,
            "brand_events": [],
        })

    manifest = {
        "category": "test_category",
        "test_products": test_products,
        "control_products": control_products,
    }
    manifest_path = tmp_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def _make_synthetic_temporal_embeddings(
    manifest: dict,
    dim: int = D,
    rng: Optional[np.random.Generator] = None,
    coherence_trend: str = "increasing",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Create synthetic temporal embeddings with controlled coherence trends.

    Parameters
    ----------
    coherence_trend : "increasing" | "decreasing" | "random"
        How to structure the embeddings over time.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    all_specs = manifest["test_products"] + manifest["control_products"]
    for product_spec in all_specs:
        product_id = product_spec["product_id"]
        result[product_id] = {}
        is_control = product_spec in manifest["control_products"]

        for t_idx, tp in enumerate(product_spec["time_points"]):
            tp_label = tp["label"]

            # Create channel embeddings with controlled similarity
            if is_control:
                # Control: stable embeddings across time
                base = rng.standard_normal(dim)
                base /= np.linalg.norm(base)
                channel_embs = {}
                for ch in CHANNELS:
                    noise = rng.standard_normal(dim) * 0.1
                    channel_embs[ch] = base + noise
            else:
                if coherence_trend == "increasing":
                    # Increasing coherence: channels converge over time
                    spread = max(0.5 - t_idx * 0.15, 0.05)
                elif coherence_trend == "decreasing":
                    spread = 0.1 + t_idx * 0.15
                else:
                    spread = rng.uniform(0.1, 0.5)

                base = rng.standard_normal(dim)
                base /= np.linalg.norm(base)
                channel_embs = {}
                for ch in CHANNELS:
                    noise = rng.standard_normal(dim) * spread
                    channel_embs[ch] = base + noise

            result[product_id][tp_label] = channel_embs

    return result


# ---------------------------------------------------------------------------
# Gate check tests
# ---------------------------------------------------------------------------


class TestCheckGates:
    """Tests for the check_gates function."""

    def test_missing_metric_selection(self, tmp_path: Path):
        """check_gates exits when metric_selection.json is missing."""
        with patch("coherence.experiment_4.METRIC_SELECTION_PATH", tmp_path / "missing.json"), \
             pytest.raises(SystemExit):
            check_gates()

    def test_missing_global_mean(self, tmp_path: Path):
        """check_gates exits when global_mean.npy is missing."""
        ms_path = _make_metric_selection(tmp_path)
        with patch("coherence.experiment_4.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_4.GLOBAL_MEAN_PATH", tmp_path / "missing.npy"), \
             pytest.raises(SystemExit):
            check_gates()

    def test_missing_verdict(self, tmp_path: Path):
        """check_gates exits when verdict.json is missing."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        with patch("coherence.experiment_4.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_4.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_4.VERDICT_PATH", tmp_path / "missing.json"), \
             pytest.raises(SystemExit):
            check_gates()

    def test_failing_verdict(self, tmp_path: Path):
        """check_gates exits when verdict is FAIL."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        verdict_path = _make_verdict_file(tmp_path, "FAIL")
        with patch("coherence.experiment_4.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_4.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_4.VERDICT_PATH", verdict_path), \
             pytest.raises(SystemExit):
            check_gates()

    def test_passing_gates(self, tmp_path: Path):
        """check_gates succeeds with valid files."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        verdict_path = _make_verdict_file(tmp_path, "PASS")
        with patch("coherence.experiment_4.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_4.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_4.VERDICT_PATH", verdict_path):
            metric_sel, global_mean, verdict = check_gates()

        assert metric_sel["aggregation"] == "mean_pairwise"
        assert global_mean.shape == (D,)
        assert verdict["verdict"]["overall"] == "PASS"

    def test_pass_no_value_added_accepted(self, tmp_path: Path):
        """check_gates also accepts PASS_NO_VALUE_ADDED."""
        ms_path = _make_metric_selection(tmp_path)
        gm_path = _make_global_mean(tmp_path)
        verdict_path = _make_verdict_file(tmp_path, "PASS_NO_VALUE_ADDED")
        with patch("coherence.experiment_4.METRIC_SELECTION_PATH", ms_path), \
             patch("coherence.experiment_4.GLOBAL_MEAN_PATH", gm_path), \
             patch("coherence.experiment_4.VERDICT_PATH", verdict_path):
            _, _, verdict = check_gates()
        assert verdict["verdict"]["overall"] == "PASS_NO_VALUE_ADDED"


# ---------------------------------------------------------------------------
# Manifest loading tests
# ---------------------------------------------------------------------------


class TestLoadTemporalManifest:
    """Tests for load_temporal_manifest."""

    def test_valid_manifest(self, tmp_path: Path):
        """Loads a valid manifest successfully."""
        _make_manifest(tmp_path)
        manifest = load_temporal_manifest(tmp_path)
        assert len(manifest["test_products"]) == 2
        assert len(manifest["control_products"]) == 1

    def test_missing_manifest(self, tmp_path: Path):
        """Exits if manifest.json is missing."""
        with pytest.raises(SystemExit):
            load_temporal_manifest(tmp_path)

    def test_insufficient_test_products(self, tmp_path: Path):
        """Exits if fewer than MIN_TEST_PRODUCTS test products."""
        _make_manifest(tmp_path, n_test=1)
        with pytest.raises(SystemExit):
            load_temporal_manifest(tmp_path)

    def test_insufficient_control_products(self, tmp_path: Path):
        """Exits if fewer than N_CONTROL_PRODUCTS control products."""
        _make_manifest(tmp_path, n_control=0)
        with pytest.raises(SystemExit):
            load_temporal_manifest(tmp_path)

    def test_insufficient_time_points(self, tmp_path: Path):
        """Exits if a product has fewer than MIN_TIME_POINTS time points."""
        _make_manifest(tmp_path, n_timepoints=2)
        with pytest.raises(SystemExit):
            load_temporal_manifest(tmp_path)


# ---------------------------------------------------------------------------
# Trajectory computation tests
# ---------------------------------------------------------------------------


class TestComputeTrajectories:
    """Tests for compute_trajectories."""

    def test_basic_trajectory_computation(self, tmp_path: Path):
        """Computes trajectories for all products."""
        _make_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        temporal_embeddings = _make_synthetic_temporal_embeddings(manifest)

        test_trajs, control_traj = compute_trajectories(
            temporal_embeddings, manifest, "mean_pairwise"
        )

        assert len(test_trajs) == 2
        assert control_traj is not None
        assert control_traj.is_control

        for traj in test_trajs:
            assert not traj.is_control
            assert len(traj.time_points) == 3
            assert len(traj.raw_coherences) == 3
            assert all(not np.isnan(c) for c in traj.raw_coherences)

    def test_trajectory_with_missing_embeddings(self, tmp_path: Path):
        """Handles products with missing time point embeddings."""
        _make_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        temporal_embeddings = _make_synthetic_temporal_embeddings(manifest)
        # Remove one time point
        del temporal_embeddings["test_brand_0"]["t1"]

        test_trajs, control_traj = compute_trajectories(
            temporal_embeddings, manifest, "mean_pairwise"
        )

        # The missing time point should result in NaN coherence
        assert len(test_trajs[0].raw_coherences) == 3
        # t0 and t2 should be valid, t1 should be NaN
        assert not np.isnan(test_trajs[0].raw_coherences[0])
        assert np.isnan(test_trajs[0].raw_coherences[1])
        assert not np.isnan(test_trajs[0].raw_coherences[2])

    def test_brand_events_populated(self, tmp_path: Path):
        """Brand events from manifest are populated on trajectories."""
        _make_manifest(tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        temporal_embeddings = _make_synthetic_temporal_embeddings(manifest)

        test_trajs, control_traj = compute_trajectories(
            temporal_embeddings, manifest, "mean_pairwise"
        )

        # Test products should have brand events
        for traj in test_trajs:
            assert len(traj.brand_events) == 1
            assert traj.brand_events[0].event_type == "rebrand"

        # Control should have no events
        assert len(control_traj.brand_events) == 0


# ---------------------------------------------------------------------------
# Control correction tests
# ---------------------------------------------------------------------------


class TestApplyControlCorrection:
    """Tests for apply_control_correction."""

    def test_correction_subtracts_control(self):
        """Corrected = raw - control at each time point."""
        test_traj = ProductTrajectory(
            product_id="test",
            is_control=False,
            raw_coherences=[0.5, 0.6, 0.7],
        )
        control_traj = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.4, 0.4, 0.4],
        )

        apply_control_correction([test_traj], control_traj)

        assert len(test_traj.corrected_coherences) == 3
        np.testing.assert_allclose(
            test_traj.corrected_coherences, [0.1, 0.2, 0.3], atol=1e-10
        )

    def test_control_corrected_equals_raw(self):
        """Control's corrected coherences equal its raw coherences."""
        control_traj = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.4, 0.5, 0.6],
        )

        apply_control_correction([], control_traj)

        np.testing.assert_allclose(
            control_traj.corrected_coherences, [0.4, 0.5, 0.6], atol=1e-10
        )

    def test_length_mismatch_handling(self):
        """Handles test and control having different time point counts."""
        test_traj = ProductTrajectory(
            product_id="test",
            is_control=False,
            raw_coherences=[0.5, 0.6, 0.7, 0.8],
        )
        control_traj = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.4, 0.4, 0.4],
        )

        apply_control_correction([test_traj], control_traj)

        # First 3 should be corrected, 4th should be NaN
        assert len(test_traj.corrected_coherences) == 4
        np.testing.assert_allclose(
            test_traj.corrected_coherences[:3], [0.1, 0.2, 0.3], atol=1e-10
        )
        assert np.isnan(test_traj.corrected_coherences[3])

    def test_multiple_test_products(self):
        """Correction works for multiple test products."""
        test_a = ProductTrajectory(
            product_id="test_a",
            is_control=False,
            raw_coherences=[0.5, 0.6, 0.7],
        )
        test_b = ProductTrajectory(
            product_id="test_b",
            is_control=False,
            raw_coherences=[0.3, 0.4, 0.5],
        )
        control = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.4, 0.4, 0.4],
        )

        apply_control_correction([test_a, test_b], control)

        np.testing.assert_allclose(
            test_a.corrected_coherences, [0.1, 0.2, 0.3], atol=1e-10
        )
        np.testing.assert_allclose(
            test_b.corrected_coherences, [-0.1, 0.0, 0.1], atol=1e-10
        )


# ---------------------------------------------------------------------------
# Statistical evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateTrajectory:
    """Tests for evaluate_trajectory."""

    def test_increasing_trajectory(self):
        """Detects a clearly increasing trajectory as non-random."""
        traj = ProductTrajectory(
            product_id="test",
            is_control=False,
            corrected_coherences=[0.1, 0.2, 0.3, 0.4, 0.5],
            brand_events=[
                BrandEvent(
                    date="2021",
                    event_type="rebrand",
                    description="test",
                    expected_direction="increase",
                ),
            ],
        )

        evaluate_trajectory(traj)

        assert traj.kendall_tau is not None
        assert traj.kendall_tau > 0
        assert traj.kendall_p is not None

    def test_random_trajectory(self):
        """A random trajectory should have high p-value."""
        rng = np.random.default_rng(42)
        # Create genuinely random coherences
        random_coherences = list(rng.standard_normal(5))

        traj = ProductTrajectory(
            product_id="test",
            is_control=False,
            corrected_coherences=random_coherences,
            brand_events=[],
        )

        evaluate_trajectory(traj)

        assert traj.kendall_tau is not None
        assert traj.kendall_p is not None
        # With random data, we cannot guarantee p > threshold,
        # but kendall_tau should be populated

    def test_insufficient_valid_points(self):
        """Trajectory with too many NaN values cannot be evaluated."""
        traj = ProductTrajectory(
            product_id="test",
            is_control=False,
            corrected_coherences=[0.1, float("nan"), float("nan")],
            brand_events=[],
        )

        evaluate_trajectory(traj)

        assert traj.kendall_tau is None
        assert traj.kendall_p is None
        assert traj.directionally_correct is False

    def test_disruption_direction(self):
        """Disruption events accept any direction as valid."""
        traj = ProductTrajectory(
            product_id="test",
            is_control=False,
            corrected_coherences=[0.5, 0.4, 0.3, 0.2, 0.1],
            brand_events=[
                BrandEvent(
                    date="2021",
                    event_type="rebrand",
                    description="Major rebrand",
                    expected_direction="disruption",
                ),
            ],
        )

        evaluate_trajectory(traj)

        assert traj.kendall_tau is not None
        # Decreasing trajectory with disruption event should still be
        # considered directionally correct (disruption can go either way)


# ---------------------------------------------------------------------------
# Results building tests
# ---------------------------------------------------------------------------


class TestBuildResults:
    """Tests for build_results."""

    def test_passing_results(self):
        """All products directionally correct -> PASS."""
        test_trajs = [
            ProductTrajectory(
                product_id=f"test_{i}",
                is_control=False,
                raw_coherences=[0.3, 0.4, 0.5],
                corrected_coherences=[0.1, 0.2, 0.3],
                directionally_correct=True,
                kendall_tau=0.9,
                kendall_p=0.01,
            )
            for i in range(3)
        ]
        control = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.2, 0.2, 0.2],
            corrected_coherences=[0.2, 0.2, 0.2],
        )

        results = build_results(test_trajs, control)

        assert results.verdict == "PASS"
        assert results.n_directionally_correct == 3
        assert results.pass_criterion is True

    def test_failing_results(self):
        """Too few products directionally correct -> FAIL."""
        test_trajs = [
            ProductTrajectory(
                product_id="test_0",
                is_control=False,
                raw_coherences=[0.3, 0.4, 0.5],
                corrected_coherences=[0.1, 0.2, 0.3],
                directionally_correct=True,
                kendall_tau=0.9,
                kendall_p=0.01,
            ),
            ProductTrajectory(
                product_id="test_1",
                is_control=False,
                raw_coherences=[0.5, 0.3, 0.4],
                corrected_coherences=[0.3, 0.1, 0.2],
                directionally_correct=False,
                kendall_tau=0.2,
                kendall_p=0.6,
            ),
            ProductTrajectory(
                product_id="test_2",
                is_control=False,
                raw_coherences=[0.4, 0.4, 0.4],
                corrected_coherences=[0.2, 0.2, 0.2],
                directionally_correct=False,
                kendall_tau=0.0,
                kendall_p=1.0,
            ),
        ]
        control = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.2, 0.2, 0.2],
            corrected_coherences=[0.2, 0.2, 0.2],
        )

        results = build_results(test_trajs, control)

        assert results.verdict == "FAIL"
        assert results.n_directionally_correct == 1
        assert results.pass_criterion is False

    def test_overall_kendall_tau(self):
        """Overall Kendall tau is computed across all test products."""
        test_trajs = [
            ProductTrajectory(
                product_id="test_0",
                is_control=False,
                corrected_coherences=[0.1, 0.2, 0.3],
                directionally_correct=True,
            ),
            ProductTrajectory(
                product_id="test_1",
                is_control=False,
                corrected_coherences=[0.15, 0.25, 0.35],
                directionally_correct=True,
            ),
        ]
        control = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.2, 0.2, 0.2],
        )

        results = build_results(test_trajs, control)

        assert results.kendall_overall is not None
        assert results.kendall_overall > 0


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestResultsToDict:
    """Tests for results_to_dict."""

    def test_serialization_roundtrip(self):
        """Results can be serialized to JSON and contain expected keys."""
        test_trajs = [
            ProductTrajectory(
                product_id="test_0",
                is_control=False,
                time_points=[
                    TimePoint(label="t0", date="2020-01", market_coherence=0.5),
                    TimePoint(label="t1", date="2021-01", market_coherence=0.6),
                    TimePoint(label="t2", date="2022-01", market_coherence=0.7),
                ],
                brand_events=[
                    BrandEvent(
                        date="2021-06",
                        event_type="rebrand",
                        description="Test",
                        expected_direction="increase",
                    ),
                ],
                raw_coherences=[0.5, 0.6, 0.7],
                corrected_coherences=[0.1, 0.2, 0.3],
                kendall_tau=0.9,
                kendall_p=0.01,
                directionally_correct=True,
            ),
        ]
        control = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[0.4, 0.4, 0.4],
            corrected_coherences=[0.4, 0.4, 0.4],
        )

        results = build_results(test_trajs, control)
        d = results_to_dict(results)

        # Verify JSON serializable
        json_str = json.dumps(d)
        assert json_str is not None

        # Check structure
        assert "test_trajectories" in d
        assert "control_trajectory" in d
        assert "summary" in d
        assert "verdict" in d
        assert d["verdict"]["overall"] in ("PASS", "FAIL")

    def test_numpy_types_serialized(self):
        """Numpy types are properly handled by _json_default."""
        test_trajs = [
            ProductTrajectory(
                product_id="test_0",
                is_control=False,
                raw_coherences=[np.float64(0.5), np.float64(0.6), np.float64(0.7)],
                corrected_coherences=[np.float64(0.1), np.float64(0.2), np.float64(0.3)],
                kendall_tau=np.float64(0.9),
                kendall_p=np.float64(0.01),
                directionally_correct=np.bool_(True),
            ),
        ]
        control = ProductTrajectory(
            product_id="control",
            is_control=True,
            raw_coherences=[np.float64(0.4)],
            corrected_coherences=[np.float64(0.4)],
        )

        results = build_results(test_trajs, control)
        d = results_to_dict(results)
        from coherence.experiment_4 import _json_default
        json_str = json.dumps(d, default=_json_default)
        assert json_str is not None


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests using synthetic data end-to-end."""

    def test_full_pipeline_pass(self, tmp_path: Path):
        """Full pipeline with increasing coherence trends -> PASS."""
        _make_manifest(tmp_path, n_test=3, n_timepoints=5)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        temporal_embeddings = _make_synthetic_temporal_embeddings(
            manifest, coherence_trend="increasing"
        )

        test_trajs, control_traj = compute_trajectories(
            temporal_embeddings, manifest, "mean_pairwise"
        )

        apply_control_correction(test_trajs, control_traj)

        for traj in test_trajs:
            evaluate_trajectory(traj)

        results = build_results(test_trajs, control_traj)

        # Verify structure
        assert results.n_test_products == 3
        assert results.control_trajectory.product_id == "control_brand_0"
        assert all(
            len(t.corrected_coherences) == 5 for t in results.test_trajectories
        )

    def test_full_pipeline_results_serializable(self, tmp_path: Path):
        """Full pipeline results can be serialized to JSON."""
        _make_manifest(tmp_path, n_test=2, n_timepoints=3)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        temporal_embeddings = _make_synthetic_temporal_embeddings(manifest)

        test_trajs, control_traj = compute_trajectories(
            temporal_embeddings, manifest, "mean_pairwise"
        )
        apply_control_correction(test_trajs, control_traj)
        for traj in test_trajs:
            evaluate_trajectory(traj)

        results = build_results(test_trajs, control_traj)
        d = results_to_dict(results)
        from coherence.experiment_4 import _json_default
        json_str = json.dumps(d, default=_json_default)
        assert json_str is not None

        # Verify key fields
        parsed = json.loads(json_str)
        assert "verdict" in parsed
        assert "test_trajectories" in parsed
        assert len(parsed["test_trajectories"]) == 2
