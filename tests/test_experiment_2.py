"""Tests for coherence.experiment_2 -- uses mock .npz files with synthetic embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from coherence.experiment_2 import (
    CONSISTENT_GAP_THRESHOLD,
    MIN_VECTORS,
    PASS_THRESHOLD,
    ProductAttribution,
    analyze_consistent_products,
    analyze_inconsistent_products,
    build_results,
    check_gates,
    load_annotations,
    reconstruct_embeddings,
    results_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 64  # reduced hidden dim for tests
CHANNELS = ("regulatory", "marketing", "retail", "social")


def _make_verdict(tmp_dir: Path, overall: str = "PASS") -> Path:
    """Create a mock verdict.json."""
    verdict_path = tmp_dir / "verdict.json"
    verdict = {
        "verdict": {"overall": overall},
        "roc": {"auc": 0.92},
    }
    with open(verdict_path, "w") as f:
        json.dump(verdict, f)
    return verdict_path


def _make_embeddings(
    tmp_dir: Path,
    n_consistent: int = 10,
    n_inconsistent: int = 10,
    channels: tuple[str, ...] = CHANNELS,
    dim: int = D,
    rng: np.random.Generator | None = None,
    outlier_channel: str = "social",
    outlier_offset: float = 2.0,
) -> tuple[Path, dict[str, dict[str, np.ndarray]]]:
    """Create mock embeddings.npz with controlled outlier structure.

    For inconsistent products, the outlier_channel embedding is shifted away
    from the cluster of other channels. For consistent products, all channels
    are drawn from the same distribution.

    Returns the path to the .npz file and the product_embeddings dict.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    embeddings_path = tmp_dir / "embeddings.npz"
    flat: dict[str, np.ndarray] = {}
    product_embeddings: dict[str, dict[str, np.ndarray]] = {}

    # Consistent products: all channels clustered together
    for i in range(n_consistent):
        pid = f"consistent_{i:03d}"
        base_direction = rng.standard_normal(dim)
        base_direction = base_direction / np.linalg.norm(base_direction)
        product_embeddings[pid] = {}
        for ch in channels:
            noise = rng.standard_normal(dim) * 0.05
            vec = base_direction + noise
            vec = vec / np.linalg.norm(vec)
            flat[f"{pid}/{ch}"] = vec
            product_embeddings[pid][ch] = vec

    # Inconsistent products: one channel is an outlier
    for i in range(n_inconsistent):
        pid = f"inconsistent_{i:03d}"
        base_direction = rng.standard_normal(dim)
        base_direction = base_direction / np.linalg.norm(base_direction)
        product_embeddings[pid] = {}
        for ch in channels:
            if ch == outlier_channel:
                # Push outlier channel in a different direction
                outlier_direction = rng.standard_normal(dim)
                outlier_direction = outlier_direction / np.linalg.norm(outlier_direction)
                vec = outlier_direction * outlier_offset + base_direction * 0.1
                vec = vec / np.linalg.norm(vec)
            else:
                noise = rng.standard_normal(dim) * 0.05
                vec = base_direction + noise
                vec = vec / np.linalg.norm(vec)
            flat[f"{pid}/{ch}"] = vec
            product_embeddings[pid][ch] = vec

    np.savez_compressed(embeddings_path, **flat)
    return embeddings_path, product_embeddings


def _make_labels(
    tmp_dir: Path,
    n_consistent: int = 10,
    n_inconsistent: int = 10,
) -> tuple[Path, dict[str, bool]]:
    """Create mock product_labels.json."""
    labels_path = tmp_dir / "product_labels.json"
    labels: dict[str, bool] = {}
    for i in range(n_consistent):
        labels[f"consistent_{i:03d}"] = True
    for i in range(n_inconsistent):
        labels[f"inconsistent_{i:03d}"] = False
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    return labels_path, labels


def _make_annotations(
    tmp_dir: Path,
    n_inconsistent: int = 10,
    outlier_channel: str = "social",
) -> tuple[Path, dict[str, str]]:
    """Create mock outlier_annotations.json."""
    annotations_path = tmp_dir / "outlier_annotations.json"
    annotations_list = []
    annotations_dict: dict[str, str] = {}
    for i in range(n_inconsistent):
        pid = f"inconsistent_{i:03d}"
        annotations_list.append({
            "product_id": pid,
            "outlier_channel": outlier_channel,
            "notes": f"Test annotation for {pid}",
        })
        annotations_dict[pid] = outlier_channel
    data = {"annotations": annotations_list}
    with open(annotations_path, "w") as f:
        json.dump(data, f)
    return annotations_path, annotations_dict


# ---------------------------------------------------------------------------
# Tests: Gate checks
# ---------------------------------------------------------------------------


class TestCheckGates:
    """Gate checks for Experiment 1 dependencies."""

    def test_all_gates_pass(self, tmp_path: Path) -> None:
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_verdict(exp1_dir, "PASS")
        _make_embeddings(exp1_dir)
        _make_labels(exp1_dir)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
        ):
            verdict, npz, labels = check_gates()

        assert verdict["verdict"]["overall"] == "PASS"
        assert len(npz.files) >= MIN_VECTORS
        assert len(labels) == 20

    def test_pass_no_value_added_accepted(self, tmp_path: Path) -> None:
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_verdict(exp1_dir, "PASS_NO_VALUE_ADDED")
        _make_embeddings(exp1_dir)
        _make_labels(exp1_dir)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
        ):
            verdict, npz, labels = check_gates()

        assert verdict["verdict"]["overall"] == "PASS_NO_VALUE_ADDED"

    def test_missing_verdict(self, tmp_path: Path) -> None:
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_embeddings(exp1_dir)
        _make_labels(exp1_dir)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
            pytest.raises(SystemExit),
        ):
            check_gates()

    def test_fail_verdict_rejected(self, tmp_path: Path) -> None:
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_verdict(exp1_dir, "FAIL")
        _make_embeddings(exp1_dir)
        _make_labels(exp1_dir)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
            pytest.raises(SystemExit),
        ):
            check_gates()

    def test_missing_embeddings(self, tmp_path: Path) -> None:
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_verdict(exp1_dir)
        _make_labels(exp1_dir)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
            pytest.raises(SystemExit),
        ):
            check_gates()

    def test_too_few_vectors(self, tmp_path: Path) -> None:
        """Embeddings with fewer than MIN_VECTORS should be rejected."""
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_verdict(exp1_dir)
        # Only 2 products * 4 channels = 8 vectors < 20
        _make_embeddings(exp1_dir, n_consistent=1, n_inconsistent=1)
        _make_labels(exp1_dir, n_consistent=1, n_inconsistent=1)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
            pytest.raises(SystemExit),
        ):
            check_gates()

    def test_missing_labels(self, tmp_path: Path) -> None:
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)
        _make_verdict(exp1_dir)
        _make_embeddings(exp1_dir)

        with (
            patch("coherence.experiment_2.VERDICT_PATH", exp1_dir / "verdict.json"),
            patch("coherence.experiment_2.EMBEDDINGS_PATH", exp1_dir / "embeddings.npz"),
            patch("coherence.experiment_2.LABELS_PATH", exp1_dir / "product_labels.json"),
            pytest.raises(SystemExit),
        ):
            check_gates()


# ---------------------------------------------------------------------------
# Tests: Embedding reconstruction
# ---------------------------------------------------------------------------


class TestReconstructEmbeddings:
    """Reconstruct product embeddings from flat .npz keys."""

    def test_basic_reconstruction(self, tmp_path: Path) -> None:
        _, expected = _make_embeddings(tmp_path, n_consistent=2, n_inconsistent=2)
        npz = np.load(tmp_path / "embeddings.npz")
        reconstructed = reconstruct_embeddings(npz)

        assert set(reconstructed.keys()) == set(expected.keys())
        for pid in expected:
            assert set(reconstructed[pid].keys()) == set(expected[pid].keys())
            for ch in expected[pid]:
                np.testing.assert_array_almost_equal(
                    reconstructed[pid][ch], expected[pid][ch]
                )

    def test_empty_npz(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.npz"
        np.savez_compressed(path)
        npz = np.load(path)
        result = reconstruct_embeddings(npz)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Load annotations
# ---------------------------------------------------------------------------


class TestLoadAnnotations:
    """Load ground-truth outlier annotations."""

    def test_valid_file(self, tmp_path: Path) -> None:
        _, annotations = _make_annotations(tmp_path, n_inconsistent=5)
        result = load_annotations(tmp_path / "outlier_annotations.json")
        assert result == annotations

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            load_annotations(tmp_path / "nonexistent.json")

    def test_missing_annotations_key(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.json"
        with open(bad_path, "w") as f:
            json.dump({"something_else": []}, f)
        with pytest.raises(SystemExit):
            load_annotations(bad_path)

    def test_incomplete_entries_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        data = {
            "annotations": [
                {"product_id": "p1", "outlier_channel": "social"},
                {"product_id": "p2"},  # missing outlier_channel
                {"outlier_channel": "marketing"},  # missing product_id
                {"product_id": "p3", "outlier_channel": "retail", "notes": "ok"},
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f)

        result = load_annotations(path)
        assert set(result.keys()) == {"p1", "p3"}
        assert result["p1"] == "social"
        assert result["p3"] == "retail"


# ---------------------------------------------------------------------------
# Tests: Inconsistent product analysis
# ---------------------------------------------------------------------------


class TestAnalyzeInconsistentProducts:
    """Outlier identification for inconsistent products."""

    def test_correct_outlier_identified(self, tmp_path: Path) -> None:
        """With a clear outlier, the correct channel should be identified."""
        rng = np.random.default_rng(42)
        outlier_ch = "social"

        _, product_embeddings = _make_embeddings(
            tmp_path,
            n_consistent=0,
            n_inconsistent=5,
            outlier_channel=outlier_ch,
            outlier_offset=3.0,
            rng=rng,
        )
        labels = {f"inconsistent_{i:03d}": False for i in range(5)}
        annotations = {f"inconsistent_{i:03d}": outlier_ch for i in range(5)}

        results = analyze_inconsistent_products(product_embeddings, labels, annotations)

        assert len(results) == 5
        n_correct = sum(1 for r in results if r.correct is True)
        # With strong offset, most should be correct
        assert n_correct >= 3

    def test_missing_product_skipped(self) -> None:
        """Products not in embeddings are skipped."""
        labels = {"missing_001": False}
        annotations = {"missing_001": "social"}
        results = analyze_inconsistent_products({}, labels, annotations)
        assert len(results) == 0

    def test_missing_ground_truth(self) -> None:
        """Products without annotation have correct=None."""
        rng = np.random.default_rng(42)
        dim = D
        pid = "inconsistent_000"
        embs = {
            pid: {
                "regulatory": rng.standard_normal(dim),
                "marketing": rng.standard_normal(dim),
                "retail": rng.standard_normal(dim),
            }
        }
        labels = {pid: False}
        annotations: dict[str, str] = {}  # empty

        results = analyze_inconsistent_products(embs, labels, annotations)
        assert len(results) == 1
        assert results[0].correct is None

    def test_too_few_channels_skipped(self) -> None:
        """Products with < 3 channels are skipped."""
        rng = np.random.default_rng(42)
        pid = "inconsistent_000"
        embs = {
            pid: {
                "regulatory": rng.standard_normal(D),
                "marketing": rng.standard_normal(D),
            }
        }
        labels = {pid: False}
        annotations = {pid: "regulatory"}

        results = analyze_inconsistent_products(embs, labels, annotations)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: Consistent product analysis
# ---------------------------------------------------------------------------


class TestAnalyzeConsistentProducts:
    """Outlier detection on consistent products (secondary check)."""

    def test_consistent_products_low_gap(self, tmp_path: Path) -> None:
        """Consistent products should have low outlier gaps."""
        rng = np.random.default_rng(42)
        _, product_embeddings = _make_embeddings(
            tmp_path,
            n_consistent=5,
            n_inconsistent=0,
            rng=rng,
        )
        labels = {f"consistent_{i:03d}": True for i in range(5)}

        results = analyze_consistent_products(product_embeddings, labels)

        assert len(results) == 5
        # With consistent products (tight clustering), most gaps should be small
        n_low_gap = sum(1 for r in results if r.gap < CONSISTENT_GAP_THRESHOLD)
        assert n_low_gap >= 3  # most should have low gap

    def test_missing_product_skipped(self) -> None:
        labels = {"missing_001": True}
        results = analyze_consistent_products({}, labels)
        assert len(results) == 0

    def test_too_few_channels_skipped(self) -> None:
        rng = np.random.default_rng(42)
        pid = "consistent_000"
        embs = {
            pid: {
                "regulatory": rng.standard_normal(D),
                "marketing": rng.standard_normal(D),
            }
        }
        labels = {pid: True}

        results = analyze_consistent_products(embs, labels)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: Build results
# ---------------------------------------------------------------------------


class TestBuildResults:
    """Verdict construction logic."""

    def _make_inconsistent_results(
        self, n: int = 10, n_correct: int = 8
    ) -> list[ProductAttribution]:
        results = []
        for i in range(n):
            results.append(
                ProductAttribution(
                    product_id=f"inconsistent_{i:03d}",
                    outlier_channel="social",
                    mean_similarities={
                        "regulatory": 0.9, "marketing": 0.85,
                        "retail": 0.88, "social": 0.5,
                    },
                    gap=0.35,
                    ground_truth_outlier="social",
                    correct=i < n_correct,
                )
            )
        return results

    def _make_consistent_results(
        self, n: int = 10, n_strong: int = 2
    ) -> list[ProductAttribution]:
        results = []
        for i in range(n):
            gap = 0.10 if i < n_strong else 0.02
            results.append(
                ProductAttribution(
                    product_id=f"consistent_{i:03d}",
                    outlier_channel="marketing",
                    mean_similarities={
                        "regulatory": 0.95, "marketing": 0.93,
                        "retail": 0.94, "social": 0.94,
                    },
                    gap=gap,
                )
            )
        return results

    def test_pass(self) -> None:
        inc = self._make_inconsistent_results(n=10, n_correct=8)
        con = self._make_consistent_results(n=10, n_strong=2)
        results = build_results(inc, con)
        assert results.verdict == "PASS"
        assert results.n_correct == 8
        assert results.pass_criterion is True
        assert results.accuracy == pytest.approx(0.8)

    def test_fail_too_few_correct(self) -> None:
        inc = self._make_inconsistent_results(n=10, n_correct=4)
        con = self._make_consistent_results(n=10, n_strong=0)
        results = build_results(inc, con)
        assert results.verdict == "FAIL"
        assert results.pass_criterion is False
        assert any("Only 4/10" in n for n in results.notes)

    def test_exactly_at_threshold(self) -> None:
        inc = self._make_inconsistent_results(n=10, n_correct=PASS_THRESHOLD)
        con = self._make_consistent_results(n=10, n_strong=0)
        results = build_results(inc, con)
        assert results.verdict == "PASS"

    def test_consistent_strong_outlier_warning(self) -> None:
        inc = self._make_inconsistent_results(n=10, n_correct=8)
        con = self._make_consistent_results(n=10, n_strong=7)
        results = build_results(inc, con)
        assert any("WARNING" in n for n in results.notes)

    def test_missing_ground_truth_noted(self) -> None:
        inc = [
            ProductAttribution(
                product_id="p1",
                outlier_channel="social",
                mean_similarities={"a": 0.9, "b": 0.8, "social": 0.5},
                gap=0.3,
                ground_truth_outlier=None,
                correct=None,
            )
        ]
        results = build_results(inc, [])
        assert any("missing ground-truth" in n for n in results.notes)

    def test_empty_inputs(self) -> None:
        results = build_results([], [])
        assert results.verdict == "FAIL"
        assert results.n_correct == 0
        assert results.accuracy == 0.0


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestResultsToDict:
    """Results serialization to dict."""

    def test_basic_serialization(self) -> None:
        inc = [
            ProductAttribution(
                product_id="p1",
                outlier_channel="social",
                mean_similarities={"regulatory": 0.9, "social": 0.5},
                gap=0.4,
                ground_truth_outlier="social",
                correct=True,
            )
        ]
        con = [
            ProductAttribution(
                product_id="p2",
                outlier_channel="marketing",
                mean_similarities={"regulatory": 0.95, "marketing": 0.93},
                gap=0.02,
            )
        ]
        results = build_results(inc, con)
        d = results_to_dict(results)

        assert "inconsistent_analysis" in d
        assert "consistent_analysis" in d
        assert "verdict" in d
        assert "notes" in d

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_numpy_types_handled(self) -> None:
        inc = [
            ProductAttribution(
                product_id="p1",
                outlier_channel="social",
                mean_similarities={"a": np.float64(0.9), "b": np.float64(0.5)},
                gap=np.float64(0.4),
                ground_truth_outlier="social",
                correct=True,
            )
        ]
        results = build_results(inc, [])
        d = results_to_dict(results)
        # Should be JSON-serializable even with numpy types in mean_similarities
        from coherence.experiment_2 import _json_default

        json_str = json.dumps(d, default=_json_default)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# Tests: Integration (end-to-end with synthetic data)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end test with synthetic embeddings."""

    def test_full_pipeline_passes(self, tmp_path: Path) -> None:
        """With clearly separated outliers, pipeline should pass."""
        rng = np.random.default_rng(42)
        outlier_ch = "social"

        # Create all necessary files
        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)

        _make_verdict(exp1_dir)
        emb_path, product_embeddings = _make_embeddings(
            exp1_dir,
            n_consistent=10,
            n_inconsistent=10,
            outlier_channel=outlier_ch,
            outlier_offset=3.0,
            rng=rng,
        )
        _, labels = _make_labels(exp1_dir)
        _, annotations = _make_annotations(exp1_dir, outlier_channel=outlier_ch)

        # Run analysis using the product_embeddings dict directly
        inconsistent_results = analyze_inconsistent_products(
            product_embeddings, labels, annotations
        )
        consistent_results = analyze_consistent_products(product_embeddings, labels)
        results = build_results(inconsistent_results, consistent_results)

        # With strong outlier offset, should pass
        assert results.verdict == "PASS"
        assert results.n_correct >= PASS_THRESHOLD
        assert results.accuracy >= 0.6

    def test_full_pipeline_with_reconstruction(self, tmp_path: Path) -> None:
        """Pipeline works when reconstructing from .npz file."""
        rng = np.random.default_rng(42)
        outlier_ch = "social"

        exp1_dir = tmp_path / "exp1"
        exp1_dir.mkdir(parents=True)

        _make_verdict(exp1_dir)
        _, _ = _make_embeddings(
            exp1_dir,
            n_consistent=10,
            n_inconsistent=10,
            outlier_channel=outlier_ch,
            outlier_offset=3.0,
            rng=rng,
        )
        _, labels = _make_labels(exp1_dir)
        _, annotations = _make_annotations(exp1_dir, outlier_channel=outlier_ch)

        # Reconstruct from .npz
        npz = np.load(exp1_dir / "embeddings.npz")
        product_embeddings = reconstruct_embeddings(npz)

        # Run
        inconsistent_results = analyze_inconsistent_products(
            product_embeddings, labels, annotations
        )
        consistent_results = analyze_consistent_products(product_embeddings, labels)
        results = build_results(inconsistent_results, consistent_results)

        assert results.verdict == "PASS"

    def test_random_embeddings_low_accuracy(self, tmp_path: Path) -> None:
        """With purely random embeddings, outlier ID should be unreliable."""
        rng = np.random.default_rng(42)
        dim = D

        # Build product embeddings where ALL channels are random (no structure)
        product_embeddings: dict[str, dict[str, np.ndarray]] = {}
        labels: dict[str, bool] = {}
        for i in range(10):
            pid = f"inconsistent_{i:03d}"
            labels[pid] = False
            product_embeddings[pid] = {}
            for ch in CHANNELS:
                vec = rng.standard_normal(dim)
                vec = vec / np.linalg.norm(vec)
                product_embeddings[pid][ch] = vec

        # Annotations always say "social" but embeddings are random
        annotations = {f"inconsistent_{i:03d}": "social" for i in range(10)}

        inconsistent_results = analyze_inconsistent_products(
            product_embeddings, labels, annotations
        )
        results = build_results(inconsistent_results, [])

        # With random embeddings, chance of guessing "social" correctly is ~1/4
        # So accuracy should be well below 60% (pass threshold)
        assert results.accuracy < 0.7
