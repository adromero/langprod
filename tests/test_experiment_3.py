"""Tests for coherence.experiment_3 -- uses synthetic embeddings, no GPU required."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from coherence.experiment_3 import (
    ACCURACY_THRESHOLD,
    MIN_PRODUCTS_PASSING,
    PROBE_LEVEL_DELTA_WARN,
    SIMILARITY_CUTOFF,
    _default_attribute_annotations,
    build_verdict,
    check_gates,
    evaluate_accuracy,
    load_exp1_embeddings,
    run_experiment,
)
from coherence.metrics import AttributeCoherenceResult, compute_attribute_coherence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D = 64  # reduced hidden dim for tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verdict_file(tmp_dir: Path, overall: str = "PASS") -> Path:
    """Create a mock verdict.json with the given overall status."""
    verdict_data = {
        "verdict": {
            "overall": overall,
            "criteria": {},
            "notes": [],
        }
    }
    verdict_path = tmp_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict_data, f)
    return verdict_path


def _make_embeddings_file(
    tmp_dir: Path,
    product_ids: List[str],
    channels: List[str],
    dim: int = D,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create a mock embeddings.npz and return the product embeddings dict."""
    if rng is None:
        rng = np.random.default_rng(42)

    flat: Dict[str, np.ndarray] = {}
    product_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    for pid in product_ids:
        product_embeddings[pid] = {}
        for ch in channels:
            vec = rng.standard_normal(dim).astype(np.float64)
            vec = vec / (np.linalg.norm(vec) + 1e-12)  # normalize
            flat[f"{pid}/{ch}"] = vec
            product_embeddings[pid][ch] = vec

    np.savez_compressed(tmp_dir / "embeddings.npz", **flat)
    return product_embeddings


def _make_synthetic_embed_fn(
    product_embeddings: Dict[str, Dict[str, np.ndarray]],
    annotations: List[Dict[str, Any]],
    dim: int = D,
    noise: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """Create a mock embedding function that produces probe embeddings.

    The mock makes probes similar to channels where ground truth is True
    and dissimilar where ground truth is False, so the metric can be tested
    under idealized conditions.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # Pre-compute what each probe "should" look like based on ground truth.
    # For each attribute, the probe embedding is a weighted average of the
    # channel embeddings where it's present, plus noise.
    def embed_fn(
        texts: List[str],
        global_mean: Optional[np.ndarray],
        metric_selection: Optional[dict],
    ) -> np.ndarray:
        """Synthetic embedding that aligns probes with present-channel embeddings."""
        # We need to figure out which texts correspond to which attributes.
        # The caller sends: sentence_probes + paragraph_probes + synonym_probes
        # We can just generate embeddings that work for the test.
        result = []
        for text in texts:
            # Find which annotation/attribute this text belongs to
            matched = False
            for ann in annotations:
                product_id = ann["product_id"]
                if product_id not in product_embeddings:
                    continue
                ch_embs = product_embeddings[product_id]
                for attr in ann["attributes"]:
                    if (
                        text == attr["sentence_probe"]
                        or text == attr["paragraph_probe"]
                        or text in attr.get("synonyms", [])
                    ):
                        # Build embedding as average of "present" channels + noise
                        gt = attr["ground_truth"]
                        present_channels = [
                            ch for ch, is_present in gt.items()
                            if is_present and ch in ch_embs
                        ]
                        absent_channels = [
                            ch for ch, is_present in gt.items()
                            if not is_present and ch in ch_embs
                        ]

                        if present_channels:
                            base = np.mean(
                                [ch_embs[ch] for ch in present_channels], axis=0
                            )
                        else:
                            # No present channels: use random direction
                            base = rng.standard_normal(dim)

                        # Add slight repulsion from absent channels
                        if absent_channels:
                            absent_mean = np.mean(
                                [ch_embs[ch] for ch in absent_channels], axis=0
                            )
                            base = base - 0.3 * absent_mean

                        # Add noise
                        base = base + noise * rng.standard_normal(dim)
                        base = base / (np.linalg.norm(base) + 1e-12)
                        result.append(base)
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                # Fallback: random embedding
                vec = rng.standard_normal(dim)
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                result.append(vec)

        return np.vstack(result)

    return embed_fn


# ---------------------------------------------------------------------------
# Tests: Gate checks
# ---------------------------------------------------------------------------


class TestCheckGates:
    """Gate checks for Experiment 1 dependencies."""

    def test_pass_verdict(self, tmp_path: Path) -> None:
        verdict_path = _make_verdict_file(tmp_path, "PASS")
        with patch("coherence.experiment_3.VERDICT_PATH", verdict_path):
            result = check_gates()
        assert result["verdict"]["overall"] == "PASS"

    def test_pass_no_value_added_verdict(self, tmp_path: Path) -> None:
        verdict_path = _make_verdict_file(tmp_path, "PASS_NO_VALUE_ADDED")
        with patch("coherence.experiment_3.VERDICT_PATH", verdict_path):
            result = check_gates()
        assert result["verdict"]["overall"] == "PASS_NO_VALUE_ADDED"

    def test_fail_verdict(self, tmp_path: Path) -> None:
        verdict_path = _make_verdict_file(tmp_path, "FAIL")
        with patch("coherence.experiment_3.VERDICT_PATH", verdict_path):
            with pytest.raises(SystemExit):
                check_gates()

    def test_missing_verdict(self, tmp_path: Path) -> None:
        verdict_path = tmp_path / "nonexistent" / "verdict.json"
        with patch("coherence.experiment_3.VERDICT_PATH", verdict_path):
            with pytest.raises(SystemExit):
                check_gates()


# ---------------------------------------------------------------------------
# Tests: Attribute annotations schema
# ---------------------------------------------------------------------------


class TestAnnotationSchema:
    """Validate the attribute annotation schema."""

    def test_three_products(self) -> None:
        annotations = _default_attribute_annotations()
        assert len(annotations) == 3

    def test_each_product_has_required_fields(self) -> None:
        annotations = _default_attribute_annotations()
        for ann in annotations:
            assert "product_id" in ann
            assert "category" in ann
            assert "wrong_product_id" in ann
            assert "attributes" in ann
            assert isinstance(ann["attributes"], list)
            assert len(ann["attributes"]) >= 3

    def test_each_attribute_has_required_fields(self) -> None:
        annotations = _default_attribute_annotations()
        for ann in annotations:
            for attr in ann["attributes"]:
                assert "name" in attr
                assert "sentence_probe" in attr
                assert "paragraph_probe" in attr
                assert "synonyms" in attr
                assert "ground_truth" in attr
                assert isinstance(attr["ground_truth"], dict)
                # Ground truth must have at least 2 channels
                assert len(attr["ground_truth"]) >= 2

    def test_wrong_product_is_different(self) -> None:
        annotations = _default_attribute_annotations()
        for ann in annotations:
            assert ann["product_id"] != ann["wrong_product_id"]

    def test_each_product_has_3_to_5_attributes(self) -> None:
        annotations = _default_attribute_annotations()
        for ann in annotations:
            n = len(ann["attributes"])
            assert 3 <= n <= 5, f"{ann['product_id']} has {n} attributes"

    def test_synonyms_are_nonempty(self) -> None:
        annotations = _default_attribute_annotations()
        for ann in annotations:
            for attr in ann["attributes"]:
                assert len(attr["synonyms"]) >= 1, (
                    f"{ann['product_id']}/{attr['name']} has no synonyms"
                )

    def test_ground_truth_has_mixed_values(self) -> None:
        """Each product should have at least one attribute with a mix of True/False."""
        annotations = _default_attribute_annotations()
        for ann in annotations:
            has_mixed = False
            for attr in ann["attributes"]:
                vals = set(attr["ground_truth"].values())
                if True in vals and False in vals:
                    has_mixed = True
                    break
            assert has_mixed, f"{ann['product_id']} has no mixed ground truth"


# ---------------------------------------------------------------------------
# Tests: Evaluate accuracy
# ---------------------------------------------------------------------------


class TestEvaluateAccuracy:
    """Test the accuracy evaluation against ground truth."""

    def test_perfect_accuracy(self) -> None:
        """When predictions match ground truth exactly, accuracy should be 1.0."""
        channel_names = ["marketing", "regulatory", "retail"]
        attributes = [
            {
                "name": "attr1",
                "ground_truth": {"marketing": True, "regulatory": False, "retail": True},
                "sentence_probe": "",
                "paragraph_probe": "",
                "synonyms": [],
            }
        ]
        # Matrix where marketing=0.8, regulatory=0.2, retail=0.7 (cutoff=0.5)
        matrix = np.array([[0.8, 0.2, 0.7]])
        result = AttributeCoherenceResult(matrix=matrix, channel_names=channel_names)
        acc, correct, total = evaluate_accuracy(result, attributes)
        assert acc == 1.0
        assert correct == 3
        assert total == 3

    def test_zero_accuracy(self) -> None:
        """When predictions are all wrong, accuracy should be 0.0."""
        channel_names = ["marketing", "regulatory"]
        attributes = [
            {
                "name": "attr1",
                "ground_truth": {"marketing": False, "regulatory": True},
                "sentence_probe": "",
                "paragraph_probe": "",
                "synonyms": [],
            }
        ]
        # marketing=0.9 (predicted True, actual False), regulatory=0.1 (predicted False, actual True)
        matrix = np.array([[0.9, 0.1]])
        result = AttributeCoherenceResult(matrix=matrix, channel_names=channel_names)
        acc, correct, total = evaluate_accuracy(result, attributes)
        assert acc == 0.0
        assert correct == 0
        assert total == 2

    def test_partial_accuracy(self) -> None:
        """50% accuracy case."""
        channel_names = ["marketing", "regulatory", "retail", "social"]
        attributes = [
            {
                "name": "attr1",
                "ground_truth": {
                    "marketing": True,
                    "regulatory": True,
                    "retail": False,
                    "social": False,
                },
                "sentence_probe": "",
                "paragraph_probe": "",
                "synonyms": [],
            }
        ]
        # marketing=0.8 (correct), regulatory=0.3 (wrong), retail=0.7 (wrong), social=0.2 (correct)
        matrix = np.array([[0.8, 0.3, 0.7, 0.2]])
        result = AttributeCoherenceResult(matrix=matrix, channel_names=channel_names)
        acc, correct, total = evaluate_accuracy(result, attributes)
        assert acc == pytest.approx(0.5)
        assert correct == 2
        assert total == 4

    def test_custom_cutoff(self) -> None:
        """Custom cutoff should change predictions."""
        channel_names = ["marketing", "regulatory"]
        attributes = [
            {
                "name": "attr1",
                "ground_truth": {"marketing": True, "regulatory": False},
                "sentence_probe": "",
                "paragraph_probe": "",
                "synonyms": [],
            }
        ]
        matrix = np.array([[0.3, 0.1]])
        result = AttributeCoherenceResult(matrix=matrix, channel_names=channel_names)

        # With default cutoff=0.5: marketing=0.3 -> False (wrong), regulatory=0.1 -> False (correct)
        acc1, _, _ = evaluate_accuracy(result, attributes, cutoff=0.5)
        assert acc1 == 0.5

        # With cutoff=0.2: marketing=0.3 -> True (correct), regulatory=0.1 -> False (correct)
        acc2, _, _ = evaluate_accuracy(result, attributes, cutoff=0.2)
        assert acc2 == 1.0

    def test_channels_not_in_ground_truth_ignored(self) -> None:
        """Channels not in ground truth should be skipped."""
        channel_names = ["marketing", "regulatory", "unknown"]
        attributes = [
            {
                "name": "attr1",
                "ground_truth": {"marketing": True, "regulatory": False},
                "sentence_probe": "",
                "paragraph_probe": "",
                "synonyms": [],
            }
        ]
        matrix = np.array([[0.8, 0.2, 0.9]])
        result = AttributeCoherenceResult(matrix=matrix, channel_names=channel_names)
        acc, correct, total = evaluate_accuracy(result, attributes)
        assert total == 2  # only marketing and regulatory counted
        assert acc == 1.0


# ---------------------------------------------------------------------------
# Tests: Build verdict
# ---------------------------------------------------------------------------


class TestBuildVerdict:
    """Test verdict construction logic."""

    def _make_product_result(
        self,
        product_id: str,
        acc_sentence: float,
        acc_paragraph: float,
        wrong_pass_sentence: bool = True,
        wrong_pass_paragraph: bool = True,
    ) -> Dict[str, Any]:
        """Build a minimal product result dict for verdict testing."""
        delta = abs(acc_sentence - acc_paragraph)
        return {
            "product_id": product_id,
            "category": "test",
            "wrong_product_id": f"wrong_{product_id}",
            "n_attributes": 3,
            "sentence": {
                "accuracy": acc_sentence,
                "n_correct": int(acc_sentence * 12),
                "n_total": 12,
                "matrix": [],
                "channel_names": [],
            },
            "paragraph": {
                "accuracy": acc_paragraph,
                "n_correct": int(acc_paragraph * 12),
                "n_total": 12,
                "matrix": [],
                "channel_names": [],
            },
            "wrong_product_control": {
                "sentence": {
                    "target_mean": 0.7,
                    "wrong_mean": 0.5 if wrong_pass_sentence else 0.8,
                    "pass": wrong_pass_sentence,
                },
                "paragraph": {
                    "target_mean": 0.7,
                    "wrong_mean": 0.5 if wrong_pass_paragraph else 0.8,
                    "pass": wrong_pass_paragraph,
                },
            },
            "synonym_report": None,
            "probe_level_delta": delta,
            "probe_level_warning": delta > PROBE_LEVEL_DELTA_WARN,
        }

    def test_all_pass(self) -> None:
        """All criteria met: PASS."""
        results = [
            self._make_product_result("A", 0.80, 0.85),
            self._make_product_result("B", 0.90, 0.88),
            self._make_product_result("C", 0.75, 0.72),
        ]
        verdict = build_verdict(results)
        assert verdict["overall"] == "PASS"
        assert verdict["criteria"]["accuracy_gate"]["pass"] is True
        assert verdict["criteria"]["wrong_product_control"]["pass"] is True

    def test_accuracy_gate_fails(self) -> None:
        """Only 1/3 products pass accuracy at both levels: FAIL."""
        results = [
            self._make_product_result("A", 0.80, 0.85),     # pass both
            self._make_product_result("B", 0.60, 0.88),     # fail sentence
            self._make_product_result("C", 0.75, 0.50),     # fail paragraph
        ]
        verdict = build_verdict(results)
        assert verdict["overall"] == "FAIL"
        assert verdict["criteria"]["accuracy_gate"]["pass"] is False
        assert verdict["criteria"]["accuracy_gate"]["products_passing"] == 1

    def test_wrong_product_control_fails(self) -> None:
        """Wrong-product control fails for one product: FAIL."""
        results = [
            self._make_product_result("A", 0.80, 0.85, wrong_pass_sentence=False),
            self._make_product_result("B", 0.90, 0.88),
            self._make_product_result("C", 0.75, 0.72),
        ]
        verdict = build_verdict(results)
        assert verdict["overall"] == "FAIL"
        assert verdict["criteria"]["wrong_product_control"]["pass"] is False
        assert len(verdict["criteria"]["wrong_product_control"]["failures"]) >= 1

    def test_probe_level_warning(self) -> None:
        """Large probe-level delta triggers warning but not failure."""
        results = [
            self._make_product_result("A", 0.95, 0.70),  # 25pp delta
            self._make_product_result("B", 0.90, 0.88),
            self._make_product_result("C", 0.75, 0.72),
        ]
        verdict = build_verdict(results)
        # A still passes both levels (both >= 0.70), but delta is > 15pp
        assert verdict["overall"] == "PASS"
        assert len(verdict["criteria"]["probe_level_agreement"]["warnings"]) >= 1

    def test_product_must_pass_both_levels(self) -> None:
        """A product passing sentence but failing paragraph is not counted."""
        results = [
            self._make_product_result("A", 0.80, 0.60),   # pass sentence, fail paragraph
            self._make_product_result("B", 0.60, 0.80),   # fail sentence, pass paragraph
            self._make_product_result("C", 0.75, 0.72),   # pass both
        ]
        verdict = build_verdict(results)
        assert verdict["overall"] == "FAIL"
        assert verdict["criteria"]["accuracy_gate"]["products_passing"] == 1

    def test_exactly_two_passing_is_enough(self) -> None:
        """2/3 products passing is sufficient."""
        results = [
            self._make_product_result("A", 0.80, 0.75),
            self._make_product_result("B", 0.90, 0.88),
            self._make_product_result("C", 0.50, 0.40),  # fail
        ]
        verdict = build_verdict(results)
        assert verdict["overall"] == "PASS"
        assert verdict["criteria"]["accuracy_gate"]["products_passing"] == 2

    def test_wrong_product_paragraph_fail(self) -> None:
        """Wrong-product control at paragraph level only also fails."""
        results = [
            self._make_product_result(
                "A", 0.80, 0.85,
                wrong_pass_sentence=True,
                wrong_pass_paragraph=False,
            ),
            self._make_product_result("B", 0.90, 0.88),
            self._make_product_result("C", 0.75, 0.72),
        ]
        verdict = build_verdict(results)
        assert verdict["overall"] == "FAIL"
        assert "A paragraph-level" in verdict["criteria"]["wrong_product_control"]["failures"]


# ---------------------------------------------------------------------------
# Tests: Load Exp1 embeddings
# ---------------------------------------------------------------------------


class TestLoadExp1Embeddings:
    """Test loading embeddings from Experiment 1 artifacts."""

    def test_load_embeddings(self, tmp_path: Path) -> None:
        products = ["prod_A", "prod_B"]
        channels = ["marketing", "regulatory", "retail"]
        expected = _make_embeddings_file(tmp_path, products, channels)

        with patch("coherence.experiment_3.EMBEDDINGS_PATH", tmp_path / "embeddings.npz"):
            loaded = load_exp1_embeddings()

        assert set(loaded.keys()) == set(products)
        for pid in products:
            assert set(loaded[pid].keys()) == set(channels)
            for ch in channels:
                np.testing.assert_allclose(loaded[pid][ch], expected[pid][ch])

    def test_missing_embeddings(self, tmp_path: Path) -> None:
        with patch(
            "coherence.experiment_3.EMBEDDINGS_PATH",
            tmp_path / "nonexistent" / "embeddings.npz",
        ):
            with pytest.raises(SystemExit):
                load_exp1_embeddings()


# ---------------------------------------------------------------------------
# Tests: Run experiment (integration with synthetic embeddings)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """Integration tests for the full experiment pipeline using synthetic embeddings."""

    def _setup_experiment(
        self,
        rng: Optional[np.random.Generator] = None,
        noise: float = 0.05,
    ) -> tuple[
        List[Dict[str, Any]],
        Dict[str, Dict[str, np.ndarray]],
        Any,
    ]:
        """Set up annotations, product embeddings, and embed function."""
        if rng is None:
            rng = np.random.default_rng(42)

        annotations = _default_attribute_annotations()

        # Build product embeddings for all products mentioned in annotations
        # (both target and wrong products)
        all_product_ids = set()
        for ann in annotations:
            all_product_ids.add(ann["product_id"])
            all_product_ids.add(ann["wrong_product_id"])

        channels = ["marketing", "regulatory", "retail", "social"]
        product_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        for pid in sorted(all_product_ids):
            product_embeddings[pid] = {}
            for ch in channels:
                vec = rng.standard_normal(D).astype(np.float64)
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                product_embeddings[pid][ch] = vec

        embed_fn = _make_synthetic_embed_fn(
            product_embeddings, annotations, dim=D, noise=noise, rng=rng
        )

        return annotations, product_embeddings, embed_fn

    def test_experiment_produces_results(self) -> None:
        """Experiment runs and produces structured results."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        assert results["experiment"] == "experiment_3_attribute_drill_down"
        assert results["n_products"] == 3
        assert len(results["products"]) == 3
        assert "verdict" in results

    def test_each_product_has_two_probe_levels(self) -> None:
        """Each product result should have sentence and paragraph sections."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        for pr in results["products"]:
            assert "sentence" in pr
            assert "paragraph" in pr
            assert "accuracy" in pr["sentence"]
            assert "accuracy" in pr["paragraph"]
            assert "matrix" in pr["sentence"]
            assert "matrix" in pr["paragraph"]

    def test_wrong_product_control_present(self) -> None:
        """Each product should have wrong-product control results."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        for pr in results["products"]:
            wpc = pr["wrong_product_control"]
            assert "sentence" in wpc
            assert "paragraph" in wpc
            assert "target_mean" in wpc["sentence"]
            assert "wrong_mean" in wpc["sentence"]
            assert "pass" in wpc["sentence"]

    def test_synonym_report_present(self) -> None:
        """Each product should have a synonym report."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        for pr in results["products"]:
            sr = pr["synonym_report"]
            assert sr is not None
            assert "mean_abs_diff" in sr
            assert "max_abs_diff" in sr
            assert sr["n_comparisons"] > 0

    def test_probe_level_delta_reported(self) -> None:
        """Each product should report probe-level accuracy delta."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        for pr in results["products"]:
            assert "probe_level_delta" in pr
            assert isinstance(pr["probe_level_delta"], float)
            assert pr["probe_level_delta"] >= 0.0

    def test_missing_product_skipped(self) -> None:
        """If a target product is missing from embeddings, it is skipped."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        # Remove one target product
        first_pid = annotations[0]["product_id"]
        del product_embeddings[first_pid]

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        assert results["n_products"] == 2
        product_ids_in_results = {pr["product_id"] for pr in results["products"]}
        assert first_pid not in product_ids_in_results

    def test_missing_wrong_product_control_fails(self) -> None:
        """If a wrong product is missing, the control reports as not passing."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        # Remove wrong product for first annotation
        wrong_pid = annotations[0]["wrong_product_id"]
        if wrong_pid in product_embeddings:
            del product_embeddings[wrong_pid]

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        first_result = next(
            pr for pr in results["products"]
            if pr["product_id"] == annotations[0]["product_id"]
        )
        assert first_result["wrong_product_control"]["sentence"]["pass"] is False
        assert first_result["wrong_product_control"]["paragraph"]["pass"] is False

    def test_idealized_conditions_high_accuracy(self) -> None:
        """Under idealized synthetic conditions (low noise), accuracy should be high."""
        rng = np.random.default_rng(999)
        annotations, product_embeddings, embed_fn = self._setup_experiment(
            rng=rng, noise=0.01
        )

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        # At least some products should have reasonable accuracy
        # (not guaranteed to be 100% even with low noise due to cosine geometry)
        accuracies = []
        for pr in results["products"]:
            accuracies.append(pr["sentence"]["accuracy"])
            accuracies.append(pr["paragraph"]["accuracy"])

        mean_acc = float(np.mean(accuracies))
        # With well-constructed synthetic embeddings, mean accuracy should be above chance
        assert mean_acc > 0.4, f"Mean accuracy {mean_acc:.3f} too low for idealized conditions"

    def test_compute_attribute_coherence_called_twice_per_product(self) -> None:
        """Verify that compute_attribute_coherence is called exactly 2x per product
        (once for sentence probes, once for paragraph probes), plus wrong-product calls."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        call_count = 0
        original_fn = compute_attribute_coherence

        def counting_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)

        with patch("coherence.experiment_3.compute_attribute_coherence", side_effect=counting_fn):
            run_experiment(
                annotations=annotations,
                product_embeddings=product_embeddings,
                embed_fn=embed_fn,
            )

        # For each of 3 products: 2 (target sentence + paragraph) + 2 (wrong sentence + paragraph) = 4
        # Total = 3 * 4 = 12
        assert call_count == 12

    def test_matrix_shapes(self) -> None:
        """Verify that similarity matrices have correct shape (n_attributes x n_channels)."""
        annotations, product_embeddings, embed_fn = self._setup_experiment()

        results = run_experiment(
            annotations=annotations,
            product_embeddings=product_embeddings,
            embed_fn=embed_fn,
        )

        for i, pr in enumerate(results["products"]):
            ann = annotations[i]
            n_attrs = len(ann["attributes"])
            n_channels = len(product_embeddings[ann["product_id"]])

            s_matrix = np.array(pr["sentence"]["matrix"])
            p_matrix = np.array(pr["paragraph"]["matrix"])

            assert s_matrix.shape[0] == n_attrs
            assert s_matrix.shape[1] == n_channels
            assert p_matrix.shape[0] == n_attrs
            assert p_matrix.shape[1] == n_channels
