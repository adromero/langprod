"""Tests for coherence.experiment_0 — uses mock HDF5 data, no GPU required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from coherence.experiment_0 import (
    BONFERRONI_P_THRESHOLD,
    CANDIDATE_HDF5_INDICES,
    EXPECTED_HIDDEN_DIM,
    EXPECTED_N_LAYERS_PLUS_ONE,
    EXPLORATION_REGISTERS,
    HELDOUT_REGISTERS,
    build_stimulus_index,
    check_vocab_narrowness,
    compute_cohens_d,
    compute_ttr_per_product,
    generate_pseudo_products,
    load_layer_vectors,
    permutation_test,
    run_part_a_formal,
    run_part_a_screen,
    run_part_c,
    split_stimuli,
    validate_hdf5,
    variant_average,
)
from coherence.metrics import AGGREGATION_METHODS


# ---------------------------------------------------------------------------
# Test data generation helpers
# ---------------------------------------------------------------------------

N_PRODUCTS = 10
N_REGISTERS = 5
N_VARIANTS = 2
N_STIMULI = N_PRODUCTS * N_REGISTERS * N_VARIANTS  # 100
D = 64  # small hidden dim for speed

REGISTERS = ["marketing", "regulatory", "casual_social", "patent", "journalistic"]
CATEGORIES = ["cat_a", "cat_b"]


def _make_stimuli(n_products: int = N_PRODUCTS) -> list[dict]:
    """Generate mock stimuli matching the real data structure."""
    stimuli = []
    for p in range(n_products):
        product_id = f"product_{p:03d}"
        category = CATEGORIES[p % len(CATEGORIES)]
        for reg in REGISTERS:
            for v in range(N_VARIANTS):
                stimuli.append({
                    "stimulus_id": f"{product_id}_{reg}_v{v}",
                    "product_id": product_id,
                    "category": category,
                    "register": reg,
                    "variant": v,
                    "is_fictional": False,
                    "text": f"This is a {reg} text for {product_id} variant {v} "
                            f"with some words to compute TTR uniquely for product {p}",
                    "token_count": 50 + p + v,
                    "generator": "test",
                    "core_attributes": {"attr_a": f"value_{p}", "attr_b": "42"},
                    "generated_at": "2025-01-01T00:00:00Z",
                })
    return stimuli


def _make_hdf5(path: Path, n_stimuli: int, stimuli: list[dict], rng: np.random.Generator) -> None:
    """Create a mock HDF5 file with realistic structure.

    Products sharing the same category/product have more similar vectors
    to ensure coherence effects are detectable.
    """
    n_layers_plus_one = EXPECTED_N_LAYERS_PLUS_ONE
    hidden_dim = D

    # Build per-product base vectors so same-product stimuli are similar
    product_ids = sorted(set(s["product_id"] for s in stimuli))
    product_bases = {}
    for pid in product_ids:
        product_bases[pid] = rng.standard_normal(hidden_dim)
        product_bases[pid] /= np.linalg.norm(product_bases[pid])

    data = np.zeros((n_stimuli, n_layers_plus_one, hidden_dim), dtype=np.float32)
    for i, s in enumerate(stimuli):
        base = product_bases[s["product_id"]]
        for layer in range(n_layers_plus_one):
            # Add layer-dependent noise — less noise in middle layers for realism
            noise_scale = 0.3 if 20 <= layer <= 45 else 0.8
            noise = rng.standard_normal(hidden_dim) * noise_scale
            vec = base + noise
            data[i, layer, :] = vec.astype(np.float32)

    stimulus_ids = [s["stimulus_id"] for s in stimuli]

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "hidden_states_mean_no_special",
            data=data,
            dtype="float32",
        )
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("stimulus_ids", data=stimulus_ids, dtype=dt)


@pytest.fixture()
def mock_data(tmp_path: Path):
    """Create mock stimuli and HDF5 data, return (stimuli, h5_path)."""
    rng = np.random.default_rng(42)
    stimuli = _make_stimuli()
    h5_path = tmp_path / "mock_hidden_states.h5"
    _make_hdf5(h5_path, len(stimuli), stimuli, rng)
    return stimuli, h5_path


# ---------------------------------------------------------------------------
# Tests: HDF5 validation
# ---------------------------------------------------------------------------


class TestValidateHDF5:
    """Validate HDF5 shape checking logic."""

    def test_valid(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        # Patch EXPECTED_HIDDEN_DIM since mock uses D=64
        with patch("coherence.experiment_0.EXPECTED_HIDDEN_DIM", D):
            assert validate_hdf5(h5_path, len(stimuli)) is True

    def test_missing_file(self, tmp_path: Path) -> None:
        assert validate_hdf5(tmp_path / "nonexistent.h5", 100) is False

    def test_wrong_stimuli_count(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        with patch("coherence.experiment_0.EXPECTED_HIDDEN_DIM", D):
            assert validate_hdf5(h5_path, 999) is False

    def test_wrong_hidden_dim(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        # Don't patch — the default expects 5120 but we have D=64
        assert validate_hdf5(h5_path, len(stimuli)) is False


# ---------------------------------------------------------------------------
# Tests: data splitting
# ---------------------------------------------------------------------------


class TestSplitStimuli:
    """Stimuli are correctly split by register."""

    def test_split_counts(self) -> None:
        stimuli = _make_stimuli()
        explore, heldout = split_stimuli(stimuli)

        explore_regs = set(s["register"] for s in explore)
        heldout_regs = set(s["register"] for s in heldout)

        assert explore_regs == EXPLORATION_REGISTERS
        assert heldout_regs == HELDOUT_REGISTERS
        assert len(explore) + len(heldout) == len(stimuli)

    def test_no_overlap(self) -> None:
        stimuli = _make_stimuli()
        explore, heldout = split_stimuli(stimuli)
        explore_ids = set(s["stimulus_id"] for s in explore)
        heldout_ids = set(s["stimulus_id"] for s in heldout)
        assert len(explore_ids & heldout_ids) == 0


# ---------------------------------------------------------------------------
# Tests: variant averaging
# ---------------------------------------------------------------------------


class TestVariantAverage:
    """Variant averaging produces correct structure."""

    def test_shape_and_keys(self) -> None:
        rng = np.random.default_rng(10)
        stimuli = _make_stimuli(n_products=3)
        # Filter to exploration registers only
        explore = [s for s in stimuli if s["register"] in EXPLORATION_REGISTERS]
        vectors = rng.standard_normal((len(explore), D))

        result = variant_average(explore, vectors)

        # Should have 3 products
        assert len(result) == 3
        for pid, reg_dict in result.items():
            # Each product should have the 3 exploration registers
            assert set(reg_dict.keys()) == EXPLORATION_REGISTERS
            for vec in reg_dict.values():
                assert vec.shape == (D,)

    def test_averaging_is_correct(self) -> None:
        """With known vectors, variant average should be element-wise mean."""
        stimuli = [
            {"stimulus_id": "p_reg_v0", "product_id": "p", "register": "reg", "variant": 0},
            {"stimulus_id": "p_reg_v1", "product_id": "p", "register": "reg", "variant": 1},
        ]
        v0 = np.array([1.0, 2.0, 3.0])
        v1 = np.array([3.0, 4.0, 5.0])
        vectors = np.stack([v0, v1])

        result = variant_average(stimuli, vectors)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result["p"]["reg"], expected)


# ---------------------------------------------------------------------------
# Tests: Cohen's d and permutation test
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
        # With sd=1 for both, pooled_sd=1, d = 10/1 = 10
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


class TestPermutationTest:
    """Permutation test for significance."""

    def test_significant_difference(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(10, 1, size=50)
        b = rng.normal(0, 1, size=50)
        d = compute_cohens_d(a, b)
        p = permutation_test(a, b, d, 200, rng)
        assert p < 0.05  # should be very significant

    def test_no_difference(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(0, 1, size=50)
        d = compute_cohens_d(a, b)
        p = permutation_test(a, b, d, 200, rng)
        assert p > 0.01  # should not be significant


# ---------------------------------------------------------------------------
# Tests: pseudo-product generation
# ---------------------------------------------------------------------------


class TestPseudoProducts:
    """Distribution B pseudo-product generation."""

    def test_count_and_structure(self) -> None:
        rng = np.random.default_rng(42)
        registers = ["marketing", "regulatory", "casual_social"]
        product_vecs = {
            f"p{i}": {reg: rng.standard_normal(D) for reg in registers}
            for i in range(10)
        }
        pseudos = generate_pseudo_products(product_vecs, registers, 20, rng)
        assert len(pseudos) == 20
        for pp in pseudos:
            assert set(pp.keys()) == set(registers)
            for vec in pp.values():
                assert vec.shape == (D,)

    def test_distinct_sources(self) -> None:
        """Each pseudo-product should draw from distinct source products."""
        rng = np.random.default_rng(42)
        registers = ["patent", "journalistic"]
        # Use distinguishable vectors
        product_vecs = {}
        for i in range(10):
            product_vecs[f"p{i}"] = {}
            for reg in registers:
                vec = np.zeros(D)
                vec[i] = 1.0  # unique per product
                product_vecs[f"p{i}"][reg] = vec

        pseudos = generate_pseudo_products(product_vecs, registers, 50, rng)
        for pp in pseudos:
            # The two vectors should come from different products
            v_patent = pp["patent"]
            v_journal = pp["journalistic"]
            # If from same product, they would have the same non-zero index
            patent_idx = np.argmax(np.abs(v_patent))
            journal_idx = np.argmax(np.abs(v_journal))
            assert patent_idx != journal_idx, "Pseudo-product drew from same source"


# ---------------------------------------------------------------------------
# Tests: vocabulary narrowness
# ---------------------------------------------------------------------------


class TestVocabNarrowness:
    """TTR-based vocabulary narrowness check."""

    def test_ttr_computation(self) -> None:
        stimuli = [
            {"product_id": "p1", "register": "marketing", "text": "hello world hello"},
            {"product_id": "p1", "register": "regulatory", "text": "a b c d e"},
        ]
        result = compute_ttr_per_product(stimuli)
        assert "p1" in result
        # "hello world hello" -> 3 words, 2 unique -> TTR = 2/3
        assert result["p1"]["marketing"] == pytest.approx(2.0 / 3, abs=1e-6)
        # "a b c d e" -> 5 words, 5 unique -> TTR = 1.0
        assert result["p1"]["regulatory"] == pytest.approx(1.0, abs=1e-6)

    def test_flag_uncorrelated(self) -> None:
        scores = {f"p{i}": float(i) for i in range(10)}
        # Random TTR values
        rng = np.random.default_rng(42)
        ttr = {f"p{i}": {"marketing": rng.uniform(0.3, 0.9)} for i in range(10)}
        flag, rho = check_vocab_narrowness(scores, ttr)
        # With random TTR, unlikely to be > 0.5
        # This is a probabilistic test but with seed it's deterministic
        assert isinstance(flag, bool)
        assert isinstance(rho, float)


# ---------------------------------------------------------------------------
# Tests: build_stimulus_index
# ---------------------------------------------------------------------------


class TestBuildStimulusIndex:
    """Mapping from stimulus_id to HDF5 row index."""

    def test_correct_mapping(self) -> None:
        all_stimuli = _make_stimuli()
        # Take a subset
        subset = [all_stimuli[i] for i in [0, 5, 10]]
        index = build_stimulus_index(subset, all_stimuli)
        assert index[all_stimuli[0]["stimulus_id"]] == 0
        assert index[all_stimuli[5]["stimulus_id"]] == 5
        assert index[all_stimuli[10]["stimulus_id"]] == 10


# ---------------------------------------------------------------------------
# Tests: load_layer_vectors
# ---------------------------------------------------------------------------


class TestLoadLayerVectors:
    """Reading specific rows and layers from HDF5."""

    def test_correct_shape(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        row_indices = [0, 1, 2]
        vectors = load_layer_vectors(h5_path, 21, row_indices)
        assert vectors.shape == (3, D)
        assert vectors.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests: Part A screening (integration, small scale)
# ---------------------------------------------------------------------------


class TestPartAScreen:
    """Integration test of Part A screening with mock data."""

    def test_returns_27_results(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        with (
            patch("coherence.experiment_0.DATA_DIR", h5_path.parent),
            patch("coherence.experiment_0.EXPECTED_HIDDEN_DIM", D),
            patch("coherence.experiment_0.HDF5_FILENAME", h5_path.name),
        ):
            results = run_part_a_screen(h5_path, stimuli, stimuli)

        assert len(results) == 27  # 9 layers x 3 methods
        for r in results:
            assert "cohens_d" in r
            assert "hdf5_index" in r
            assert "aggregation" in r
            assert r["hdf5_index"] in CANDIDATE_HDF5_INDICES
            assert r["aggregation"] in AGGREGATION_METHODS

    def test_sorted_by_d(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        with (
            patch("coherence.experiment_0.DATA_DIR", h5_path.parent),
            patch("coherence.experiment_0.EXPECTED_HIDDEN_DIM", D),
            patch("coherence.experiment_0.HDF5_FILENAME", h5_path.name),
        ):
            results = run_part_a_screen(h5_path, stimuli, stimuli)

        for i in range(len(results) - 1):
            assert abs(results[i]["cohens_d"]) >= abs(results[i + 1]["cohens_d"])


# ---------------------------------------------------------------------------
# Tests: Part A formal testing (integration, small scale)
# ---------------------------------------------------------------------------


class TestPartAFormal:
    """Integration test of Part A formal testing with mock data."""

    def test_returns_9_results(self, mock_data: tuple) -> None:
        stimuli, h5_path = mock_data
        with (
            patch("coherence.experiment_0.DATA_DIR", h5_path.parent),
            patch("coherence.experiment_0.EXPECTED_HIDDEN_DIM", D),
            patch("coherence.experiment_0.HDF5_FILENAME", h5_path.name),
            patch("coherence.experiment_0.N_PERMUTATIONS", 50),  # speed up
        ):
            screen_results = run_part_a_screen(h5_path, stimuli, stimuli)
            formal_results = run_part_a_formal(h5_path, stimuli, stimuli, screen_results)

        assert len(formal_results) == 9  # 3 layers x 3 methods
        for r in formal_results:
            assert "cohens_d" in r
            assert "p_value" in r
            assert "significant_bonferroni" in r
            assert 0.0 <= r["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: Part C metric lock
# ---------------------------------------------------------------------------


class TestPartC:
    """Metric lock writes correct JSON."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        formal_results = [
            {
                "hdf5_index": 31,
                "transformer_layer": 30,
                "aggregation": "mean_pairwise",
                "cohens_d": 0.75,
                "p_value": 0.001,
                "significant_bonferroni": True,
                "mean_a": 0.85,
                "mean_b": 0.60,
                "std_a": 0.12,
                "std_b": 0.15,
                "vocab_narrowness_flag": False,
                "vocab_narrowness_rho": 0.1,
            },
        ]

        with patch("coherence.experiment_0.OUTPUT_DIR", tmp_path):
            result = run_part_c(formal_results, None)

        output_path = tmp_path / "metric_selection.json"
        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["model"] == "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
        assert loaded["layer_hdf5_index"] == 31
        assert loaded["layer_transformer"] == 30
        assert loaded["correction"] == "mean_centering"
        assert loaded["aggregation"] == "mean_pairwise"
        assert loaded["distance"] == "cosine"
        assert loaded["vocab_narrowness_flag"] is False
        assert loaded["effect_size_d"] == pytest.approx(0.75)
        assert loaded["p_value"] == pytest.approx(0.001)
        assert loaded["status"] == "locked"

    def test_failure_protocol_weak(self, tmp_path: Path) -> None:
        """d < 0.3 triggers contrastive escalation."""
        formal_results = [
            {
                "hdf5_index": 31,
                "transformer_layer": 30,
                "aggregation": "mean_pairwise",
                "cohens_d": 0.2,
                "p_value": 0.1,
                "significant_bonferroni": False,
                "mean_a": 0.50,
                "mean_b": 0.48,
                "std_a": 0.12,
                "std_b": 0.15,
                "vocab_narrowness_flag": False,
                "vocab_narrowness_rho": 0.1,
            },
        ]

        with patch("coherence.experiment_0.OUTPUT_DIR", tmp_path):
            result = run_part_c(formal_results, None)

        assert result["status"] == "failed_needs_contrastive"

    def test_failure_protocol_marginal(self, tmp_path: Path) -> None:
        """0.3 <= d < 0.5 triggers marginal status."""
        formal_results = [
            {
                "hdf5_index": 31,
                "transformer_layer": 30,
                "aggregation": "silhouette",
                "cohens_d": 0.4,
                "p_value": 0.01,
                "significant_bonferroni": False,
                "mean_a": 0.70,
                "mean_b": 0.58,
                "std_a": 0.12,
                "std_b": 0.15,
                "vocab_narrowness_flag": False,
                "vocab_narrowness_rho": 0.1,
            },
        ]

        with patch("coherence.experiment_0.OUTPUT_DIR", tmp_path):
            result = run_part_c(formal_results, None)

        assert result["status"] == "marginal_try_contrastive"

    def test_degradation_issue(self, tmp_path: Path) -> None:
        """d >= 0.5 but Part B not robust."""
        formal_results = [
            {
                "hdf5_index": 31,
                "transformer_layer": 30,
                "aggregation": "mean_pairwise",
                "cohens_d": 0.8,
                "p_value": 0.001,
                "significant_bonferroni": True,
                "mean_a": 0.85,
                "mean_b": 0.55,
                "std_a": 0.12,
                "std_b": 0.15,
                "vocab_narrowness_flag": False,
                "vocab_narrowness_rho": 0.1,
            },
        ]
        part_b = {"all_robust": False, "conditions": {}}

        with patch("coherence.experiment_0.OUTPUT_DIR", tmp_path):
            result = run_part_c(formal_results, part_b)

        assert result["status"] == "strong_but_degradation_issue"
