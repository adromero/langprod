"""Tests for coherence.baselines module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coherence.baselines import (
    compare_methods,
    compute_sbert_coherence,
    compute_tfidf_coherence,
)
from coherence.metrics import CoherenceResult


# ---------------------------------------------------------------------------
# Fixtures: channel text sets
# ---------------------------------------------------------------------------


@pytest.fixture()
def identical_texts() -> dict[str, str]:
    """All channels have the same text."""
    text = "This product provides excellent results for skin hydration and moisture."
    return {
        "regulatory": text,
        "marketing": text,
        "retail": text,
        "social": text,
    }


@pytest.fixture()
def unrelated_texts() -> dict[str, str]:
    """Channels with completely unrelated vocabulary."""
    return {
        "regulatory": (
            "Active ingredient benzoyl peroxide topical application dermatological "
            "treatment acne vulgaris clinical pharmacology mechanism action bactericidal "
            "antimicrobial keratolytic sebaceous follicle"
        ),
        "marketing": (
            "Discover amazing tropical paradise vacation resort luxury beachfront "
            "infinity pool sunset cruise snorkeling coral reef island getaway "
            "all-inclusive spa relaxation"
        ),
        "retail": (
            "Industrial welding equipment argon gas tungsten electrode amperage "
            "voltage shielding fabrication metallurgy thermal conductivity flux "
            "slag bead penetration filler rod"
        ),
        "social": (
            "Cryptocurrency blockchain decentralized finance yield farming liquidity "
            "pool governance token staking validator node consensus proof stake "
            "smart contract ethereum solana"
        ),
    }


@pytest.fixture()
def mixed_texts() -> dict[str, str]:
    """Some channels similar, some different — for two-tier testing."""
    return {
        "regulatory": (
            "This dietary supplement contains vitamin C ascorbic acid and zinc "
            "for immune system support daily nutritional supplementation"
        ),
        "marketing": (
            "Boost your immune system with our powerful vitamin C and zinc formula "
            "for daily health and wellness nutritional support"
        ),
        "retail": (
            "Premium vitamin C supplement with zinc immune support 60 capsules "
            "dietary supplement daily nutritional health formula"
        ),
        "social": (
            "Quantum computing qubits entanglement superposition decoherence "
            "topology cryogenic dilution refrigerator superconducting transmon"
        ),
        "consumer_review": (
            "Deep space astrophysics nebula pulsar magnetar quasar redshift "
            "gravitational lensing dark matter cosmic microwave background"
        ),
    }


# ---------------------------------------------------------------------------
# Mock helper for SBERT
# ---------------------------------------------------------------------------


def _make_sbert_mock(texts_to_embeddings: dict[str, np.ndarray] | None = None):
    """Create a mock SentenceTransformer that returns deterministic embeddings.

    If texts_to_embeddings is None, uses a simple hash-based approach to
    produce deterministic vectors: identical texts get identical vectors,
    different texts get different vectors.
    """
    mock_model = MagicMock()

    def encode_side_effect(texts, convert_to_numpy=True):
        dim = 64
        result = []
        for text in texts:
            # Use hash of text to seed a deterministic RNG
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim)
            vec = vec / np.linalg.norm(vec)
            result.append(vec)
        return np.array(result)

    mock_model.encode = MagicMock(side_effect=encode_side_effect)
    return mock_model


# ---------------------------------------------------------------------------
# Tests: TF-IDF coherence (real sklearn, no mocking needed)
# ---------------------------------------------------------------------------


class TestTfidfCoherence:
    """Test compute_tfidf_coherence with real TF-IDF."""

    def test_returns_coherence_result(self, identical_texts: dict[str, str]) -> None:
        result = compute_tfidf_coherence(identical_texts)
        assert isinstance(result, CoherenceResult)
        assert result.method == "tfidf"
        assert result.vocab_narrowness_flag is None
        assert result.ttr_values is None

    def test_identical_texts_perfect_coherence(self, identical_texts: dict[str, str]) -> None:
        result = compute_tfidf_coherence(identical_texts)
        assert result.market_coherence == pytest.approx(1.0, abs=1e-6)
        assert result.brand_coherence is not None
        assert result.brand_coherence == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_texts_low_coherence(self, unrelated_texts: dict[str, str]) -> None:
        result = compute_tfidf_coherence(unrelated_texts)
        assert result.market_coherence < 0.5
        assert result.brand_coherence is not None
        assert result.brand_coherence < 0.5

    def test_brand_vs_market_with_divergent_channel(self, mixed_texts: dict[str, str]) -> None:
        result = compute_tfidf_coherence(mixed_texts)
        assert result.brand_coherence is not None
        # Brand channels (regulatory, marketing, retail) share vocab;
        # social and consumer_review diverge — brand should be >= market
        assert result.brand_coherence >= result.market_coherence

    def test_fewer_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            compute_tfidf_coherence({"only_one": "some text here"})

    def test_brand_coherence_none_when_few_controlled(self) -> None:
        """Only 1 controlled channel means brand_coherence is None."""
        texts = {
            "regulatory": "some text about regulations and compliance",
            "consumer_review": "different text about product reviews",
            "analyst_report": "third text about financial analysis",
        }
        result = compute_tfidf_coherence(texts)
        assert result.brand_coherence is None


# ---------------------------------------------------------------------------
# Tests: SBERT coherence (mocked model)
# ---------------------------------------------------------------------------


class TestSbertCoherence:
    """Test compute_sbert_coherence with a mocked sentence-transformer."""

    @patch("coherence.baselines._get_sbert_model")
    def test_returns_coherence_result(
        self, mock_get_model: MagicMock, identical_texts: dict[str, str]
    ) -> None:
        mock_get_model.return_value = _make_sbert_mock()
        result = compute_sbert_coherence(identical_texts)
        assert isinstance(result, CoherenceResult)
        assert result.method == "sbert"
        assert result.vocab_narrowness_flag is None
        assert result.ttr_values is None

    @patch("coherence.baselines._get_sbert_model")
    def test_identical_texts_perfect_coherence(
        self, mock_get_model: MagicMock, identical_texts: dict[str, str]
    ) -> None:
        mock_get_model.return_value = _make_sbert_mock()
        result = compute_sbert_coherence(identical_texts)
        # Identical texts produce identical embeddings via the mock, so cosine = 1.0
        assert result.market_coherence == pytest.approx(1.0, abs=1e-6)
        assert result.brand_coherence is not None
        assert result.brand_coherence == pytest.approx(1.0, abs=1e-6)

    @patch("coherence.baselines._get_sbert_model")
    def test_unrelated_texts_low_coherence(
        self, mock_get_model: MagicMock, unrelated_texts: dict[str, str]
    ) -> None:
        mock_get_model.return_value = _make_sbert_mock()
        result = compute_sbert_coherence(unrelated_texts)
        # Different texts get different random unit vectors; expected cosine ~ 0
        # With dim=64 random unit vectors, pairwise cosine is typically |~0.1|
        assert result.market_coherence < 0.5

    @patch("coherence.baselines._get_sbert_model")
    def test_fewer_than_2_raises(self, mock_get_model: MagicMock) -> None:
        mock_get_model.return_value = _make_sbert_mock()
        with pytest.raises(ValueError, match="At least 2"):
            compute_sbert_coherence({"only_one": "some text here"})

    @patch("coherence.baselines._get_sbert_model")
    def test_brand_coherence_none_when_few_controlled(
        self, mock_get_model: MagicMock
    ) -> None:
        mock_get_model.return_value = _make_sbert_mock()
        texts = {
            "regulatory": "regulatory text about compliance",
            "consumer_review": "different review text",
            "analyst_report": "another different text",
        }
        result = compute_sbert_coherence(texts)
        assert result.brand_coherence is None


# ---------------------------------------------------------------------------
# Tests: compare_methods
# ---------------------------------------------------------------------------


class TestCompareMethods:
    """Test compare_methods Spearman correlation output."""

    def test_basic_structure(self) -> None:
        """Two methods with 4 products should produce one comparison entry + summary."""
        results_a = {
            f"p{i}": CoherenceResult(
                brand_coherence=None,
                market_coherence=0.5 + i * 0.1,
                method="tfidf",
            )
            for i in range(4)
        }
        results_b = {
            f"p{i}": CoherenceResult(
                brand_coherence=None,
                market_coherence=0.5 + i * 0.1,
                method="sbert",
            )
            for i in range(4)
        }
        output = compare_methods({"tfidf": results_a, "sbert": results_b})
        assert "sbert_vs_tfidf" in output
        assert "summary" in output
        entry = output["sbert_vs_tfidf"]
        assert "spearman_rho" in entry
        assert "p_value" in entry
        assert "n" in entry
        assert entry["n"] == 4

    def test_perfect_correlation(self) -> None:
        """Identical market_coherence rankings should give rho = 1.0."""
        results_a = {
            f"p{i}": CoherenceResult(
                brand_coherence=None,
                market_coherence=float(i),
                method="method_a",
            )
            for i in range(5)
        }
        results_b = {
            f"p{i}": CoherenceResult(
                brand_coherence=None,
                market_coherence=float(i) * 2.0,  # same rank order
                method="method_b",
            )
            for i in range(5)
        }
        output = compare_methods({"method_a": results_a, "method_b": results_b})
        entry = output["method_a_vs_method_b"]
        assert entry["spearman_rho"] == pytest.approx(1.0, abs=1e-6)

    def test_three_methods(self) -> None:
        """Three methods should produce 3 pairwise comparisons."""
        methods = {}
        for m in ["hidden_state", "sbert", "tfidf"]:
            methods[m] = {
                f"p{i}": CoherenceResult(
                    brand_coherence=None,
                    market_coherence=float(i),
                    method=m,
                )
                for i in range(5)
            }
        output = compare_methods(methods)
        # 3 choose 2 = 3 pairs
        comparison_keys = [k for k in output if k != "summary"]
        assert len(comparison_keys) == 3
        assert "hidden_state_vs_sbert" in output
        assert "hidden_state_vs_tfidf" in output
        assert "sbert_vs_tfidf" in output

    def test_insufficient_products_returns_none(self) -> None:
        """Fewer than 3 common products should give spearman_rho = None."""
        results_a = {
            "p0": CoherenceResult(brand_coherence=None, market_coherence=0.5, method="a"),
            "p1": CoherenceResult(brand_coherence=None, market_coherence=0.6, method="a"),
        }
        results_b = {
            "p0": CoherenceResult(brand_coherence=None, market_coherence=0.7, method="b"),
            "p1": CoherenceResult(brand_coherence=None, market_coherence=0.8, method="b"),
        }
        output = compare_methods({"a": results_a, "b": results_b})
        assert output["a_vs_b"]["spearman_rho"] is None
        assert output["a_vs_b"]["n"] == 2

    def test_json_serializable(self) -> None:
        """Output must be JSON-serializable."""
        import json

        methods = {}
        for m in ["sbert", "tfidf"]:
            methods[m] = {
                f"p{i}": CoherenceResult(
                    brand_coherence=None,
                    market_coherence=float(i),
                    method=m,
                )
                for i in range(5)
            }
        output = compare_methods(methods)
        # This should not raise
        serialized = json.dumps(output)
        assert isinstance(serialized, str)

    def test_partial_overlap(self) -> None:
        """Methods with partially overlapping product sets."""
        results_a = {
            f"p{i}": CoherenceResult(
                brand_coherence=None, market_coherence=float(i), method="a"
            )
            for i in range(6)
        }
        results_b = {
            f"p{i}": CoherenceResult(
                brand_coherence=None, market_coherence=float(i) * 1.5, method="b"
            )
            for i in range(3, 8)  # overlap on p3, p4, p5
        }
        output = compare_methods({"a": results_a, "b": results_b})
        assert output["a_vs_b"]["n"] == 3
        assert output["a_vs_b"]["spearman_rho"] is not None
