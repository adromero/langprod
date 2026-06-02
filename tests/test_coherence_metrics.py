"""Tests for coherence.metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from coherence.metrics import (
    AGGREGATION_METHODS,
    CONTROLLED_CHANNELS,
    AttributeCoherenceResult,
    CoherenceResult,
    OutlierResult,
    batch_coherence_scores,
    compute_attribute_coherence,
    compute_coherence_score,
    compute_pairwise_coherence,
    identify_outlier_channel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 64  # embedding dimensionality for most tests


@pytest.fixture()
def identical_embeddings() -> dict[str, np.ndarray]:
    """4 controlled channels with identical embeddings."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(D)
    vec = vec / np.linalg.norm(vec)
    return {
        "regulatory": vec.copy(),
        "marketing": vec.copy(),
        "retail": vec.copy(),
        "social": vec.copy(),
    }


@pytest.fixture()
def orthogonal_embeddings() -> dict[str, np.ndarray]:
    """4 mutually orthogonal unit vectors (needs D >= 4)."""
    embs: dict[str, np.ndarray] = {}
    names = ["regulatory", "marketing", "retail", "social"]
    for i, name in enumerate(names):
        vec = np.zeros(D)
        vec[i] = 1.0
        embs[name] = vec
    return embs


@pytest.fixture()
def two_tier_embeddings() -> dict[str, np.ndarray]:
    """5 channels: 4 controlled tightly clustered, 1 consumer_review divergent."""
    rng = np.random.default_rng(99)
    base = rng.standard_normal(D)
    base = base / np.linalg.norm(base)

    embs: dict[str, np.ndarray] = {}
    for name in ["regulatory", "marketing", "retail", "social"]:
        noise = rng.standard_normal(D) * 0.01
        v = base + noise
        v = v / np.linalg.norm(v)
        embs[name] = v

    # Divergent channel
    divergent = rng.standard_normal(D)
    divergent = divergent / np.linalg.norm(divergent)
    # Make sure it's actually divergent (far from base)
    divergent = divergent - np.dot(divergent, base) * base
    divergent = divergent / np.linalg.norm(divergent)
    embs["consumer_review"] = divergent
    return embs


@pytest.fixture()
def outlier_embeddings() -> dict[str, np.ndarray]:
    """5 channels: 4 clustered, 1 distant outlier."""
    rng = np.random.default_rng(77)
    base = rng.standard_normal(D)
    base = base / np.linalg.norm(base)

    embs: dict[str, np.ndarray] = {}
    for name in ["regulatory", "marketing", "retail", "social"]:
        noise = rng.standard_normal(D) * 0.02
        v = base + noise
        v = v / np.linalg.norm(v)
        embs[name] = v

    # Outlier: orthogonal to base
    outlier = rng.standard_normal(D)
    outlier = outlier - np.dot(outlier, base) * base
    outlier = outlier / np.linalg.norm(outlier)
    embs["consumer_review"] = outlier
    return embs


# ---------------------------------------------------------------------------
# Tests: perfect coherence (identical embeddings)
# ---------------------------------------------------------------------------


class TestPerfectCoherence:
    """Identical embeddings should yield perfect scores."""

    def test_pairwise_all_ones(self, identical_embeddings: dict[str, np.ndarray]) -> None:
        pairs = compute_pairwise_coherence(identical_embeddings)
        for key, sim in pairs.items():
            assert sim == pytest.approx(1.0, abs=1e-6), f"Pair {key} not 1.0"

    def test_mean_pairwise(self, identical_embeddings: dict[str, np.ndarray]) -> None:
        result = compute_coherence_score(identical_embeddings, method="mean_pairwise")
        assert result.brand_coherence == pytest.approx(1.0, abs=1e-6)
        assert result.market_coherence == pytest.approx(1.0, abs=1e-6)

    def test_centroid_distance(self, identical_embeddings: dict[str, np.ndarray]) -> None:
        result = compute_coherence_score(identical_embeddings, method="centroid_distance")
        assert result.brand_coherence == pytest.approx(1.0, abs=1e-6)
        assert result.market_coherence == pytest.approx(1.0, abs=1e-6)

    def test_silhouette(self, identical_embeddings: dict[str, np.ndarray]) -> None:
        result = compute_coherence_score(identical_embeddings, method="silhouette")
        assert result.brand_coherence == pytest.approx(1.0, abs=1e-6)
        assert result.market_coherence == pytest.approx(1.0, abs=1e-6)

    def test_outlier_gap_zero(self, identical_embeddings: dict[str, np.ndarray]) -> None:
        # Need at least 3 channels; identical_embeddings has 4
        result = identify_outlier_channel(identical_embeddings)
        assert result.gap == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: zero/low coherence (orthogonal) — method-specific
# ---------------------------------------------------------------------------


class TestOrthogonalCoherence:
    """Mutually orthogonal unit vectors (zero pairwise similarity)."""

    def test_pairwise_all_zero(self, orthogonal_embeddings: dict[str, np.ndarray]) -> None:
        pairs = compute_pairwise_coherence(orthogonal_embeddings)
        for key, sim in pairs.items():
            assert sim == pytest.approx(0.0, abs=1e-6), f"Pair {key} not 0.0"

    def test_mean_pairwise_zero(self, orthogonal_embeddings: dict[str, np.ndarray]) -> None:
        result = compute_coherence_score(orthogonal_embeddings, method="mean_pairwise")
        assert result.market_coherence == pytest.approx(0.0, abs=1e-6)

    def test_centroid_distance_half(self, orthogonal_embeddings: dict[str, np.ndarray]) -> None:
        """For N=4 orthogonal unit vectors, centroid = (1/4, 1/4, 1/4, 1/4, 0, ...0).
        cos_sim(e_i, centroid) = (1/4) / (1 * 1/sqrt(4)) = (1/4) / (1/2) = 1/2.
        """
        result = compute_coherence_score(orthogonal_embeddings, method="centroid_distance")
        assert result.market_coherence == pytest.approx(0.5, abs=1e-6)

    def test_silhouette_zero(self, orthogonal_embeddings: dict[str, np.ndarray]) -> None:
        """All pairwise distances = 1.0, mean a_i = 1.0, silhouette = 0.0."""
        result = compute_coherence_score(orthogonal_embeddings, method="silhouette")
        assert result.market_coherence == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: two-tier divergence
# ---------------------------------------------------------------------------


class TestTwoTierDivergence:
    """Brand coherence (controlled channels only) > market coherence (all channels)."""

    def test_brand_greater_than_market(
        self, two_tier_embeddings: dict[str, np.ndarray]
    ) -> None:
        for method in AGGREGATION_METHODS:
            result = compute_coherence_score(two_tier_embeddings, method=method)
            assert result.brand_coherence is not None
            assert result.brand_coherence > result.market_coherence, (
                f"method={method}: brand={result.brand_coherence} "
                f"should be > market={result.market_coherence}"
            )


# ---------------------------------------------------------------------------
# Tests: outlier detection
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    """Correctly identify the divergent channel."""

    def test_outlier_identified(self, outlier_embeddings: dict[str, np.ndarray]) -> None:
        result = identify_outlier_channel(outlier_embeddings)
        assert result.outlier_channel == "consumer_review"
        assert result.gap > 0.1


# ---------------------------------------------------------------------------
# Tests: attribute coherence
# ---------------------------------------------------------------------------


class TestAttributeCoherence:
    """Attribute probes × channel embeddings produce correct matrix."""

    def test_shape_and_names(self) -> None:
        rng = np.random.default_rng(11)
        probes = rng.standard_normal((3, D))
        names = ["attr_a", "attr_b", "attr_c"]
        channels = {
            "marketing": rng.standard_normal(D),
            "regulatory": rng.standard_normal(D),
            "retail": rng.standard_normal(D),
            "social": rng.standard_normal(D),
        }
        result = compute_attribute_coherence(probes, names, channels)
        assert isinstance(result, AttributeCoherenceResult)
        assert result.matrix.shape == (3, 4)
        assert result.channel_names == sorted(channels.keys())

    def test_identical_probe_and_channel(self) -> None:
        """If a probe is identical to a channel embedding, similarity = 1.0."""
        vec = np.random.default_rng(22).standard_normal(D)
        vec = vec / np.linalg.norm(vec)
        probes = vec.reshape(1, D)
        channels = {"alpha": vec.copy(), "beta": np.zeros(D)}
        # beta is zero vector — similarity should be 0.0
        channels["beta"][0] = 1.0  # make it non-zero but orthogonal-ish
        # Actually make it truly orthogonal for clean test
        orth = np.zeros(D)
        orth[1] = 1.0
        # Make probe along dim 0 only
        probe_vec = np.zeros(D)
        probe_vec[0] = 1.0
        probes = probe_vec.reshape(1, D)
        channels = {"alpha": probe_vec.copy(), "beta": orth}
        result = compute_attribute_coherence(probes, ["p1"], channels)
        assert result.matrix[0, 0] == pytest.approx(1.0, abs=1e-6)  # alpha
        assert result.matrix[0, 1] == pytest.approx(0.0, abs=1e-6)  # beta


# ---------------------------------------------------------------------------
# Tests: vocab narrowness (batch)
# ---------------------------------------------------------------------------


class TestVocabNarrowness:
    """Batch flag logic based on Spearman correlation."""

    @staticmethod
    def _make_products(
        n: int, rng: np.random.Generator
    ) -> dict[str, dict[str, np.ndarray]]:
        """Helper: n products with 3 channels each."""
        products: dict[str, dict[str, np.ndarray]] = {}
        for i in range(n):
            channels: dict[str, np.ndarray] = {}
            for ch in ["regulatory", "marketing", "retail"]:
                v = rng.standard_normal(D)
                channels[ch] = v / np.linalg.norm(v)
            products[f"product_{i}"] = channels
        return products

    def test_correlated_flag_true(self) -> None:
        """When coherence and mean inverse TTR are strongly correlated, flag = True."""
        rng = np.random.default_rng(300)
        n = 6

        # Build products with varying coherence via controlled similarity
        products: dict[str, dict[str, np.ndarray]] = {}
        ttr_vals: dict[str, dict[str, float]] = {}

        for i in range(n):
            base = rng.standard_normal(D)
            base = base / np.linalg.norm(base)
            # Spread increases with i → coherence decreases with i
            spread = 0.01 + i * 0.15
            channels: dict[str, np.ndarray] = {}
            for ch in ["regulatory", "marketing", "retail"]:
                noise = rng.standard_normal(D) * spread
                v = base + noise
                channels[ch] = v / np.linalg.norm(v)
            products[f"p{i}"] = channels
            # Coherence decreases with i (more spread).
            # For positive rho(coherence, inv_ttr): inv_ttr should also decrease with i.
            # So TTR increases with i → inv_ttr = 1 - TTR decreases with i.
            ttr = 0.4 + i * 0.1
            ttr_vals[f"p{i}"] = {ch: ttr for ch in ["regulatory", "marketing", "retail"]}

        results = batch_coherence_scores(products, ttr_values=ttr_vals)
        for r in results.values():
            assert r.vocab_narrowness_flag is True

    def test_uncorrelated_flag_false(self) -> None:
        """When coherence and mean inverse TTR are uncorrelated, flag = False."""
        rng = np.random.default_rng(400)
        n = 6

        products: dict[str, dict[str, np.ndarray]] = {}
        ttr_vals: dict[str, dict[str, float]] = {}

        for i in range(n):
            base = rng.standard_normal(D)
            base = base / np.linalg.norm(base)
            spread = 0.01 + i * 0.15
            channels: dict[str, np.ndarray] = {}
            for ch in ["regulatory", "marketing", "retail"]:
                noise = rng.standard_normal(D) * spread
                v = base + noise
                channels[ch] = v / np.linalg.norm(v)
            products[f"p{i}"] = channels
            # TTR: increases with i → inv_ttr decreases → negative correlation with
            # coherence (which also decreases with i) → actually positively correlated
            # We need UNCORRELATED: randomize TTR
            ttr_vals[f"p{i}"] = {
                ch: rng.uniform(0.3, 0.9) for ch in ["regulatory", "marketing", "retail"]
            }

        # Make sure TTR and coherence are not correlated by design:
        # Shuffle the products to break any accidental correlation
        # Actually, we just use random TTR values, which should be uncorrelated
        results = batch_coherence_scores(products, ttr_values=ttr_vals)
        # With random TTR values, the correlation should be low (flag=False)
        for r in results.values():
            assert r.vocab_narrowness_flag is False

    def test_no_ttr_flag_none(self) -> None:
        """Without ttr_values, flag stays None."""
        rng = np.random.default_rng(500)
        products = self._make_products(6, rng)
        results = batch_coherence_scores(products, ttr_values=None)
        for r in results.values():
            assert r.vocab_narrowness_flag is None


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_one_controlled_brand_none(self) -> None:
        """With only 1 controlled channel, brand_coherence is None."""
        rng = np.random.default_rng(600)
        embs = {
            "regulatory": rng.standard_normal(D),
            "consumer_review": rng.standard_normal(D),
            "press_release": rng.standard_normal(D),
        }
        result = compute_coherence_score(embs)
        assert result.brand_coherence is None

    def test_fewer_than_2_total_raises(self) -> None:
        """< 2 total channels raises ValueError."""
        with pytest.raises(ValueError, match="At least 2"):
            compute_coherence_score({"only_one": np.zeros(D)})

    def test_fewer_than_3_for_outlier_raises(self) -> None:
        """< 3 channels for outlier detection raises ValueError."""
        embs = {
            "a": np.ones(D),
            "b": np.ones(D),
        }
        with pytest.raises(ValueError, match="At least 3"):
            identify_outlier_channel(embs)

    def test_no_ttr_single_product_flag_none(self) -> None:
        """Single product call without ttr_values: flag is None."""
        rng = np.random.default_rng(700)
        embs = {
            "regulatory": rng.standard_normal(D),
            "marketing": rng.standard_normal(D),
        }
        result = compute_coherence_score(embs, ttr_values=None)
        assert result.vocab_narrowness_flag is None

    def test_unknown_method_raises(self) -> None:
        """Unknown aggregation method raises ValueError."""
        embs = {
            "regulatory": np.ones(D),
            "marketing": np.ones(D),
        }
        with pytest.raises(ValueError, match="Unknown method"):
            compute_coherence_score(embs, method="bogus")

    def test_constants_defined(self) -> None:
        """Module-level constants are correctly defined."""
        assert CONTROLLED_CHANNELS == {"regulatory", "marketing", "retail", "social"}
        assert AGGREGATION_METHODS == ["mean_pairwise", "centroid_distance", "silhouette"]
