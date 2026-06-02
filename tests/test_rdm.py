"""Tests for RDM computation and model RDM construction (analysis.py)."""

import numpy as np
import pytest

from analysis import (
    build_product_model_rdm,
    build_within_category_model_rdm,
    compute_rdm,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# compute_rdm
# ---------------------------------------------------------------------------


def test_rdm_symmetric():
    """compute_rdm on random data produces a symmetric matrix."""
    X = RNG.standard_normal((20, 64))
    rdm = compute_rdm(X, metric="cosine")
    np.testing.assert_array_almost_equal(rdm, rdm.T)


def test_rdm_zero_diagonal():
    """compute_rdm produces all-zero diagonal."""
    X = RNG.standard_normal((15, 32))
    rdm = compute_rdm(X, metric="cosine")
    np.testing.assert_array_almost_equal(np.diag(rdm), 0.0)


def test_rdm_shape():
    """N stimuli produce an NxN RDM."""
    N = 25
    X = RNG.standard_normal((N, 48))
    rdm = compute_rdm(X, metric="cosine")
    assert rdm.shape == (N, N)


# ---------------------------------------------------------------------------
# build_product_model_rdm
# ---------------------------------------------------------------------------


def test_product_model_rdm():
    """Product model RDM encodes three-level distance: 0, 0.5, 1.0."""
    stimuli_meta = [
        {"product_id": "A", "category": "cat1"},
        {"product_id": "A", "category": "cat1"},
        {"product_id": "B", "category": "cat1"},
        {"product_id": "C", "category": "cat2"},
    ]
    rdm = build_product_model_rdm(stimuli_meta)

    # Same product (indices 0,1)
    assert rdm[0, 1] == pytest.approx(0.0)
    # Same category, different product (indices 0,2 and 1,2)
    assert rdm[0, 2] == pytest.approx(0.5)
    assert rdm[1, 2] == pytest.approx(0.5)
    # Different category (indices 0,3 and 1,3 and 2,3)
    assert rdm[0, 3] == pytest.approx(1.0)
    assert rdm[1, 3] == pytest.approx(1.0)
    assert rdm[2, 3] == pytest.approx(1.0)
    # Diagonal is zero
    for i in range(4):
        assert rdm[i, i] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_within_category_model_rdm
# ---------------------------------------------------------------------------


def test_within_category_model_rdm():
    """Within-category model RDM marks cross-category pairs as NaN and within-category correctly."""
    stimuli_meta = [
        {"product_id": "A", "category": "cat1"},
        {"product_id": "A", "category": "cat1"},
        {"product_id": "B", "category": "cat1"},
        {"product_id": "C", "category": "cat2"},
        {"product_id": "D", "category": "cat2"},
    ]
    rdm = build_within_category_model_rdm(stimuli_meta)

    # Same product, same category → 0
    assert rdm[0, 1] == pytest.approx(0.0)
    # Different product, same category → 1
    assert rdm[0, 2] == pytest.approx(1.0)
    assert rdm[1, 2] == pytest.approx(1.0)
    # Cross-category → NaN
    assert np.isnan(rdm[0, 3])
    assert np.isnan(rdm[0, 4])
    assert np.isnan(rdm[2, 3])
    # Within cat2, different products → 1
    assert rdm[3, 4] == pytest.approx(1.0)
    # Diagonal → 0
    for i in range(5):
        assert rdm[i, i] == pytest.approx(0.0)
