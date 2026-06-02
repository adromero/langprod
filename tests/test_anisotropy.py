"""Tests for anisotropy correction (analysis.py — correct_anisotropy)."""

import numpy as np
import pytest

from analysis import correct_anisotropy

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# "none" passthrough
# ---------------------------------------------------------------------------


def test_none_passthrough():
    """method='none' returns a copy of the input, unchanged."""
    X = RNG.standard_normal((20, 64))
    result = correct_anisotropy(X, method="none")
    np.testing.assert_array_equal(result, X)
    # Must be a copy, not the same object
    assert result is not X


# ---------------------------------------------------------------------------
# Mean centering
# ---------------------------------------------------------------------------


def test_mean_centering_zero_mean():
    """After mean centering, the column-wise mean is approximately zero."""
    X = RNG.standard_normal((30, 64)) + 5.0  # offset so mean is non-zero
    result = correct_anisotropy(X, method="mean_centering")
    col_means = result.mean(axis=0)
    np.testing.assert_allclose(col_means, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------


def test_whitening_unit_variance():
    """After whitening, each component has approximately unit variance."""
    N = 50
    D = 20
    X = RNG.standard_normal((N, D)) * np.arange(1, D + 1)  # varying scale
    result = correct_anisotropy(X, method="whitening")
    variances = np.var(result, axis=0)
    np.testing.assert_allclose(variances, 1.0, atol=0.15)


def test_whitening_uncorrelated():
    """After whitening, components are approximately uncorrelated."""
    N = 50
    D = 10
    # Create correlated data
    rng = np.random.default_rng(42)
    cov = rng.standard_normal((D, D))
    cov = cov @ cov.T  # positive semi-definite
    L = np.linalg.cholesky(cov + 0.1 * np.eye(D))
    X = rng.standard_normal((N, D)) @ L.T

    result = correct_anisotropy(X, method="whitening")
    # Compute correlation matrix of whitened data
    corr = np.corrcoef(result.T)
    # Off-diagonal elements should be near zero
    off_diag = corr[np.triu_indices(corr.shape[0], k=1)]
    np.testing.assert_allclose(off_diag, 0.0, atol=0.15)


def test_whitening_no_nan_rank_deficient():
    """Whitening handles N < D (rank-deficient) without producing NaN or Inf."""
    N = 5  # fewer samples than dimensions
    D = 50
    X = RNG.standard_normal((N, D))
    result = correct_anisotropy(X, method="whitening")
    assert not np.any(np.isnan(result)), "NaN found in whitened output"
    assert not np.any(np.isinf(result)), "Inf found in whitened output"
    # Output should have N rows (same as input)
    assert result.shape[0] == N
    # n_components defaults to min(N-1, D) = 4, so output columns = 4
    assert result.shape[1] == min(N - 1, D)
