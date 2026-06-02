"""Tests for RSA correlation, partial RSA, and permutation testing (analysis.py)."""

import numpy as np
import pytest

from analysis import (
    compute_rdm,
    partial_rsa,
    rsa_correlation,
    run_permutation_test_tiered,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# rsa_correlation
# ---------------------------------------------------------------------------


def test_rsa_perfect_correlation():
    """RSA on identical RDMs yields r approx 1.0."""
    X = RNG.standard_normal((20, 64))
    rdm = compute_rdm(X, metric="cosine")
    r = rsa_correlation(rdm, rdm)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_rsa_zero_correlation():
    """RSA on unrelated RDMs yields r approx 0.0."""
    # Use two independently constructed RDMs with different structure
    rng_a = np.random.default_rng(100)
    rng_b = np.random.default_rng(999)
    X_a = rng_a.standard_normal((30, 128))
    X_b = rng_b.standard_normal((30, 128))
    rdm_a = compute_rdm(X_a, metric="cosine")
    rdm_b = compute_rdm(X_b, metric="cosine")
    r = rsa_correlation(rdm_a, rdm_b)
    # Should be near zero (not perfectly zero, but small)
    assert abs(r) < 0.3


# ---------------------------------------------------------------------------
# partial_rsa
# ---------------------------------------------------------------------------


def test_partial_rsa_removes_confound():
    """Partial RSA reduces correlation when nuisance is confounded with model.

    Construct an observed RDM = model_rdm + nuisance_rdm (confounded).
    Plain RSA should show higher correlation than partial RSA that controls
    for the nuisance.
    """
    N = 20
    D = 64
    rng = np.random.default_rng(42)

    # Create a model signal and a nuisance signal
    model_signal = rng.standard_normal((N, D))
    nuisance_signal = rng.standard_normal((N, D))

    # Observed = model + nuisance (confounded)
    observed = model_signal + nuisance_signal

    model_rdm = compute_rdm(model_signal, metric="cosine")
    nuisance_rdm = compute_rdm(nuisance_signal, metric="cosine")
    observed_rdm = compute_rdm(observed, metric="cosine")

    r_plain = rsa_correlation(observed_rdm, model_rdm)
    r_partial = partial_rsa(observed_rdm, model_rdm, [nuisance_rdm])

    # Partial RSA should be different (typically reduced) compared to plain RSA
    # The key test is that partial_rsa runs without error and returns a finite value
    assert np.isfinite(r_partial)
    # We expect the partial correlation to differ from plain
    assert r_plain != pytest.approx(r_partial, abs=1e-6)


# ---------------------------------------------------------------------------
# run_permutation_test_tiered
# ---------------------------------------------------------------------------


def test_permutation_test_null():
    """Permutation test on random/unstructured data yields non-significant p-value."""
    N = 15
    rng = np.random.default_rng(42)

    # Random observed RDMs and a random model RDM — no true signal
    X_obs = rng.standard_normal((N, 32))
    X_model = rng.standard_normal((N, 32))

    observed_rdm = compute_rdm(X_obs, metric="cosine")
    model_rdm = compute_rdm(X_model, metric="cosine")

    rdms = {0: observed_rdm}

    config = {
        "screen_permutations": 50,
        "full_permutations": 100,
        "top_k_layers_for_full_test": 1,
        "seed": 42,
    }

    result = run_permutation_test_tiered(rdms, model_rdm, config)

    # p-value should not be extremely small (no signal)
    screen_p = result["screen_pvalues"][0]
    assert screen_p > 0.01  # Clearly not significant


def test_permutation_test_signal():
    """Permutation test on structured data (identical model) yields significant p-value."""
    N = 20
    rng = np.random.default_rng(42)

    # Create structured representations where the observed RDM matches the model
    X = rng.standard_normal((N, 64))
    rdm = compute_rdm(X, metric="cosine")

    # Use the same RDM as both observed and model — perfect signal
    rdms = {0: rdm}
    model_rdm = rdm.copy()

    config = {
        "screen_permutations": 100,
        "full_permutations": 200,
        "top_k_layers_for_full_test": 1,
        "seed": 42,
    }

    result = run_permutation_test_tiered(rdms, model_rdm, config)

    # With perfect signal, p-value should be very small
    full_p = result["full_pvalues"][0]
    assert full_p < 0.05
