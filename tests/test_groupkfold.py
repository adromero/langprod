"""Tests for GroupKFold behavior as used in probes.py (train_probe_at_layer).

These tests validate the GroupKFold splitting logic with synthetic data
that mimics the actual project structure: products have multiple variants
(stimuli), and no product should leak across train/test splits.
"""

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

RNG = np.random.default_rng(42)


def _make_synthetic_data(n_products=20, variants_per_product=5, n_features=32):
    """Create synthetic data mimicking the project's product-variant structure.

    Each product has `variants_per_product` stimuli. The group array uses
    the product index as the group ID.
    """
    N = n_products * variants_per_product
    X = RNG.standard_normal((N, n_features))
    # Labels: product index (0..n_products-1), repeated for each variant
    labels = np.repeat(np.arange(n_products), variants_per_product)
    # Groups: same as product index (all variants of one product share a group)
    groups = np.repeat(np.arange(n_products), variants_per_product)
    return X, labels, groups


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_product_leaks():
    """No product_id (group) appears in both train and test sets."""
    X, labels, groups = _make_synthetic_data(n_products=20, variants_per_product=5)
    cv = GroupKFold(n_splits=5)

    for train_idx, test_idx in cv.split(X, labels, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        assert len(overlap) == 0, f"Product leak detected: groups {overlap} in both train and test"


def test_all_variants_together():
    """All variants of one product stay in the same split (all in train or all in test)."""
    X, labels, groups = _make_synthetic_data(n_products=20, variants_per_product=5)
    cv = GroupKFold(n_splits=5)

    for train_idx, test_idx in cv.split(X, labels, groups=groups):
        # For each unique group, check that ALL its indices are either
        # entirely in train or entirely in test
        for g in np.unique(groups):
            g_indices = set(np.where(groups == g)[0])
            in_train = g_indices & set(train_idx)
            in_test = g_indices & set(test_idx)
            # All indices should be in exactly one of {train, test}
            assert (len(in_train) == 0) or (len(in_test) == 0), (
                f"Group {g} split across train ({len(in_train)}) "
                f"and test ({len(in_test)})"
            )


def test_fold_balance():
    """Fold sizes are roughly balanced (no fold is 2x another)."""
    X, labels, groups = _make_synthetic_data(n_products=20, variants_per_product=5)
    cv = GroupKFold(n_splits=5)

    fold_sizes = []
    for _, test_idx in cv.split(X, labels, groups=groups):
        fold_sizes.append(len(test_idx))

    max_size = max(fold_sizes)
    min_size = min(fold_sizes)
    # With 20 groups and 5 folds, each fold gets 4 groups = 20 samples
    # Allow some tolerance but the ratio should be reasonable
    assert max_size <= 2 * min_size, (
        f"Fold sizes are imbalanced: {fold_sizes} "
        f"(max={max_size}, min={min_size})"
    )


def test_5_folds():
    """GroupKFold with n_splits=5 produces exactly 5 folds."""
    X, labels, groups = _make_synthetic_data(n_products=20, variants_per_product=5)
    cv = GroupKFold(n_splits=5)

    folds = list(cv.split(X, labels, groups=groups))
    assert len(folds) == 5
