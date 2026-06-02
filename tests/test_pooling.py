"""Tests for pooling functions (extraction.py — mean_pool_no_special, last_token_pool).

Mocks the 'transformers' and 'h5py' imports that extraction.py uses at module
level, since those are not needed for the pooling function tests.
"""

import sys
import types
import unittest.mock

import torch
import pytest

# Mock heavy imports that extraction.py needs at module level
_mock_transformers = types.ModuleType("transformers")
_mock_transformers.AutoModelForCausalLM = unittest.mock.MagicMock()
_mock_transformers.AutoTokenizer = unittest.mock.MagicMock()
sys.modules.setdefault("transformers", _mock_transformers)

_mock_h5py = types.ModuleType("h5py")
_mock_h5py.File = unittest.mock.MagicMock()
_mock_h5py.special_dtype = unittest.mock.MagicMock()
sys.modules.setdefault("h5py", _mock_h5py)

_mock_tqdm_mod = types.ModuleType("tqdm")
_mock_tqdm_mod.tqdm = lambda x, **kw: x
sys.modules.setdefault("tqdm", _mock_tqdm_mod)

from extraction import mean_pool_no_special, last_token_pool


# ---------------------------------------------------------------------------
# mean_pool_no_special
# ---------------------------------------------------------------------------


def test_mean_pool_excludes_special():
    """Mean pooling correctly excludes special (BOS/EOS/PAD) tokens."""
    # Setup: batch=1, seq_len=5, hidden_dim=4
    # Tokens: [BOS=1, tok_A=10, tok_B=20, EOS=2, PAD=0]
    # Special IDs: {0, 1, 2}
    # Only tok_A and tok_B should be averaged

    hidden_states = torch.tensor([
        [
            [100.0, 100.0, 100.0, 100.0],  # BOS (id=1) — exclude
            [2.0, 4.0, 6.0, 8.0],          # tok_A (id=10) — include
            [4.0, 8.0, 12.0, 16.0],        # tok_B (id=20) — include
            [999.0, 999.0, 999.0, 999.0],  # EOS (id=2) — exclude
            [0.0, 0.0, 0.0, 0.0],          # PAD (id=0) — exclude
        ]
    ], dtype=torch.float32)  # shape (1, 5, 4)

    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)  # PAD=0
    input_ids = torch.tensor([[1, 10, 20, 2, 0]], dtype=torch.long)
    special_ids = {0, 1, 2}

    result = mean_pool_no_special(hidden_states, attention_mask, special_ids, input_ids)

    # Expected: mean of tok_A and tok_B = (2+4)/2, (4+8)/2, (6+12)/2, (8+16)/2
    expected = torch.tensor([3.0, 6.0, 9.0, 12.0])
    assert torch.allclose(result, expected, atol=1e-5)


def test_mean_pool_shape():
    """Mean pool output shape is (hidden_dim,) for batch=1."""
    batch, seq_len, hidden_dim = 1, 8, 16
    hidden_states = torch.randn(batch, seq_len, hidden_dim)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    input_ids = torch.full((batch, seq_len), 10, dtype=torch.long)
    special_ids = {0, 1, 2}  # None of these appear in input_ids

    result = mean_pool_no_special(hidden_states, attention_mask, special_ids, input_ids)
    # With batch=1, the function squeezes dim 0 → shape (hidden_dim,)
    assert result.shape == (hidden_dim,)


# ---------------------------------------------------------------------------
# last_token_pool
# ---------------------------------------------------------------------------


def test_last_token_pool_correct_position():
    """Last token pool selects the last non-pad token's hidden state."""
    # Setup: batch=1, seq_len=5, hidden_dim=3
    # Attention mask: [1, 1, 1, 0, 0] → last attended token is index 2
    hidden_states = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],   # ← this should be selected (last attended)
            [99.0, 99.0, 99.0],  # PAD
            [99.0, 99.0, 99.0],  # PAD
        ]
    ], dtype=torch.float32)

    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    result = last_token_pool(hidden_states, attention_mask)
    expected = torch.tensor([7.0, 8.0, 9.0])
    assert torch.allclose(result, expected, atol=1e-5)


def test_last_token_pool_shape():
    """Last token pool output shape is (hidden_dim,) for batch=1."""
    batch, seq_len, hidden_dim = 1, 10, 32
    hidden_states = torch.randn(batch, seq_len, hidden_dim)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    result = last_token_pool(hidden_states, attention_mask)
    assert result.shape == (hidden_dim,)
