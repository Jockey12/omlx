# SPDX-License-Identifier: Apache-2.0
"""Tests for M1 single-row decode kernel."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import m1_decode_qmm
from omlx.patches.m1_decode_qmm import M1DecodeLinear

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def m1_enabled(monkeypatch):
    """Enable the M1 decode kernel for testing."""
    monkeypatch.setattr(m1_decode_qmm, "enabled", lambda: True)
    yield


def _make_quantized_weight(K, N, bits, group_size, dtype=mx.float16):
    """Create a quantized weight tensor for testing."""
    mx.random.seed(42)
    weight = (mx.random.normal((N, K)) * 0.05).astype(dtype)
    return mx.quantize(weight, group_size=group_size, bits=bits)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_decode_matches_standard(m1_enabled, bits, group_size):
    """Verify decode kernel produces same results as standard path."""
    M, K, N = 1, 2048, 1024
    w, scales, biases = _make_quantized_weight(K, N, bits, group_size)
    x = mx.random.normal((M, K)).astype(mx.float16)

    # Standard path
    y_std = mx.quantized_matmul(
        x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
    )

    # M1 decode path
    y_decode = m1_decode_qmm.m1_decode_qmm(
        x, w, scales, biases, bits=bits, group_size=group_size
    )

    # Results should be identical (same kernel, different routing)
    assert y_decode.shape == y_std.shape
    assert mx.allclose(y_decode, y_std, atol=1e-5, rtol=1e-5).item()


def test_eligibility_gating(m1_enabled):
    """Test that eligibility checks work correctly."""
    # M != 1 (not decode)
    assert not m1_decode_qmm.eligible(2, 2048, 1024, 4, 64, mx.float16)

    # M == 1 (decode)
    assert m1_decode_qmm.eligible(1, 2048, 1024, 4, 64, mx.float16)

    # Wrong bits
    assert not m1_decode_qmm.eligible(1, 2048, 1024, 2, 64, mx.float16)

    # Wrong dtype
    assert not m1_decode_qmm.eligible(1, 2048, 1024, 4, 64, mx.float32)

    # N too small
    assert not m1_decode_qmm.eligible(1, 2048, 256, 4, 64, mx.float16)

    # K not divisible by group size
    assert not m1_decode_qmm.eligible(1, 2049, 1024, 4, 64, mx.float16)


def test_decode_linear_module(m1_enabled):
    """Test the M1DecodeLinear module."""
    M, K, N = 1, 2048, 1024
    linear = M1DecodeLinear(K, N, group_size=64, bits=4)

    # Set up weights
    w, scales, biases = _make_quantized_weight(K, N, 4, 64)
    linear.weight = w
    linear.scales = scales
    linear.biases = biases

    x = mx.random.normal((M, K)).astype(mx.float16)
    y = linear(x)

    assert y.shape == (M, N)
    assert y.dtype == mx.float16


def test_decode_linear_multirow_fallback(m1_enabled):
    """Test that multi-row inputs fall back to standard path."""
    M, K, N = 4, 2048, 1024
    linear = M1DecodeLinear(K, N, group_size=64, bits=4)

    # Set up weights
    w, scales, biases = _make_quantized_weight(K, N, 4, 64)
    linear.weight = w
    linear.scales = scales
    linear.biases = biases

    x = mx.random.normal((M, K)).astype(mx.float16)
    y = linear(x)

    # Should fall back to standard path for M > 1
    assert y.shape == (M, N)


def test_bf16_input(m1_enabled):
    """Test with bf16 input (M1 doesn't have native bf16 ALUs)."""
    M, K, N = 1, 2048, 1024
    w, scales, biases = _make_quantized_weight(K, N, 4, 64, dtype=mx.bfloat16)
    x = mx.random.normal((M, K)).astype(mx.bfloat16)

    y = m1_decode_qmm.m1_decode_qmm(
        x, w, scales, biases, bits=4, group_size=64
    )

    assert y.shape == (M, N)
    assert y.dtype == mx.bfloat16