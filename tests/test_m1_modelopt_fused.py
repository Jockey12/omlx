# SPDX-License-Identifier: Apache-2.0
"""Tests for M1 ModelOpt fused scale kernel."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import m1_modelopt_fused
from omlx.patches.m1_modelopt_fused import FusedScaledQuantizedLinear

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def m1_enabled(monkeypatch):
    """Enable the M1 fused scale kernel for testing."""
    monkeypatch.setattr(m1_modelopt_fused, "enabled", lambda: True)
    yield


def _make_quantized_weight(K, N, bits, group_size, dtype=mx.float16):
    """Create a quantized weight tensor for testing."""
    mx.random.seed(42)
    weight = (mx.random.normal((N, K)) * 0.05).astype(dtype)
    return mx.quantize(weight, group_size=group_size, bits=bits)


def _make_mxfp8_weight(K, N, bits, group_size):
    """Create a quantized weight tensor with uint8 E4M3 FP8 scales for testing."""
    mx.random.seed(42)
    weight = (mx.random.normal((N, K)) * 0.05).astype(mx.float16)
    wq, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    # Convert float16 scales to uint8 E4M3 FP8
    scales_fp8 = mx.to_fp8(scales)
    return wq, scales_fp8, biases


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_fused_matches_standard(m1_enabled, bits, group_size):
    """Verify fused kernel produces same results as standard path."""
    M, K, N = 1, 2048, 1024
    w, scales, biases = _make_quantized_weight(K, N, bits, group_size)
    x = mx.random.normal((M, K)).astype(mx.float16)
    global_scale = mx.array(0.5, dtype=mx.float32)

    # Standard path: apply global scale separately
    y_std = mx.quantized_matmul(
        x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
    )
    y_std = y_std * global_scale

    # Fused path: compute fused scales once, then use fused_qmm
    fused_scales, fused_biases = m1_modelopt_fused.compute_fused_scales(
        scales, global_scale, bits, group_size, biases
    )
    y_fused = mx.quantized_matmul(
        x, w, fused_scales, fused_biases, transpose=True, group_size=group_size, bits=bits
    )

    # Results should be close (not bit-identical due to different accumulation order)
    assert y_fused.shape == y_std.shape
    max_diff = mx.max(mx.abs(y_fused - y_std)).item()
    rel_diff = max_diff / mx.max(mx.abs(y_std)).item()
    assert rel_diff < 1e-2, f"Relative difference too large: {rel_diff}"


def test_eligibility_gating(m1_enabled):
    """Test that eligibility checks work correctly."""
    # Too small N
    assert not m1_modelopt_fused.eligible(1, 2048, 128, 4, 64, mx.float16, True)

    # Wrong bits
    assert not m1_modelopt_fused.eligible(1, 2048, 1024, 2, 64, mx.float16, True)

    # Wrong dtype
    assert not m1_modelopt_fused.eligible(1, 2048, 1024, 4, 64, mx.float32, True)

    # No global scale
    assert not m1_modelopt_fused.eligible(1, 2048, 1024, 4, 64, mx.float16, False)

    # K not divisible by group size
    assert not m1_modelopt_fused.eligible(1, 2049, 1024, 4, 64, mx.float16, True)

    # Eligible case
    assert m1_modelopt_fused.eligible(1, 2048, 1024, 4, 64, mx.float16, True)


def test_fused_linear_module(m1_enabled):
    """Test the FusedScaledQuantizedLinear module."""
    M, K, N = 1, 2048, 1024
    linear = FusedScaledQuantizedLinear(
        K, N, group_size=64, bits=4, mode="mxfp8", scale_shape=(N,)
    )

    # Set up weights with uint8 E4M3 FP8 scales
    w, scales, biases = _make_mxfp8_weight(K, N, 4, 64)
    linear.weight = w
    linear.scales = scales
    linear.biases = biases
    linear.global_scale = mx.ones((N,), dtype=mx.float32) * 0.5

    x = mx.random.normal((M, K)).astype(mx.float16)
    y = linear(x)

    assert y.shape == (M, N)
    assert y.dtype == mx.float16


def test_per_channel_scale(m1_enabled):
    """Test with per-channel global scale."""
    M, K, N = 1, 2048, 1024
    w, scales, biases = _make_quantized_weight(K, N, 4, 64)
    x = mx.random.normal((M, K)).astype(mx.float16)

    # Per-channel scale
    global_scale = mx.random.uniform(0.4, 0.6, (N,)).astype(mx.float32)

    # Standard path: apply global scale separately
    y_std = mx.quantized_matmul(
        x, w, scales, biases, transpose=True, group_size=64, bits=4
    )
    y_std = y_std * global_scale[None, :]

    # Fused path: compute fused scales once, then use fused_qmm
    fused_scales, fused_biases = m1_modelopt_fused.compute_fused_scales(
        scales, global_scale, 4, 64, biases
    )
    y_fused = mx.quantized_matmul(
        x, w, fused_scales, fused_biases, transpose=True, group_size=64, bits=4
    )

    assert y_fused.shape == y_std.shape
    max_diff = mx.max(mx.abs(y_fused - y_std)).item()
    rel_diff = max_diff / mx.max(mx.abs(y_std)).item()
    assert rel_diff < 1e-2, f"Relative difference too large: {rel_diff}"


def test_3d_input(m1_enabled):
    """Test with 3D input (batch, seq, hidden)."""
    B, S, K, N = 2, 4, 2048, 1024
    w, scales, biases = _make_quantized_weight(K, N, 4, 64)
    x = mx.random.normal((B, S, K)).astype(mx.float16)
    global_scale = mx.array(0.5, dtype=mx.float32)

    # Standard path: apply global scale separately
    x_2d = x.reshape(B * S, K)
    y_std = mx.quantized_matmul(
        x_2d, w, scales, biases, transpose=True, group_size=64, bits=4
    )
    y_std = y_std * global_scale
    y_std = y_std.reshape(B, S, N)

    # Fused path
    fused_scales, fused_biases = m1_modelopt_fused.compute_fused_scales(
        scales, global_scale, 4, 64, biases
    )
    y_fused = mx.quantized_matmul(
        x, w, fused_scales, fused_biases, transpose=True, group_size=64, bits=4
    )

    assert y_fused.shape == y_std.shape
    max_diff = mx.max(mx.abs(y_fused - y_std)).item()
    rel_diff = max_diff / mx.max(mx.abs(y_std)).item()
    assert rel_diff < 1e-2, f"Relative difference too large: {rel_diff}"


def test_with_bias(m1_enabled):
    """Test with output bias."""
    M, K, N = 1, 2048, 1024
    w, scales, biases = _make_quantized_weight(K, N, 4, 64)
    x = mx.random.normal((M, K)).astype(mx.float16)
    global_scale = mx.array(0.5, dtype=mx.float32)
    output_bias = mx.random.normal((N,)).astype(mx.float16)

    # Standard path: apply global scale separately
    y_std = mx.quantized_matmul(
        x, w, scales, biases, transpose=True, group_size=64, bits=4
    )
    y_std = y_std * global_scale + output_bias[None, :]

    # Fused path
    fused_scales, fused_biases = m1_modelopt_fused.compute_fused_scales(
        scales, global_scale, 4, 64, biases
    )
    y_fused = mx.quantized_matmul(
        x, w, fused_scales, fused_biases, transpose=True, group_size=64, bits=4
    )
    y_fused = y_fused + output_bias[None, :]

    assert y_fused.shape == y_std.shape
    max_diff = mx.max(mx.abs(y_fused - y_std)).item()
    rel_diff = max_diff / mx.max(mx.abs(y_std)).item()
    assert rel_diff < 1e-2, f"Relative difference too large: {rel_diff}"