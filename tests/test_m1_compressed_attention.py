# SPDX-License-Identifier: Apache-2.0
"""Tests for M1 compressed KV cache attention."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import m1_compressed_attention
from omlx.patches.m1_compressed_attention import CompressedKVCache

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def m1_enabled(monkeypatch):
    """Enable compressed attention for testing."""
    monkeypatch.setattr(m1_compressed_attention, "enabled", lambda: True)
    yield


def test_compress_decompress_roundtrip():
    """Test that compression/decompression is lossy but close."""
    mx.random.seed(42)
    key = mx.random.normal((1, 4, 100, 64)).astype(mx.float16)

    # Compress
    compressed = m1_compressed_attention.compress_key(key)
    assert compressed.dtype == mx.int8

    # Decompress
    scale = mx.abs(key).max().item()
    decompressed = m1_compressed_attention.decompress_key(compressed, scale)
    assert decompressed.dtype == mx.float16

    # Check that decompressed is close to original
    max_diff = mx.max(mx.abs(decompressed - key)).item()
    rel_diff = max_diff / mx.max(mx.abs(key)).item()
    # int8 quantization should give ~0.8% error
    assert rel_diff < 0.02, f"Relative difference too large: {rel_diff}"


def test_compressed_cache_basic():
    """Test basic compressed KV cache operations."""
    cache = CompressedKVCache(
        num_heads=4,
        head_dim=64,
        dtype=mx.float16,
        recent_window=10,
    )

    # Add some tokens
    for i in range(5):
        key = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        value = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        cache.update(key, value)

    assert cache.get_size() == 5
    keys = cache.get_keys()
    values = cache.get_values()
    assert keys.shape == (1, 4, 5, 64)
    assert values.shape == (1, 4, 5, 64)


def test_compressed_cache_triggers_compression():
    """Test that compression is triggered when cache exceeds recent window."""
    cache = CompressedKVCache(
        num_heads=4,
        head_dim=64,
        dtype=mx.float16,
        recent_window=5,
    )

    # Add tokens up to the window size
    for i in range(5):
        key = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        value = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        cache.update(key, value)

    # No compression yet
    assert cache.keys_old is None

    # Add one more token to trigger compression
    key = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
    value = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
    cache.update(key, value)

    # Now compression should have been triggered
    assert cache.keys_old is not None
    assert cache.keys_old.shape[2] == 1  # 1 old token compressed
    assert cache.keys_recent.shape[2] == 5  # 5 recent tokens in full precision


def test_compressed_cache_get_keys_after_compression():
    """Test that get_keys returns all keys after compression."""
    cache = CompressedKVCache(
        num_heads=4,
        head_dim=64,
        dtype=mx.float16,
        recent_window=5,
    )

    # Add tokens to trigger compression
    for i in range(7):
        key = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        value = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        cache.update(key, value)

    # Get all keys
    keys = cache.get_keys()
    assert keys.shape == (1, 4, 7, 64)
    assert keys.dtype == mx.float16


def test_compressed_cache_multiple_compression_cycles():
    """Test multiple compression cycles."""
    cache = CompressedKVCache(
        num_heads=4,
        head_dim=64,
        dtype=mx.float16,
        recent_window=4,
    )

    # Add tokens in batches to trigger multiple compression cycles
    for batch in range(3):
        for i in range(4):
            key = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
            value = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
            cache.update(key, value)

    # Total: 12 tokens
    assert cache.get_size() == 12

    # Get all keys
    keys = cache.get_keys()
    assert keys.shape == (1, 4, 12, 64)