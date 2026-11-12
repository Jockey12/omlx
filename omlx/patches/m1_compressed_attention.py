# SPDX-License-Identifier: Apache-2.0
"""Compressed KV cache attention for M1/M2 long-context inference.

On M1/M2, the unified memory bandwidth (400 GB/s on M1 Max) becomes a
bottleneck at long contexts (>8k tokens) because the KV cache grows
linearly with context length. This kernel compresses older KV entries
to reduce memory bandwidth pressure.

Strategy:
- Recent tokens (last `recent_window` tokens): full precision (fp16/bf16)
- Older tokens: compressed (int8 for keys, fp16 for values)

The compression is applied during KV cache updates, and decompression
happens on-the-fly during attention computation.

Gating: applegpu_g13 only, controlled by OMLX_M1_COMPRESSED_ATTN env var.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Number of recent tokens to keep in full precision
DEFAULT_RECENT_WINDOW = 4096


@cache
def enabled() -> bool:
    """Check if compressed attention should be used."""
    import os

    from omlx.utils.hardware import get_apple_gpu_generation

    env = os.environ.get("OMLX_M1_COMPRESSED_ATTN", "1")
    if env == "0":
        return False

    gen = get_apple_gpu_generation()
    if gen == 13:
        logger.info("M1 compressed attention enabled (recent_window=%d)", DEFAULT_RECENT_WINDOW)
        return True
    return False


def compress_key(key: mx.array) -> mx.array:
    """Compress a key tensor from fp16 to int8.

    Args:
        key: Key tensor of shape (..., seq_len, head_dim) in fp16

    Returns:
        Compressed key tensor in int8
    """
    # Scale to [-127, 127] range for int8
    max_val = mx.abs(key).max()
    if max_val == 0:
        return key.astype(mx.int8)
    scaled = (key / max_val * 127.0).astype(mx.int8)
    return scaled


def decompress_key(key_compressed: mx.array, scale: float) -> mx.array:
    """Decompress an int8 key tensor back to fp16.

    Args:
        key_compressed: Compressed key tensor in int8
        scale: Scale factor used during compression

    Returns:
        Decompressed key tensor in fp16
    """
    return (key_compressed.astype(mx.float16) * scale / 127.0)


class CompressedKVCache:
    """KV cache with compression for older entries.

    Recent tokens (last `recent_window` tokens) are stored in full precision.
    Older tokens are compressed to int8 (keys) / fp16 (values).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype,
        recent_window: int = DEFAULT_RECENT_WINDOW,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.recent_window = recent_window

        # Full precision cache for recent tokens
        self.keys_recent = None
        self.values_recent = None

        # Compressed cache for older tokens
        self.keys_old = None
        self.values_old = None
        self.key_scale = 0.0

        # Cache size tracking
        self.size = 0

        # Cached decompressed keys (invalidated when cache is updated)
        self._cached_keys = None
        self._cached_size = 0

    def update(self, key: mx.array, value: mx.array) -> None:
        """Update the cache with new key/value tensors.

        Args:
            key: Key tensor of shape (batch, heads, seq_len, head_dim)
            value: Value tensor of shape (batch, heads, seq_len, head_dim)
        """
        batch, heads, seq_len, head_dim = key.shape

        if self.keys_recent is None:
            # First update
            self.keys_recent = key
            self.values_recent = value
            self.size = seq_len
            return

        # Append new tokens to recent cache
        self.keys_recent = mx.concatenate([self.keys_recent, key], axis=2)
        self.values_recent = mx.concatenate([self.values_recent, value], axis=2)
        self.size += seq_len

        # Invalidate cached decompressed keys
        self._cached_keys = None

        # Check if we need to compress older tokens
        if self.size > self.recent_window:
            # Compress the oldest tokens
            self._compress_old_tokens()

    def _compress_old_tokens(self) -> None:
        """Compress tokens older than the recent window."""
        # Keep only the most recent `recent_window` tokens in full precision
        if self.keys_recent.shape[2] > self.recent_window:
            # Split into old and recent
            old_len = self.keys_recent.shape[2] - self.recent_window
            keys_old = self.keys_recent[:, :, :old_len]
            values_old = self.values_recent[:, :, :old_len]
            self.keys_recent = self.keys_recent[:, :, old_len:]
            self.values_recent = self.values_recent[:, :, old_len:]

            # Compress old tokens
            self.key_scale = mx.abs(keys_old).max().item()
            keys_old_compressed = compress_key(keys_old)

            if self.keys_old is None:
                self.keys_old = keys_old_compressed
                self.values_old = values_old
            else:
                self.keys_old = mx.concatenate([self.keys_old, keys_old_compressed], axis=2)
                self.values_old = mx.concatenate([self.values_old, values_old], axis=2)

    def get_keys(self) -> mx.array:
        """Get all keys (decompressing old tokens on-the-fly).

        Caches the decompressed keys to avoid repeated decompression.
        """
        if self.keys_old is None:
            return self.keys_recent

        # Return cached decompressed keys if available
        if self._cached_keys is not None and self._cached_size == self.size:
            return self._cached_keys

        # Decompress old keys
        keys_old_decompressed = decompress_key(self.keys_old, self.key_scale)

        # Concatenate with recent keys
        result = mx.concatenate([keys_old_decompressed, self.keys_recent], axis=2)

        # Cache the result
        self._cached_keys = result
        self._cached_size = self.size

        return result

    def get_values(self) -> mx.array:
        """Get all values."""
        if self.values_old is None:
            return self.values_recent

        return mx.concatenate([self.values_old, self.values_recent], axis=2)

    def get_size(self) -> int:
        """Get the total number of tokens in the cache."""
        return self.size