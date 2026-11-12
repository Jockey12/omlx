#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark KV cache memory footprint for standard vs compressed attention."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from omlx.patches.m1_compressed_attention import CompressedKVCache


def measure_memory(seq_len: int, num_heads: int, head_dim: int) -> dict:
    """Measure memory footprint of standard vs compressed KV cache."""
    # Standard KV cache: fp16 keys and values
    keys_std = mx.random.normal((1, num_heads, seq_len, head_dim)).astype(mx.float16)
    values_std = mx.random.normal((1, num_heads, seq_len, head_dim)).astype(mx.float16)

    std_key_bytes = keys_std.nbytes
    std_val_bytes = values_std.nbytes
    std_total = std_key_bytes + std_val_bytes

    # Compressed KV cache: int8 keys, fp16 values
    cache = CompressedKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=mx.float16,
        recent_window=4096,
    )

    # Add all tokens
    batch_size = 128
    for i in range(0, seq_len, batch_size):
        end = min(i + batch_size, seq_len)
        k = mx.random.normal((1, num_heads, end - i, head_dim)).astype(mx.float16)
        v = mx.random.normal((1, num_heads, end - i, head_dim)).astype(mx.float16)
        cache.update(k, v)

    # Measure compressed cache size
    if cache.keys_old is not None:
        # Old keys are int8
        old_key_bytes = cache.keys_old.nbytes
        # Recent keys are fp16
        recent_key_bytes = cache.keys_recent.nbytes
        # Values are all fp16
        val_bytes = cache.values_old.nbytes + cache.values_recent.nbytes
        comp_total = old_key_bytes + recent_key_bytes + val_bytes
    else:
        comp_total = cache.keys_recent.nbytes + cache.values_recent.nbytes

    return {
        "seq_len": seq_len,
        "std_bytes": std_total,
        "comp_bytes": comp_total,
        "savings_bytes": std_total - comp_total,
        "savings_pct": (std_total - comp_total) / std_total * 100,
    }


def main():
    print("=" * 80)
    print("KV Cache Memory Footprint: Standard vs Compressed")
    print("=" * 80)

    num_heads = 128
    head_dim = 128

    for seq_len in [4096, 8192, 16384, 32768]:
        result = measure_memory(seq_len, num_heads, head_dim)
        print(f"\nseq_len={result['seq_len']}:")
        print(f"  Standard: {result['std_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Compressed: {result['comp_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Savings: {result['savings_bytes'] / 1024 / 1024:.2f} MB ({result['savings_pct']:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()