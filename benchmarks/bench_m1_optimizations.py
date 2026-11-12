#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark M1/M2 optimizations for Qwen3.8 27B.

Measures the actual performance gains of:
1. Fused scale application (m1_modelopt_fused)
2. Single-row decode kernel (m1_decode_qmm)
3. Compressed KV attention (m1_compressed_attention)
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import mlx.core as mx
import numpy as np


def benchmark_qmm_fused_scale(
    M: int, K: int, N: int, bits: int, group_size: int
) -> dict[str, Any]:
    """Benchmark fused vs standard scale application."""
    from omlx.patches import m1_modelopt_fused

    # Create quantized weight
    mx.random.seed(42)
    weight = (mx.random.normal((N, K)) * 0.05).astype(mx.float16)
    wq, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)

    x = mx.random.normal((M, K)).astype(mx.float16)
    global_scale = mx.array(0.5, dtype=mx.float32)

    # Warmup
    for _ in range(5):
        mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        mx.eval()

    # Benchmark standard path (5 runs, report median)
    # Force separate kernel launches by evaluating after qmm
    times_std = []
    for _ in range(5):
        start = time.perf_counter()
        y_qmm = mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        mx.eval(y_qmm)  # Force kernel launch
        y_std = y_qmm * global_scale
        mx.eval(y_std)
        times_std.append(time.perf_counter() - start)

    # Pre-compute fused scales (done once during loading, not during inference)
    fused_scales, fused_biases = m1_modelopt_fused.compute_fused_scales(
        scales, global_scale, biases
    )

    # Benchmark fused path (5 runs, report median)
    times_fused = []
    for _ in range(5):
        start = time.perf_counter()
        y_fused = m1_modelopt_fused.fused_qmm(
            x, wq, fused_scales, fused_biases, bits=bits, group_size=group_size
        )
        mx.eval(y_fused)
        times_fused.append(time.perf_counter() - start)

    # Verify correctness
    assert mx.allclose(y_std, y_fused, atol=1e-3, rtol=1e-3).item()

    std_ms = np.median(times_std) * 1000
    fused_ms = np.median(times_fused) * 1000
    speedup = std_ms / fused_ms

    return {
        "M": M,
        "K": K,
        "N": N,
        "bits": bits,
        "group_size": group_size,
        "std_ms": std_ms,
        "fused_ms": fused_ms,
        "speedup": speedup,
    }


def benchmark_decode_kernel(
    K: int, N: int, bits: int, group_size: int
) -> dict[str, Any]:
    """Benchmark single-row decode kernel."""
    from omlx.patches import m1_decode_qmm

    # Create quantized weight
    mx.random.seed(42)
    weight = (mx.random.normal((N, K)) * 0.05).astype(mx.float16)
    wq, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)

    x = mx.random.normal((1, K)).astype(mx.float16)

    # Warmup
    for _ in range(3):
        mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        mx.eval()

    # Benchmark standard path (5 runs, report median)
    times_std = []
    for _ in range(5):
        start = time.perf_counter()
        y_std = mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        mx.eval(y_std)
        times_std.append(time.perf_counter() - start)

    # Benchmark M1 decode path (5 runs, report median)
    times_decode = []
    for _ in range(5):
        start = time.perf_counter()
        y_decode = m1_decode_qmm.m1_decode_qmm(
            x, wq, scales, biases, bits=bits, group_size=group_size
        )
        mx.eval(y_decode)
        times_decode.append(time.perf_counter() - start)

    # Verify correctness
    assert mx.allclose(y_std, y_decode, atol=1e-5, rtol=1e-5).item()

    std_ms = np.median(times_std) * 1000
    decode_ms = np.median(times_decode) * 1000
    speedup = std_ms / decode_ms

    return {
        "K": K,
        "N": N,
        "bits": bits,
        "group_size": group_size,
        "std_ms": std_ms,
        "decode_ms": decode_ms,
        "speedup": speedup,
    }


def benchmark_compressed_attention(
    seq_len: int, num_heads: int, head_dim: int
) -> dict[str, Any]:
    """Benchmark compressed vs standard attention."""
    from omlx.patches.m1_compressed_attention import CompressedKVCache

    # Standard attention: full precision KV cache
    # Compressed attention: int8 keys, fp16 values for older tokens

    # Create test data
    mx.random.seed(42)
    query = mx.random.normal((1, num_heads, 1, head_dim)).astype(mx.float16)

    # Build standard KV cache
    keys_std = mx.random.normal((1, num_heads, seq_len, head_dim)).astype(mx.float16)
    values_std = mx.random.normal((1, num_heads, seq_len, head_dim)).astype(mx.float16)

    # Build compressed KV cache
    cache = CompressedKVCache(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=mx.float16,
        recent_window=4096,
    )

    # Add tokens in batches
    batch_size = 128
    for i in range(0, seq_len, batch_size):
        end = min(i + batch_size, seq_len)
        k = keys_std[:, :, i:end]
        v = values_std[:, :, i:end]
        cache.update(k, v)

    # Warmup: standard attention
    for _ in range(3):
        scores = query @ keys_std.transpose(0, 1, 3, 2)
        mx.eval(scores)

    # Benchmark standard attention (5 runs, report median)
    times_std = []
    for _ in range(5):
        start = time.perf_counter()
        scores = query @ keys_std.transpose(0, 1, 3, 2)
        mx.eval(scores)
        times_std.append(time.perf_counter() - start)

    # Warmup: compressed attention
    for _ in range(5):
        keys_comp = cache.get_keys()
        scores = query @ keys_comp.transpose(0, 1, 3, 2)
        mx.eval(scores)

    # Benchmark compressed attention (5 runs, report median)
    times_comp = []
    for _ in range(5):
        start = time.perf_counter()
        keys_comp = cache.get_keys()
        scores = query @ keys_comp.transpose(0, 1, 3, 2)
        mx.eval(scores)
        times_comp.append(time.perf_counter() - start)

    std_ms = np.median(times_std) * 1000
    comp_ms = np.median(times_comp) * 1000
    speedup = std_ms / comp_ms

    return {
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "std_ms": std_ms,
        "comp_ms": comp_ms,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark M1 optimizations")
    parser.add_argument("--fused", action="store_true", help="Benchmark fused scale")
    parser.add_argument("--decode", action="store_true", help="Benchmark decode kernel")
    parser.add_argument("--attention", action="store_true", help="Benchmark compressed attention")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if not (args.fused or args.decode or args.attention or args.all):
        args.all = True

    print("=" * 80)
    print("M1/M2 Optimization Benchmarks for Qwen3.8 27B")
    print("=" * 80)

    if args.fused or args.all:
        print("\n--- Fused Scale Application ---")
        # Test with qwen3.8 27b shapes: hidden=5120, intermediate=18432
        for K, N in [(5120, 18432), (18432, 5120)]:
            result = benchmark_qmm_fused_scale(1, K, N, 4, 64)
            print(
                f"M={result['M']}, K={result['K']}, N={result['N']}: "
                f"std={result['std_ms']:.3f}ms, fused={result['fused_ms']:.3f}ms, "
                f"speedup={result['speedup']:.3f}x"
            )

    if args.decode or args.all:
        print("\n--- Single-Row Decode Kernel ---")
        for K, N in [(5120, 18432), (18432, 5120)]:
            result = benchmark_decode_kernel(K, N, 4, 64)
            print(
                f"K={result['K']}, N={result['N']}: "
                f"std={result['std_ms']:.3f}ms, decode={result['decode_ms']:.3f}ms, "
                f"speedup={result['speedup']:.3f}x"
            )

    if args.attention or args.all:
        print("\n--- Compressed KV Attention ---")
        for seq_len in [4096, 8192, 16384]:
            result = benchmark_compressed_attention(seq_len, 128, 128)
            print(
                f"seq_len={result['seq_len']}: "
                f"std={result['std_ms']:.3f}ms, comp={result['comp_ms']:.3f}ms, "
                f"speedup={result['speedup']:.3f}x"
            )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()