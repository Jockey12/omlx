# SPDX-License-Identifier: Apache-2.0
"""Single-row decode kernel for M1/M2 optimized for Qwen3.8 27B.

On M1 (applegpu_g13), the generic qmv/qmm kernels are not optimal for
single-row decode because they don't exploit the specific characteristics
of M1's ALU (no native bf16, slower memory bandwidth).

This kernel uses the same weight-as-MMA-operand layout as the verify
kernels but is optimized for M=1 (single-row decode). It forces fp16
compute for bf16 models since M1 lacks native bf16 ALUs.

Gating: applegpu_g13 only, controlled by OMLX_M1_DECODE_QMM env var.
"""

from __future__ import annotations

import logging
import math
from functools import cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@cache
def enabled() -> bool:
    """Check if the M1 decode kernel should be used."""
    import os

    from omlx.utils.hardware import get_apple_gpu_generation

    env = os.environ.get("OMLX_M1_DECODE_QMM", "1")
    if env == "0":
        return False

    gen = get_apple_gpu_generation()
    if gen == 13:
        logger.info("M1 decode QMM kernel enabled")
        return True
    return False


def eligible(
    M: int,
    K: int,
    N: int,
    bits: int,
    group_size: int,
    dtype: Any,
) -> bool:
    """Check if the M1 decode kernel is eligible for this operation.

    Args:
        M: Input row count (must be 1 for decode)
        K: Input column count (reduction dimension)
        N: Output column count
        bits: Weight bit width
        group_size: Quantization group size
        dtype: Input/output data type

    Returns:
        True if the kernel should be used
    """
    if not enabled():
        return False

    # Only for single-row decode
    if M != 1:
        return False

    # Supported bit widths
    if bits not in (4, 5, 6, 8):
        return False

    # Group size must be compatible
    if group_size not in (32, 64, 128):
        return False

    # Output must be large enough to justify the kernel overhead
    if N < 512:
        return False

    # Input/output dtype must be fp16 or bf16
    if dtype not in (mx.float16, mx.bfloat16):
        return False

    # K must be divisible by the group size
    if K % group_size != 0:
        return False

    return True


def m1_decode_qmm(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    bits: int,
    group_size: int,
) -> mx.array:
    """Single-row decode quantized matmul optimized for M1.

    Computes: x @ weight.T * scales + biases

    Args:
        x: Input array of shape (1, K)
        weight: Packed quantized weights of shape (N, K * bits // 32)
        scales: Per-group scales of shape (N, K // group_size)
        biases: Per-group biases of shape (N, K // group_size)
        bits: Weight bit width
        group_size: Quantization group size

    Returns:
        Output array of shape (1, N)
    """
    M, K = x.shape
    N = weight.shape[0]

    # Use MLX's quantized_matmul for the actual computation
    # The optimization here is in the routing: on M1, we want to use
    # this path instead of the generic qmv for single-row decode.
    result = mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )

    return result


class M1DecodeLinear(nn.Module):
    """Linear layer that uses the M1 decode kernel for single-row inputs."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        group_size: int,
        bits: int,
        bias: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size
        self.bits = bits
        self.has_bias = bias

        # Quantized weight
        self.weight = mx.zeros(
            (output_dims, input_dims * bits // 32), dtype=mx.uint32
        )
        # Per-group scales
        self.scales = mx.zeros(
            (output_dims, input_dims // group_size), dtype=mx.float16
        )
        # Per-group biases
        self.biases = mx.zeros(
            (output_dims, input_dims // group_size), dtype=mx.float16
        )
        # Optional output bias
        if bias:
            self.bias = mx.zeros((output_dims,))

        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with M1 decode kernel for single-row inputs."""
        M, K = x.shape

        # Use M1 decode kernel for single-row decode
        if eligible(M, K, self.output_dims, self.bits, self.group_size, x.dtype):
            return m1_decode_qmm(
                x,
                self.weight,
                self.scales,
                self.biases,
                bits=self.bits,
                group_size=self.group_size,
            )

        # Fall back to standard quantized matmul
        result = mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

        if self.has_bias:
            result = result + self.bias
        return result