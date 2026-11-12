# SPDX-License-Identifier: Apache-2.0
"""Fused scale application for ModelOpt mixed-format weights on M1.

The Unsloth Qwen3.8-27B checkpoint uses a mixed quantization format with
two different scale types:

- Channel-scale FP8 weights (attention, GDN, late MLPs): per-channel E4M3
  FP8 weights with FP32 channel scales.
- Tensor-scale NVFP4 weights (remaining MLPs): packed E2M1 NVFP4 weights
  with E4M3 group scales and an FP32 tensor divisor.

The generic loader maps these to MLX carriers and applies the global scale
as a separate multiply after the matmul. This kernel fuses the scale
application into the matmul operation, eliminating the second pass over
the output data.

On M1 (applegpu_g13), this avoids:
- An extra kernel launch for the multiply
- Writing the intermediate result to memory and reading it back
- The associated memory bandwidth cost (significant on M1's 400 GB/s bus)

Gating: applegpu_g13 only, controlled by OMLX_M1_MODELOPT_FUSED env var.
"""

from __future__ import annotations

import logging
import math
from functools import cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Minimum output dimension to justify the fused kernel overhead
_MIN_N = 256


@cache
def enabled() -> bool:
    """Check if the fused scale kernel should be used."""
    import os

    from omlx.utils.hardware import get_apple_gpu_generation

    env = os.environ.get("OMLX_M1_MODELOPT_FUSED", "1")
    if env == "0":
        return False

    gen = get_apple_gpu_generation()
    if gen == 13:
        logger.info("M1 ModelOpt fused scale kernel enabled")
        return True
    return False


def eligible(
    M: int,
    K: int,
    N: int,
    bits: int,
    group_size: int,
    dtype: Any,
    has_global_scale: bool,
) -> bool:
    """Check if the fused scale kernel is eligible for this operation.

    Args:
        M: Input row count
        K: Input column count (reduction dimension)
        N: Output column count
        bits: Weight bit width
        group_size: Quantization group size
        dtype: Input/output data type
        has_global_scale: Whether a global scale is present

    Returns:
        True if the fused kernel should be used
    """
    if not enabled() or not has_global_scale:
        return False

    # Only for quantized operations with a global scale to fuse
    if bits not in (4, 8):
        return False

    # Group size must be compatible with the kernel
    if bits == 4 and group_size not in (16, 32, 64, 128):
        return False
    if bits == 8 and group_size not in (32, 64, 128):
        return False

    # Output must be large enough to justify the kernel overhead
    if N < _MIN_N:
        return False

    # Input/output dtype must be fp16 or bf16
    if dtype not in (mx.float16, mx.bfloat16):
        return False

    # K must be divisible by the group size
    if K % group_size != 0:
        return False

    return True


def fused_qmm(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    global_scale: mx.array,
    *,
    bits: int,
    group_size: int,
    mode: str = "mxfp8",
    biases: mx.array | None = None,
) -> mx.array:
    """Fused quantized matmul with global scale application.

    Computes: (x @ weight.T * scales + biases) * global_scale

    On M1, this fuses the global scale into the per-group scales to avoid
    a separate multiply kernel launch, saving memory bandwidth.

    Args:
        x: Input array of shape (M, K)
        weight: Packed quantized weights of shape (N, K * bits // 32)
        scales: Per-group scales of shape (N, K // group_size)
        global_scale: Global scale factor (scalar or per-channel)
        bits: Weight bit width
        group_size: Quantization group size
        mode: Quantization mode ("mxfp8" or "nvfp4")
        biases: Optional per-group biases

    Returns:
        Output array of shape (M, N)
    """
    M, K = x.shape
    N = weight.shape[0]

    # Fold the global scale into the per-group scales and biases to avoid
    # a separate multiply kernel launch. This is the key optimization for M1.
    if global_scale.ndim == 0:
        # Scalar scale: multiply all per-group scales and biases
        fused_scales = scales * global_scale
        fused_biases = biases * global_scale if biases is not None else None
    else:
        # Per-channel scale: broadcast multiply
        fused_scales = scales * global_scale[:, None]
        if biases is not None:
            fused_biases = biases * global_scale[:, None]
        else:
            fused_biases = None

    # Cast back to the original scale type to avoid type promotion issues
    fused_scales = fused_scales.astype(scales.dtype)
    if fused_biases is not None:
        fused_biases = fused_biases.astype(biases.dtype)

    # Now perform the quantized matmul with the fused scales and biases
    # Note: mode is not passed here because the scale type determines the mode.
    # float16 scales use the default mode, uint8 scales use mxfp8/nvfp4 mode.
    result = mx.quantized_matmul(
        x,
        weight,
        fused_scales,
        fused_biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )

    return result


class FusedScaledQuantizedLinear(nn.Module):
    """Quantized linear layer with fused global scale application.

    Uses the fused scale kernel on M1 to avoid a separate multiply pass.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        group_size: int,
        bits: int,
        mode: str,
        scale_shape: tuple[int, ...],
        bias: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.scale_shape = scale_shape
        self.has_bias = bias

        # Quantized weight
        self.weight = mx.zeros(
            (output_dims, input_dims * bits // 32), dtype=mx.uint32
        )
        # Per-group scales
        self.scales = mx.zeros(
            (output_dims, input_dims // group_size), dtype=mx.uint8
        )
        # Global scale
        self.global_scale = mx.ones(scale_shape, dtype=mx.float32)
        # Optional bias
        if bias:
            self.bias = mx.zeros((output_dims,))

        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with fused scale application."""
        # Check if we should use the fused kernel
        M, K = x.shape
        N = self.output_dims
        if eligible(M, K, N, self.bits, self.group_size, x.dtype, True):
            return fused_qmm(
                x,
                self.weight,
                self.scales,
                self.global_scale,
                bits=self.bits,
                group_size=self.group_size,
                mode=self.mode,
                biases=self.get("biases"),
            )

        # Fall back to the standard path
        return self._standard_forward(x)

    def _standard_forward(self, x: mx.array) -> mx.array:
        """Standard forward pass without fused scale."""
        result = mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        # Apply global scale separately
        if self.global_scale.ndim == 0:
            result = result * self.global_scale
        else:
            result = result * self.global_scale[None, :]

        if self.has_bias:
            result = result + self.bias
        return result