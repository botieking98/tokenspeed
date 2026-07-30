# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tokenspeed_kernel.profiling import bootstrap_profiling_from_env

bootstrap_profiling_from_env()

# Kernel ops may require CUDA-specific backends; import resiliently so the
# package can be loaded on Ascend NPU where only a subset of ops exist.
try:
    from tokenspeed_kernel.ops.attention import (
        attn_merge_state,
        attn_plan,
        mha_decode_with_kvcache,
        mha_extend_with_kvcache,
        mha_prefill,
        mla_decode_with_kvcache,
        mla_prefill,
    )
except (ImportError, ModuleNotFoundError):
    attn_merge_state = None
    attn_plan = None
    mha_decode_with_kvcache = None
    mha_extend_with_kvcache = None
    mha_prefill = None
    mla_decode_with_kvcache = None
    mla_prefill = None

try:
    from tokenspeed_kernel.ops.gemm import mm
except (ImportError, ModuleNotFoundError):
    mm = None

try:
    from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
except (ImportError, ModuleNotFoundError):
    moe_apply = None
    moe_plan = None
    moe_process_weights = None

try:
    from tokenspeed_kernel.ops.quantization import (
        quantize_fp8,
        quantize_fp8_with_scale,
        quantize_mxfp4,
        quantize_mxfp8,
        quantize_nvfp4,
    )
except (ImportError, ModuleNotFoundError):
    quantize_fp8 = None
    quantize_fp8_with_scale = None
    quantize_mxfp4 = None
    quantize_mxfp8 = None
    quantize_nvfp4 = None

try:
    from tokenspeed_kernel.ops.sampling import argmax
except (ImportError, ModuleNotFoundError):
    argmax = None

try:
    from tokenspeed_kernel.selection import NoKernelFoundError
except (ImportError, ModuleNotFoundError):
    class NoKernelFoundError(Exception):
        pass

__all__ = [
    # exceptions
    "NoKernelFoundError",
    # gemm
    "mm",
    # attention
    "attn_plan",
    "mha_prefill",
    "mha_extend_with_kvcache",
    "mha_decode_with_kvcache",
    "mla_prefill",
    "mla_decode_with_kvcache",
    "attn_merge_state",
    # moe
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
    # quantization
    "quantize_fp8",
    "quantize_fp8_with_scale",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
    # sampling
    "argmax",
]
