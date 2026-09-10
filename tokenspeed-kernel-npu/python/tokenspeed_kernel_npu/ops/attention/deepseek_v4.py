# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""DeepSeek-V4 Flash Ascend operators.

TokenSpeed owns ``npu_scatter_nd_update_v2``. Other custom operators come from
the vllm-ascend-derived extension bundled inside ``tokenspeed-kernel-npu``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tokenspeed_kernel_npu._ascend import register_cann_vendors
from tokenspeed_kernel_npu._vllm_ascend import ensure_vllm_ascend_ops


_PACKAGE_LIBRARY_PATH = (
    Path(__file__).resolve().parents[2] / "lib" / "libtokenspeed_ascend_c.so"
)
_BUILD_LIBRARY_PATH = (
    Path(__file__).resolve().parents[4]
    / "build"
    / "ascend"
    / "tokenspeed-extension"
    / "libtokenspeed_ascend_c.so"
)
_LIBRARY_PATHS = (_PACKAGE_LIBRARY_PATH, _BUILD_LIBRARY_PATH)
_TOKENSPEED_LIBRARY_LOADED = False


def _ensure_tokenspeed_library() -> None:
    """Load TokenSpeed's native ``npu_scatter_nd_update_v2`` binding."""
    global _TOKENSPEED_LIBRARY_LOADED
    if _TOKENSPEED_LIBRARY_LOADED:
        return

    registered_vendors = register_cann_vendors()
    if not any(
        vendor_path.name == "custom_transformer"
        for vendor_path in registered_vendors
    ):
        raise FileNotFoundError(
            "TokenSpeed CANN vendor not found. Build "
            "tokenspeed-kernel-npu with CANN and torch_npu installed, or "
            "install a wheel containing "
            "tokenspeed_kernel_npu/_cann_ops_custom/vendors/custom_transformer."
        )

    library_path = next(
        (candidate for candidate in _LIBRARY_PATHS if candidate.exists()),
        None,
    )
    if library_path is None:
        raise FileNotFoundError(
            f"TokenSpeed Ascend kernel library not found: "
            f"{_PACKAGE_LIBRARY_PATH} or {_BUILD_LIBRARY_PATH}."
        )

    torch.ops.load_library(str(library_path))
    _TOKENSPEED_LIBRARY_LOADED = True


def _vllm_ascend_ops() -> Any:
    """Load and return the vllm-ascend Torch operator namespace."""
    ensure_vllm_ascend_ops()
    return torch.ops._C_ascend


def compressor(*args: Any, **kwargs: Any) -> Any:
    """Run the DSV4 compressor."""
    return _vllm_ascend_ops().compressor(*args, **kwargs)


def compressor_metadata(*args: Any, **kwargs: Any) -> Any:
    """Build DSV4 compressor metadata."""
    return _vllm_ascend_ops().compressor_metadata(*args, **kwargs)


def compressor_metadata_out(*args: Any, **kwargs: Any) -> Any:
    """Build compressor metadata into caller-owned tensors."""
    return _vllm_ascend_ops().compressor_metadata_out(*args, **kwargs)


def npu_quant_lightning_indexer(*args: Any, **kwargs: Any) -> Any:
    """Run the quantized lightning indexer."""
    return _vllm_ascend_ops().npu_vllm_quant_lightning_indexer(*args, **kwargs)


def npu_quant_lightning_indexer_metadata(*args: Any, **kwargs: Any) -> Any:
    """Build lightning-indexer metadata."""
    return _vllm_ascend_ops().npu_vllm_quant_lightning_indexer_metadata(
        *args,
        **kwargs,
    )


def npu_sparse_attn_sharedkv(*args: Any, **kwargs: Any) -> Any:
    """Run shared-KV sparse attention."""
    return _vllm_ascend_ops().npu_sparse_attn_sharedkv(*args, **kwargs)


def npu_sparse_attn_sharedkv_metadata(*args: Any, **kwargs: Any) -> Any:
    """Build shared-KV sparse-attention metadata."""
    return _vllm_ascend_ops().npu_sparse_attn_sharedkv_metadata(*args, **kwargs)


def npu_hc_post(*args: Any, **kwargs: Any) -> Any:
    """Run the DSV4 HC post operation."""
    return _vllm_ascend_ops().npu_hc_post(*args, **kwargs)


def npu_hc_pre(*args: Any, **kwargs: Any) -> Any:
    """Run the DSV4 HC pre operation."""
    return _vllm_ascend_ops().npu_hc_pre_v2(*args, **kwargs)


def inplace_partial_rotary_mul(*args: Any, **kwargs: Any) -> Any:
    """Apply partial rotary multiplication."""
    return _vllm_ascend_ops().inplace_partial_rotary_mul(*args, **kwargs)


def npu_rms_norm_dynamic_quant(*args: Any, **kwargs: Any) -> Any:
    """Run RMSNorm and dynamic quantization."""
    return _vllm_ascend_ops().npu_rms_norm_dynamic_quant(*args, **kwargs)


def npu_scatter_nd_update(*args: Any, **kwargs: Any) -> Any:
    """Run TokenSpeed's scatter update operator."""
    return npu_scatter_nd_update_v2(*args, **kwargs)


def npu_scatter_nd_update_v2(
    var: torch.Tensor,
    indices: torch.Tensor,
    update: torch.Tensor,
) -> None:
    """Scatter ``update`` into ``var`` using TokenSpeed's CANN operator."""
    _ensure_tokenspeed_library()
    torch.ops.tokenspeed_ascend.npu_scatter_nd_update_v2(var, indices, update)


def npu_moe_gating_top_k_hash(*args: Any, **kwargs: Any) -> Any:
    """Run hashed MoE routing."""
    return _vllm_ascend_ops().moe_gating_top_k_hash(*args, **kwargs)


def npu_grouped_matmul_swiglu_quant_weight_nz(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run fused grouped matmul and SwiGLU quantization."""
    return _vllm_ascend_ops().grouped_matmul_swiglu_quant_weight_nz(
        *args,
        **kwargs,
    )


def npu_dispatch_ffn_combine(*args: Any, **kwargs: Any) -> Any:
    """Run fused MoE dispatch and combine."""
    return _vllm_ascend_ops().dispatch_ffn_combine(*args, **kwargs)


__all__ = [
    "compressor",
    "compressor_metadata",
    "compressor_metadata_out",
    "inplace_partial_rotary_mul",
    "npu_dispatch_ffn_combine",
    "npu_grouped_matmul_swiglu_quant_weight_nz",
    "npu_hc_post",
    "npu_hc_pre",
    "npu_moe_gating_top_k_hash",
    "npu_quant_lightning_indexer",
    "npu_quant_lightning_indexer_metadata",
    "npu_rms_norm_dynamic_quant",
    "npu_scatter_nd_update",
    "npu_scatter_nd_update_v2",
    "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
]
