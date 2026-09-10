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

"""Ascend attention kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.mha import (
        mha_decode_with_kvcache as _mha_decode_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import (
        mha_extend_with_kvcache as _mha_extend_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import mha_prefill as _mha_prefill

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}
    _OPTIONS = {
        "sliding_window": frozenset({False}),
        "support_sinks": frozenset({False}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    }

    @register_kernel(
        "attention",
        "mha_prefill",
        name="ascend_mha_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits=_OPTIONS,
        tags={"portability"},
    )
    def mha_prefill(**kwargs):
        return _mha_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="ascend_mha_extend_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_OPTIONS,
            "page_size": frozenset({64, 128}),
            "is_causal": frozenset({False, True}),
        },
        tags={"portability"},
    )
    def mha_extend_with_kvcache(**kwargs):
        return _mha_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="ascend_mha_decode_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_OPTIONS,
            "page_size": frozenset({64, 128}),
            "q_len": frozenset({1}),
        },
        tags={"portability"},
    )
    def mha_decode_with_kvcache(**kwargs):
        return _mha_decode_with_kvcache(**kwargs)

    from tokenspeed_kernel_npu.ops.attention import deepseek_v4 as _dsv4_ops

    def _register_aclnn(family, mode, name, implementation, tags):
        def kernel(*args, **kwargs):
            return implementation(*args, **kwargs)

        register_kernel(
            family,
            mode,
            name=name,
            solution="aclnn",
            capability=_CAPABILITY,
            signatures=frozenset({format_signature()}),
            priority=Priority.SPECIALIZED,
            tags=tags,
        )(kernel)

    for family, mode, name, implementation in (
        (
            "attention",
            "dsv4_compressor",
            "ascend_aclnn_dsv4_compressor",
            _dsv4_ops.compressor,
        ),
        (
            "attention",
            "dsv4_compressor_metadata",
            "ascend_aclnn_dsv4_compressor_metadata",
            _dsv4_ops.compressor_metadata,
        ),
        (
            "attention",
            "dsv4_inplace_partial_rotary_mul",
            "ascend_aclnn_dsv4_partial_rotary_mul",
            _dsv4_ops.inplace_partial_rotary_mul,
        ),
        (
            "attention",
            "dsv4_sparse_attn_sharedkv",
            "ascend_aclnn_dsv4_sparse_attn_sharedkv",
            _dsv4_ops.npu_sparse_attn_sharedkv,
        ),
        (
            "attention",
            "dsv4_sparse_attn_sharedkv_metadata",
            "ascend_aclnn_dsv4_sparse_attn_metadata",
            _dsv4_ops.npu_sparse_attn_sharedkv_metadata,
        ),
        (
            "attention",
            "dsv4_quant_lightning_indexer",
            "ascend_aclnn_dsv4_lightning_indexer",
            _dsv4_ops.npu_quant_lightning_indexer,
        ),
        (
            "attention",
            "dsv4_quant_lightning_indexer_metadata",
            "ascend_aclnn_dsv4_lightning_indexer_metadata",
            _dsv4_ops.npu_quant_lightning_indexer_metadata,
        ),
        (
            "moe",
            "dsv4_dispatch_ffn_combine",
            "ascend_aclnn_dsv4_dispatch_ffn_combine",
            _dsv4_ops.npu_dispatch_ffn_combine,
        ),
        (
            "moe",
            "dsv4_grouped_matmul_swiglu_quant",
            "ascend_aclnn_dsv4_grouped_matmul_swiglu_quant",
            _dsv4_ops.npu_grouped_matmul_swiglu_quant_weight_nz,
        ),
        (
            "moe",
            "dsv4_moe_gating_top_k_hash",
            "ascend_aclnn_dsv4_moe_gating_hash",
            _dsv4_ops.npu_moe_gating_top_k_hash,
        ),
        (
            "layernorm",
            "dsv4_rms_norm_dynamic_quant",
            "ascend_aclnn_dsv4_rms_norm_dynamic_quant",
            _dsv4_ops.npu_rms_norm_dynamic_quant,
        ),
        (
            "transform",
            "dsv4_scatter_nd_update",
            "ascend_aclnn_dsv4_scatter_nd_update",
            _dsv4_ops.npu_scatter_nd_update,
        ),
        (
            "transform",
            "dsv4_scatter_nd_update_v2",
            "ascend_aclnn_dsv4_scatter_nd_update_v2",
            _dsv4_ops.npu_scatter_nd_update_v2,
        ),
    ):
        _register_aclnn(
            family,
            mode,
            name,
            implementation,
            tags={"ascend", "deepseek_v4"},
        )


__all__ = [
    "dsv4_compressor",
    "dsv4_compressor_metadata",
    "dsv4_inplace_partial_rotary_mul",
    "dsv4_npu_dispatch_ffn_combine",
    "dsv4_npu_grouped_matmul_swiglu_quant_weight_nz",
    "npu_hc_post",
    "npu_hc_pre",
    "dsv4_npu_moe_gating_top_k_hash",
    "dsv4_npu_quant_lightning_indexer",
    "dsv4_npu_quant_lightning_indexer_metadata",
    "dsv4_npu_rms_norm_dynamic_quant",
    "npu_scatter_nd_update",
    "npu_scatter_nd_update_v2",
    "dsv4_npu_sparse_attn_sharedkv",
    "dsv4_npu_sparse_attn_sharedkv_metadata",
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]


def _select_aclnn(family, mode):
    return select_kernel(
        family,
        mode,
        format_signature(),
        solution="aclnn",
    )


def dsv4_compressor(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_compressor")(*args, **kwargs)


def dsv4_compressor_metadata(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_compressor_metadata")(
        *args,
        **kwargs,
    )


def dsv4_inplace_partial_rotary_mul(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_inplace_partial_rotary_mul")(
        *args,
        **kwargs,
    )


def dsv4_npu_dispatch_ffn_combine(*args, **kwargs):
    return _select_aclnn("moe", "dsv4_dispatch_ffn_combine")(*args, **kwargs)


def dsv4_npu_grouped_matmul_swiglu_quant_weight_nz(*args, **kwargs):
    return _select_aclnn("moe", "dsv4_grouped_matmul_swiglu_quant")(
        *args,
        **kwargs,
    )


def dsv4_npu_moe_gating_top_k_hash(*args, **kwargs):
    return _select_aclnn("moe", "dsv4_moe_gating_top_k_hash")(
        *args,
        **kwargs,
    )


def dsv4_npu_quant_lightning_indexer(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_quant_lightning_indexer")(
        *args,
        **kwargs,
    )


def dsv4_npu_quant_lightning_indexer_metadata(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_quant_lightning_indexer_metadata")(
        *args,
        **kwargs,
    )


def dsv4_npu_rms_norm_dynamic_quant(*args, **kwargs):
    return _select_aclnn("layernorm", "dsv4_rms_norm_dynamic_quant")(
        *args,
        **kwargs,
    )


def npu_scatter_nd_update(*args, **kwargs):
    return _select_aclnn("transform", "dsv4_scatter_nd_update")(
        *args,
        **kwargs,
    )


def npu_scatter_nd_update_v2(*args, **kwargs):
    return _select_aclnn("transform", "dsv4_scatter_nd_update_v2")(
        *args,
        **kwargs,
    )


def dsv4_npu_sparse_attn_sharedkv(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_sparse_attn_sharedkv")(
        *args,
        **kwargs,
    )


def dsv4_npu_sparse_attn_sharedkv_metadata(*args, **kwargs):
    return _select_aclnn("attention", "dsv4_sparse_attn_sharedkv_metadata")(
        *args,
        **kwargs,
    )


def npu_hc_post(*args, **kwargs):
    return _dsv4_ops.npu_hc_post(*args, **kwargs)


def npu_hc_pre(*args, **kwargs):
    return _dsv4_ops.npu_hc_pre(*args, **kwargs)
