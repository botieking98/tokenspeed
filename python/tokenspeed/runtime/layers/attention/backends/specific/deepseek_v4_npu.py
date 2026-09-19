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

"""DeepSeek V4 attention backend for Ascend 910C."""

from __future__ import annotations

import torch

from tokenspeed_kernel import dsv4_npu_sparse_attn_sharedkv
from typing_extensions import override

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.registry import register_backend

NPU_DSV4_KERNEL_PAGE_SIZE = 32
NPU_DSV4_LCM_PREFIX_GRANULARITY = 4096


class DeepseekV4NpuAttentionBackend(DeepseekV4AttentionBackend):
    """Reuse upstream metadata/graph ownership, run ACLNN attention."""

    def __init__(self, config, spec) -> None:
        super().__init__(config, spec)
        self.kernel_page_size = NPU_DSV4_KERNEL_PAGE_SIZE
        self.max_num_pages = max(
            1,
            (
                self.context_len
                + NPU_DSV4_KERNEL_PAGE_SIZE
                - 1
            )
            // NPU_DSV4_KERNEL_PAGE_SIZE,
        )

    @override
    def _prepare_cache_group_tables(
        self,
        block_tables,
        *,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        device: torch.device,
        phase: str,
        output_buffers=None,
    ):
        expected_device = torch.device(device)
        if expected_device.type == "npu" and expected_device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        return super()._prepare_cache_group_tables(
            block_tables,
            bs=bs,
            actual_bs=actual_bs,
            seq_lens=seq_lens,
            device=device,
            phase=phase,
            output_buffers=output_buffers,
        )

    @override
    def _assert_active_cache_pages(
        self,
        tables,
        *,
        seq_lens,
        actual_bs,
        phase,
    ):
        """Keep scheduler-owned cache invariants off the NPU decode hot path.

        The scheduler owns active-page publication, and packed-table upload
        validates page IDs on the host before H2D. Torch NPU's asynchronous
        assertion synchronizes with the device, which would serialize decode.
        """
        if actual_bs == 0:
            return

    def _forward_shared_kv(
        self,
        *,
        q,
        token_to_kv_pool,
        layer_id: int,
        compress_ratio: int,
        window_size: int,
        softmax_scale: float,
        attn_sink,
        topk_indices,
        npu_metadata,
    ):
        if npu_metadata is None:
            raise RuntimeError("DeepSeek V4 NPU attention requires ACLNN metadata")
        if compress_ratio <= 0:
            raise ValueError(f"invalid DeepSeek V4 NPU compress ratio: {compress_ratio}")
        compressed = (
            token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id)
            if compress_ratio > 1
            else None
        )
        return dsv4_npu_sparse_attn_sharedkv(
            q,
            ori_kv=token_to_kv_pool.get_swa_kv_buffer(layer_id),
            cmp_kv=compressed,
            cmp_sparse_indices=topk_indices,
            ori_block_table=npu_metadata.swa_block_table,
            cmp_block_table=(
                npu_metadata.compressed_block_tables.get(compress_ratio)
                if compress_ratio > 1
                else None
            ),
            cu_seqlens_q=npu_metadata.query_start_loc,
            cu_seqlens_ori_kv=npu_metadata.cu_ori_seqlens,
            cu_seqlens_cmp_kv=npu_metadata.cu_cmp_seqlens.get(compress_ratio),
            seqused_q=None,
            seqused_kv=npu_metadata.seq_lens,
            sinks=attn_sink,
            metadata=npu_metadata.sas_metadata[compress_ratio],
            softmax_scale=softmax_scale,
            cmp_ratio=compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )[0]

    def forward_deepseek_v4_decode(
        self,
        *,
        q,
        positions,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink,
        topk_indices,
        metadata=None,
        npu_metadata=None,
    ):
        del positions, kind, num_local_heads, padded_heads, head_dim, metadata
        return self._forward_shared_kv(
            q=q,
            token_to_kv_pool=token_to_kv_pool,
            layer_id=layer_id,
            compress_ratio=compress_ratio,
            window_size=window_size,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            topk_indices=topk_indices,
            npu_metadata=npu_metadata,
        )

    def forward_deepseek_v4_mixed(
        self,
        *,
        q,
        positions,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink,
        topk_indices,
        metadata=None,
        npu_metadata=None,
    ):
        del positions, kind, num_local_heads, padded_heads, head_dim, metadata
        return self._forward_shared_kv(
            q=q,
            token_to_kv_pool=token_to_kv_pool,
            layer_id=layer_id,
            compress_ratio=compress_ratio,
            window_size=window_size,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            topk_indices=topk_indices,
            npu_metadata=npu_metadata,
        )

    def forward_deepseek_v4_prefill(
        self,
        *,
        q,
        positions,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink,
        topk_indices,
        metadata=None,
        npu_metadata=None,
    ):
        del positions, kind, num_local_heads, padded_heads, head_dim, metadata
        return self._forward_shared_kv(
            q=q,
            token_to_kv_pool=token_to_kv_pool,
            layer_id=layer_id,
            compress_ratio=compress_ratio,
            window_size=window_size,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            topk_indices=topk_indices,
            npu_metadata=npu_metadata,
        )


register_backend(
    "deepseek_v4_npu",
    {AttentionArch.MLA},
    DeepseekV4NpuAttentionBackend,
)
