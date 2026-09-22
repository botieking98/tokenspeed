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

"""DeepSeek-V4 Flash model for Ascend 910C.

The implementation keeps the NPU path separate from the CUDA model.  It uses
TokenSpeed's scheduler, attention metadata owner, and checkpoint loader, while
running DSV4 attention, compression, indexing, and W8A8 MoE through torch_npu
and the vendored ACLNN bindings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

try:
    import torch_npu
except ImportError:
    torch_npu = None  # type: ignore[assignment]

if torch_npu is not None:
    torch_npu.npu.config.allow_internal_format = True

from tokenspeed_kernel import (
    dsv4_compressor as aclnn_compressor,
    dsv4_compressor_metadata as aclnn_compressor_metadata,
    dsv4_inplace_partial_rotary_mul as inplace_partial_rotary_mul,
    dsv4_npu_dispatch_ffn_combine as dispatch_ffn_combine,
    npu_hc_post,
    npu_hc_pre,
    dsv4_npu_quant_lightning_indexer as npu_quant_lightning_indexer,
    dsv4_npu_quant_lightning_indexer_metadata as npu_quant_lightning_indexer_metadata,
    dsv4_npu_rms_norm_dynamic_quant as npu_rms_norm_dynamic_quant,
    npu_scatter_nd_update,
    npu_scatter_nd_update_v2,
    dsv4_npu_sparse_attn_sharedkv_metadata as npu_sparse_attn_sharedkv_metadata,
)

from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4_npu import (
    HybridDeepseekV4NpuTokenToKVPool,
)
from tokenspeed.runtime.layers.attention.page_table import (
    group_slot_mapping_from_raw as _group_slot_mapping_from_raw,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

NPU_DSV4_BLOCK_SIZE = 32
NPU_DSV4_C4_STATE_BLOCK_SIZE = 2
NPU_DSV4_C128_STATE_BLOCK_SIZE = 8
DSA_METADATA_BUFFER_SIZE = 1024
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2


def _require_torch_npu() -> None:
    if torch_npu is None:
        raise RuntimeError("deepseek_v4_npu requires torch_npu")


def _maybe_fractal_nz(tensor: torch.Tensor) -> torch.Tensor:
    _require_torch_npu()
    if tensor.device.type != "npu":
        return tensor
    return torch_npu.npu_format_cast(tensor, 29)


def _float32_scale_to_int64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64)


class NpuDynamicLinear(nn.Module):
    """Replicated or pre-sharded dynamic W8A8 linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        transpose_weight: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transpose_weight = transpose_weight
        self._weights_processed = False
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32),
            requires_grad=False,
        )
        self.weight_offset = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32),
            requires_grad=False,
        )

    def process_weights_after_loading(self, module=None) -> None:
        del module
        if self._weights_processed:
            return
        if not self.transpose_weight:
            self.weight_scale.data = self.weight_scale.data.reshape(-1)
            self.weight_offset.data = self.weight_offset.data.reshape(-1)
            self._weights_processed = True
            return
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = _maybe_fractal_nz(self.weight.data)
        self.weight_scale.data = self.weight_scale.data.reshape(-1).contiguous()
        self.weight_offset.data = self.weight_offset.data.reshape(-1).contiguous()
        self._weights_processed = True

    def process_grouped_weights_after_loading(self) -> None:
        if self._weights_processed:
            return
        transposed_weight = self.weight.data.transpose(0, 1)
        weight = transposed_weight.new_empty((1, *transposed_weight.shape))
        weight[0].copy_(transposed_weight)
        self.weight = nn.Parameter(_maybe_fractal_nz(weight), requires_grad=False)
        self.weight_scale = nn.Parameter(
            self.weight_scale.data.reshape(1, -1).to(torch.bfloat16),
            requires_grad=False,
        )
        self.weight_offset = nn.Parameter(
            self.weight_offset.data.reshape(1, -1),
            requires_grad=False,
        )
        self._weights_processed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_torch_npu()
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(
            x,
            dst_type=torch.int8,
        )
        return self.forward_quantized(quant_x, pertoken_scale, output_dtype=x.dtype)

    def forward_quantized(
        self,
        quant_x: torch.Tensor,
        pertoken_scale: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        _require_torch_npu()
        if pertoken_scale.dim() == 2:
            quant_x = quant_x.squeeze(1)
            pertoken_scale = pertoken_scale.squeeze(1)
        return torch_npu.npu_quant_matmul(
            quant_x,
            self.weight,
            self.weight_scale,
            pertoken_scale=pertoken_scale,
            output_dtype=output_dtype,
        )


class NpuBfloat16Linear(nn.Module):
    """Unquantized linear weight retained in checkpoint layout."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def _build_hadamard(index_head_dim: int, device: torch.device) -> torch.Tensor:
    size = 1 << (int(index_head_dim) - 1).bit_length()
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)],
            dim=0,
        )
    return matrix.to(device=device, dtype=torch.bfloat16)


def _hadamard_rotate(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    dim = shape[-1]
    flat = x.reshape(-1, dim)
    padded_dim = hadamard.shape[0]
    if dim != padded_dim:
        flat = F.pad(flat, (0, padded_dim - dim))
    out = F.linear(flat, hadamard) * (dim**-0.5)
    return out[..., :dim].reshape(shape)


def _zero_new_request_state_pages(
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    start_pos: torch.Tensor,
) -> None:
    rows = min(int(state_block_table.shape[0]), int(start_pos.numel()))
    if rows == 0:
        return
    new_requests = start_pos == 0
    pages = state_block_table[: start_pos.numel()][new_requests].reshape(-1)
    pages = pages[pages >= 0].unique().to(torch.long)
    if pages.numel() > 0:
        state_cache.index_fill_(0, pages, 0.0)


def _is_npu_graph_capturing(tensor: torch.Tensor) -> bool:
    return (
        tensor.device.type == "npu"
        and torch.npu.is_available()
        and torch.npu.is_current_stream_capturing()
    )


def _yarn_inv_freq(
    rotary_dim: int,
    original_max_position: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    position_freqs = base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    )
    extrapolation = 1.0 / position_freqs
    interpolation = 1.0 / (factor * position_freqs)

    def correction_dim(rotations: int) -> float:
        return (
            rotary_dim
            * math.log(original_max_position / (rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), rotary_dim - 1)
    ramp = torch.clamp(
        (torch.arange(rotary_dim // 2, dtype=torch.float32) - low)
        / max(high - low, 0.001),
        0,
        1,
    )
    mask = 1 - ramp
    return interpolation * (1 - mask) + extrapolation * mask


class NpuYarnRopeCache:
    """Shared YaRN cos/sin tables for attention and compressor kernels."""

    def __init__(
        self,
        rotary_dim: int,
        max_position: int,
        original_max_position: int,
        base: float,
        factor: float,
        beta_fast: int,
        beta_slow: int,
        device: torch.device,
    ) -> None:
        positions = torch.arange(int(max_position), dtype=torch.float32, device=device)
        inv_freq = _yarn_inv_freq(
            rotary_dim,
            original_max_position,
            base,
            factor,
            beta_fast,
            beta_slow,
        ).to(device)
        freqs = torch.outer(positions, inv_freq)
        self.cos = freqs.cos().repeat_interleave(2, dim=-1)
        self.sin = freqs.sin().repeat_interleave(2, dim=-1)

    def full(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos.unsqueeze(1).unsqueeze(1),
            self.sin.unsqueeze(1).unsqueeze(1),
        )

    def select(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index = positions.to(device=self.cos.device, dtype=torch.long)
        return self.cos[index].unsqueeze(1).unsqueeze(1), self.sin[index].unsqueeze(
            1
        ).unsqueeze(1)


@dataclass
class NpuDsaRequestMetadata:
    query_start_loc: torch.Tensor
    cu_ori_seqlens: torch.Tensor | None
    seq_lens: torch.Tensor
    token_to_req_indices: torch.Tensor
    start_pos: torch.Tensor
    swa_block_table: torch.Tensor
    swa_slot_mapping: torch.Tensor
    sas_metadata: dict[int, torch.Tensor]
    qli_metadata: torch.Tensor | None
    default_cos: dict[int, torch.Tensor]
    default_sin: dict[int, torch.Tensor]
    compress_cos: torch.Tensor
    compress_sin: torch.Tensor
    cu_cmp_seqlens: dict[int, torch.Tensor]
    compressed_block_tables: dict[int, torch.Tensor]
    state_block_tables: dict[int, torch.Tensor]
    indexer_state_block_table: torch.Tensor | None
    decode_only: bool = False


class NpuDsaMetadataBuilder:
    """Build ACLNN request metadata from TokenSpeed's DSV4 metadata."""

    def __init__(
        self,
        config: PretrainedConfig,
        pool: HybridDeepseekV4NpuTokenToKVPool,
        device: torch.device,
        num_local_heads: int,
        max_context_len: int,
        max_batch_size: int,
        max_table_pages: dict[str, int],
    ) -> None:
        self.config = config
        self.pool = pool
        self.device = device
        self.num_local_heads = num_local_heads
        self.max_context_len = max(1, int(max_context_len))
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_table_pages = max_table_pages
        self.compress_ratios = tuple(
            sorted(
                {
                    int(ratio)
                    for ratio in pool.layout.layer_ratio
                    if int(ratio) > 1
                }
            )
        )
        self.index_n_heads = int(config.index_n_heads)
        self.index_head_dim = int(config.index_head_dim)
        self.index_topk = int(config.index_topk)
        self.window_size = int(config.sliding_window)
        self.sas_buffers = {
            ratio: torch.zeros(
                DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device
            )
            for ratio in (1, 4, 128)
        }
        self.qli_buffer = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device
        )
        self.seqused_q = torch.empty(0, dtype=torch.int32, device=device)
        self.block_table_buffers: dict[str, torch.Tensor] = {}

    def _block_table(
        self, metadata: Any, group_id: str
    ) -> torch.Tensor:
        table = metadata.cache.block_tables.get(group_id)
        if table is None:
            raise RuntimeError(f"Missing NPU DSV4 cache block table: {group_id}")
        max_pages = self.max_table_pages.get(group_id)
        if max_pages is None:
            raise RuntimeError(f"Missing NPU DSV4 graph table width: {group_id}")
        max_pages = max(1, int(max_pages))
        rows = max(self.max_batch_size, int(table.shape[0]))
        buffer = self.block_table_buffers.get(group_id)
        if (
            buffer is None
            or buffer.shape[0] < rows
            or buffer.shape[1] < max_pages + 1
        ):
            buffer = torch.zeros(
                (rows, max_pages + 1),
                dtype=torch.int32,
                device=self.device,
            )
            self.block_table_buffers[group_id] = buffer
        if table.shape[1] > max_pages:
            raise RuntimeError(
                f"NPU DSV4 block table for {group_id} exceeds max context"
            )
        buffer[: table.shape[0]].zero_()
        buffer[: table.shape[0], : table.shape[1]].copy_(
            table[: table.shape[0], : table.shape[1]].to(
                device=buffer.device,
                dtype=buffer.dtype,
            )
        )
        return buffer[: table.shape[0], :max_pages]

    def build(
        self,
        metadata: Any,
        positions: torch.Tensor,
        rope: NpuYarnRopeCache,
        compress_rope: NpuYarnRopeCache,
    ) -> NpuDsaRequestMetadata:
        num_tokens = int(positions.numel())
        num_reqs = int(metadata.seq_lens.numel())
        query_start_loc = metadata.query_start_loc[: num_reqs + 1].contiguous()
        seq_lens = metadata.seq_lens[:num_reqs].contiguous()
        token_to_req = metadata.token_to_req_indices[:num_tokens].contiguous()
        start_pos = seq_lens - metadata.query_lens[:num_reqs]
        default_cos, default_sin = rope.select(positions)
        compressed_cos, compressed_sin = compress_rope.select(positions)
        forward_mode = metadata.forward_mode
        decode_only = forward_mode is not None and forward_mode.is_decode_or_idle()

        query_lens_cpu = metadata.query_lens_cpu
        seq_lens_cpu = metadata.seq_lens_cpu
        graph_capturing = _is_npu_graph_capturing(positions)
        if graph_capturing:
            max_query_len = 1
            max_seq_len = self.max_context_len
        elif not num_reqs:
            max_query_len = 1
            max_seq_len = 1
        else:
            if query_lens_cpu is not None and query_lens_cpu.numel() >= num_reqs:
                max_query_len = max(1, int(query_lens_cpu[:num_reqs].max().item()))
            else:
                max_query_len = max(1, int(metadata.query_lens[:num_reqs].max().item()))
            if seq_lens_cpu is not None and seq_lens_cpu.numel() >= num_reqs:
                max_seq_len = max(1, int(seq_lens_cpu[:num_reqs].max().item()))
            else:
                max_seq_len = max(1, int(seq_lens.max().item()))
        has_prefill = bool(not decode_only and metadata.num_prefill_reqs > 0)
        cu_ori = query_start_loc if has_prefill else None
        cu_cmp: dict[int, torch.Tensor] = {}
        for ratio in self.compress_ratios:
            if has_prefill:
                cu_cmp[ratio] = None
            else:
                cu_cmp[ratio] = torch.empty(0, dtype=torch.int32, device=self.device)

        sas_metadata: dict[int, torch.Tensor] = {}
        for ratio in (1, *self.compress_ratios):
            built = npu_sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_local_heads,
                num_heads_kv=1,
                head_dim=int(self.config.head_dim),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_ori,
                cu_seqlens_cmp_kv=cu_cmp.get(ratio),
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_kv=max_seq_len,
                ori_topk=0,
                cmp_topk=self.index_topk if ratio == 4 else 0,
                cmp_ratio=max(ratio, 1),
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=ratio > 1,
                device=str(self.device),
            )
            self.sas_buffers[ratio].copy_(built)
            sas_metadata[ratio] = self.sas_buffers[ratio]

        qli_metadata = None
        if 4 in self.compress_ratios:
            built = npu_quant_lightning_indexer_metadata(
                num_heads_q=self.index_n_heads,
                num_heads_k=1,
                head_dim=self.index_head_dim,
                query_quant_mode=0,
                key_quant_mode=0,
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=seq_lens.clone(),
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                cmp_ratio=4,
                device=str(self.device),
            )
            self.qli_buffer.copy_(built)
            qli_metadata = self.qli_buffer

        swa_table = self._block_table(
            metadata,
            V4_SWA_KV_GROUP_ID,
        )
        swa_slots = _group_slot_mapping_from_raw(
            positions,
            token_to_req,
            swa_table,
            NPU_DSV4_BLOCK_SIZE,
        )
        if metadata.is_valid_token is not None:
            valid = metadata.is_valid_token[:num_tokens]
            swa_slots = torch.where(valid, swa_slots, torch.full_like(swa_slots, -1))
        swa_slots = torch.stack(
            [
                swa_slots // NPU_DSV4_BLOCK_SIZE,
                swa_slots % NPU_DSV4_BLOCK_SIZE,
            ],
            dim=-1,
        )

        compressed_tables = {
            ratio: self._block_table(
                metadata,
                v4_compressed_kv_group_id(ratio),
            )
            for ratio in self.compress_ratios
        }
        state_tables = {
            ratio: self._block_table(
                metadata,
                v4_compressor_state_group_id(ratio),
            )
            for ratio in self.compress_ratios
        }
        indexer_state_table = None
        if 4 in self.compress_ratios:
            indexer_state_table = self._block_table(
                metadata,
                V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
            )

        layer_cos: dict[int, torch.Tensor] = {}
        layer_sin: dict[int, torch.Tensor] = {}
        for layer_id in range(int(self.config.num_hidden_layers)):
            compress_ratio = max(1, int(self.config.compress_ratios[layer_id]))
            if compress_ratio > 1:
                layer_cos[layer_id] = compressed_cos
                layer_sin[layer_id] = compressed_sin
            else:
                layer_cos[layer_id] = default_cos
                layer_sin[layer_id] = default_sin

        return NpuDsaRequestMetadata(
            query_start_loc=query_start_loc,
            cu_ori_seqlens=cu_ori,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req,
            start_pos=start_pos,
            swa_block_table=swa_table,
            swa_slot_mapping=swa_slots,
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            default_cos=layer_cos,
            default_sin=layer_sin,
            compress_cos=compressed_cos,
            compress_sin=compressed_sin,
            cu_cmp_seqlens=cu_cmp,
            compressed_block_tables=compressed_tables,
            state_block_tables=state_tables,
            indexer_state_block_table=indexer_state_table,
            decode_only=decode_only,
        )



class NpuCompressor(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        compress_ratio: int,
        head_dim: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self.norm_eps = float(config.rms_norm_eps)
        self.coff = 2 if compress_ratio == 4 else 1
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.wkv = NpuBfloat16Linear(config.hidden_size, self.coff * head_dim)
        self.wgate = NpuBfloat16Linear(config.hidden_size, self.coff * head_dim)
        self.norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: torch.Tensor,
        request_metadata: NpuDsaRequestMetadata,
        full_cos: torch.Tensor,
        full_sin: torch.Tensor,
        compressed_block_table: torch.Tensor,
        state_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        num_reqs = request_metadata.seq_lens.numel()
        num_compressed = min(num_tokens, num_tokens // self.compress_ratio + num_reqs)
        flattened_full_cos = full_cos.view(full_cos.shape[0], full_cos.shape[-1])
        flattened_full_sin = full_sin.view(full_sin.shape[0], full_sin.shape[-1])
        compress_cos, compress_sin, slot_mapping = aclnn_compressor_metadata(
            flattened_full_cos,
            flattened_full_sin,
            request_metadata.query_start_loc,
            request_metadata.start_pos,
            compressed_block_table,
            kv_block_size=NPU_DSV4_BLOCK_SIZE,
            slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            compress_ratio=self.compress_ratio,
            num_compressed_tokens=num_compressed,
            num_reqs_actual=num_reqs,
        )
        compressed_kv = aclnn_compressor(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            state_cache.squeeze(-2),
            self.ape,
            self.norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table,
            request_metadata.query_start_loc,
            None,
            request_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=self.coff,
            norm_eps=self.norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        return compressed_kv, slot_mapping


class NpuIndexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        hadamard: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self.index_topk = int(config.index_topk)
        self.q_lora_rank = int(config.q_lora_rank)
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = NpuDynamicLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = NpuCompressor(
            config,
            4,
            self.head_dim,
            add_prefix("compressor", prefix),
        )
        self.hadamard = hadamard

    def forward(
        self,
        hidden_states: torch.Tensor,
        quant_qr: torch.Tensor,
        qr_scale: torch.Tensor,
        request_metadata: NpuDsaRequestMetadata,
        pool: HybridDeepseekV4NpuTokenToKVPool,
        layer_id: int,
        full_compress_cos: torch.Tensor,
        full_compress_sin: torch.Tensor,
    ) -> torch.Tensor:
        _require_torch_npu()
        q = self._project_query(
            hidden_states, quant_qr, qr_scale, request_metadata, layer_id
        )
        self._prepare_cache(
            hidden_states,
            request_metadata,
            pool,
            layer_id,
            full_compress_cos,
            full_compress_sin,
        )
        return self._select_topk(hidden_states, q, request_metadata, pool, layer_id)

    def _project_query(
        self,
        hidden_states: torch.Tensor,
        quant_qr: torch.Tensor,
        qr_scale: torch.Tensor,
        request_metadata: NpuDsaRequestMetadata,
        layer_id: int,
    ) -> torch.Tensor:
        q = torch_npu.npu_quant_matmul(
            quant_qr,
            self.wq_b.weight,
            self.wq_b.weight_scale,
            pertoken_scale=qr_scale,
            output_dtype=hidden_states.dtype,
        ).view(-1, self.n_heads, self.head_dim)
        inplace_partial_rotary_mul(
            q.unsqueeze(1),
            request_metadata.default_cos[layer_id],
            request_metadata.default_sin[layer_id],
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )
        return _hadamard_rotate(q, self.hadamard)

    def _prepare_cache(
        self,
        hidden_states: torch.Tensor,
        request_metadata: NpuDsaRequestMetadata,
        pool: HybridDeepseekV4NpuTokenToKVPool,
        layer_id: int,
        full_compress_cos: torch.Tensor,
        full_compress_sin: torch.Tensor,
    ) -> None:
        indexer_state_cache = pool.get_indexer_state_buffer(layer_id)
        if not request_metadata.decode_only:
            _zero_new_request_state_pages(
                indexer_state_cache,
                request_metadata.indexer_state_block_table,
                request_metadata.start_pos,
            )
        kv, slot_mapping = self.compressor(
            hidden_states,
            indexer_state_cache,
            request_metadata,
            full_compress_cos,
            full_compress_sin,
            request_metadata.compressed_block_tables[4],
            request_metadata.indexer_state_block_table,
        )
        if kv.numel() > 0:
            kv = _hadamard_rotate(kv, self.hadamard)
            quant_kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=torch.int8)
            kv_scale = kv_scale.unsqueeze(-1).to(torch.float16)
            if kv_scale.ndim < 4:
                kv_scale = kv_scale.unsqueeze(-1)
            npu_scatter_nd_update_v2(
                pool.get_indexer_kv_buffer_2d(layer_id), slot_mapping, quant_kv
            )
            npu_scatter_nd_update_v2(
                pool.get_indexer_scale_buffer(layer_id), slot_mapping, kv_scale
            )

    def _select_topk(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        request_metadata: NpuDsaRequestMetadata,
        pool: HybridDeepseekV4NpuTokenToKVPool,
        layer_id: int,
    ) -> torch.Tensor:
        main_stream = torch.npu.current_stream()
        aux_stream = _get_dsa_overlap_stream()
        weights_proj_start = main_stream.record_event()
        with torch.npu.stream(aux_stream):
            aux_stream.wait_event(weights_proj_start)
            weights = self.weights_proj(hidden_states)[0]
            weights_ready = aux_stream.record_event()
        quant_q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=torch.int8)
        main_stream.wait_event(weights_ready)
        weights = weights * (self.softmax_scale * self.n_heads**-0.5)
        topk_indices, _ = npu_quant_lightning_indexer(
            quant_q,
            pool.get_indexer_kv_buffer_2d(layer_id),
            weights.to(torch.float16),
            q_scale.to(torch.float16),
            pool.get_indexer_scale_buffer(layer_id).squeeze(-2).to(torch.float16),
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=request_metadata.query_start_loc[1:],
            actual_seq_lengths_key=request_metadata.seq_lens,
            block_table=request_metadata.compressed_block_tables[4],
            metadata=request_metadata.qli_metadata,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            cmp_ratio=4,
            return_value=False,
        )
        return topk_indices


class NpuAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int,
        prefix: str,
        hadamard: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.compress_ratio = max(1, int(config.compress_ratios[layer_id]))
        self.num_heads = int(config.num_attention_heads)
        self.num_local_heads = self.num_heads // mapping.attn.tp_size
        self.head_dim = int(config.head_dim)
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.attention_kind = (
            "swa"
            if self.compress_ratio <= 1
            else "csa" if self.compress_ratio == 4 else "hca"
        )
        self.q_lora_rank = int(config.q_lora_rank)
        self.o_lora_rank = int(config.o_lora_rank)
        self.o_groups = int(config.o_groups)
        tp_size = mapping.attn.tp_size
        heads_per_group = self.num_heads // self.o_groups
        if tp_size <= self.o_groups:
            self.num_local_groups = self.o_groups // tp_size
            wo_a_input_dim = heads_per_group * self.head_dim
        else:
            self.num_local_groups = 1
            wo_a_input_dim = self.num_local_heads * self.head_dim
        self.window_size = int(config.sliding_window)
        self.scale = self.head_dim**-0.5
        self.eps = float(config.rms_norm_eps)
        self.wq_a = NpuDynamicLinear(config.hidden_size, self.q_lora_rank)
        self.wkv = NpuDynamicLinear(config.hidden_size, self.head_dim)
        self.wq_b = NpuDynamicLinear(
            self.q_lora_rank,
            self.num_local_heads * self.head_dim,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_head_norm_weight = nn.Parameter(
            torch.ones(self.head_dim), requires_grad=False
        )
        self.attn_sink = nn.Parameter(
            torch.full((self.num_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        self.wo_a = nn.Parameter(
            torch.empty(
                self.num_local_groups * self.o_lora_rank,
                wo_a_input_dim,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.wo_b = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.num_local_groups * self.o_lora_rank,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.compressor = (
            NpuCompressor(
                config,
                self.compress_ratio,
                self.head_dim,
                add_prefix("compressor", prefix),
            )
            if self.compress_ratio > 1
            else None
        )
        self.indexer = (
            NpuIndexer(
                config,
                add_prefix("indexer", prefix),
                hadamard,
            )
            if self.compress_ratio == 4
            else None
        )

    def _project_output(
        self,
        attn_output: torch.Tensor,
        metadata: NpuDsaRequestMetadata,
    ) -> torch.Tensor:
        inplace_partial_rotary_mul(
            attn_output.unsqueeze(1),
            metadata.default_cos[self.layer_id],
            -metadata.default_sin[self.layer_id],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        num_tokens = attn_output.shape[0]
        group_hidden_dim = (
            attn_output.shape[1] * attn_output.shape[2] // self.num_local_groups
        )
        grouped_input = attn_output.reshape(
            num_tokens, self.num_local_groups, group_hidden_dim
        )
        weight = self.wo_a
        _require_torch_npu()
        z = torch.bmm(grouped_input.transpose(0, 1), weight).transpose(0, 1)
        z = z.reshape(num_tokens, -1)
        return F.linear(z, self.wo_b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        metadata: NpuDsaRequestMetadata,
        rope: NpuYarnRopeCache,
        compress_rope: NpuYarnRopeCache,
        pool: HybridDeepseekV4NpuTokenToKVPool,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        _require_torch_npu()
        main_stream = torch.npu.current_stream()
        aux_stream = _get_dsa_overlap_stream()
        quant_hidden, hidden_scale = torch_npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.int8
        )
        if hidden_scale.dim() == 2:
            quant_hidden = quant_hidden.squeeze(1)
            hidden_scale = hidden_scale.squeeze(1)
        projection_ready = main_stream.record_event()
        with torch.npu.stream(aux_stream):
            aux_stream.wait_event(projection_ready)
            kv = self.wkv.forward_quantized(
                quant_hidden,
                hidden_scale,
                output_dtype=hidden_states.dtype,
            )
            kv = self.kv_norm(kv).view(-1, 1, self.head_dim)
            inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                metadata.default_cos[self.layer_id],
                metadata.default_sin[self.layer_id],
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            npu_scatter_nd_update_v2(
                pool.get_swa_kv_buffer(self.layer_id),
                metadata.swa_slot_mapping,
                kv,
            )
            if self.compress_ratio > 1:
                assert self.compressor is not None
                state_cache = pool.get_compressor_state_buffer(self.layer_id)
                state_block_table = metadata.state_block_tables[
                    self.compress_ratio
                ]
                if not metadata.decode_only:
                    _zero_new_request_state_pages(
                        state_cache,
                        state_block_table,
                        metadata.start_pos,
                    )
                compressed_kv, compressed_slots = self.compressor(
                    hidden_states,
                    state_cache,
                    metadata,
                    *compress_rope.full(),
                    metadata.compressed_block_tables[self.compress_ratio],
                    state_block_table,
                )
                if compressed_kv.numel() > 0:
                    npu_scatter_nd_update_v2(
                        pool.get_compressed_kv_buffer_2d(self.layer_id),
                        compressed_slots,
                        compressed_kv,
                    )
            if self.indexer is not None:
                self.indexer._prepare_cache(
                    hidden_states,
                    metadata,
                    pool,
                    self.layer_id,
                    *compress_rope.full(),
                )
                indexer_cache_ready = aux_stream.record_event()
        q_a = self.wq_a.forward_quantized(
            quant_hidden,
            hidden_scale,
            output_dtype=hidden_states.dtype,
        )
        quant_qr, qr_scale = npu_rms_norm_dynamic_quant(
            q_a, self.q_norm.weight, epsilon=self.eps
        )
        q = torch_npu.npu_quant_matmul(
            quant_qr,
            self.wq_b.weight,
            self.wq_b.weight_scale,
            pertoken_scale=qr_scale,
            output_dtype=hidden_states.dtype,
        ).view(-1, self.num_local_heads, self.head_dim)
        q = torch_npu.npu_rms_norm(q, self.q_head_norm_weight, epsilon=self.eps)[0]
        inplace_partial_rotary_mul(
            q.unsqueeze(1),
            metadata.default_cos[self.layer_id],
            metadata.default_sin[self.layer_id],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        topk_indices = None
        if self.indexer is not None:
            indexer_q = self.indexer._project_query(
                hidden_states, quant_qr, qr_scale, metadata, self.layer_id
            )
            if indexer_cache_ready is not None:
                main_stream.wait_event(indexer_cache_ready)
            topk_indices = self.indexer._select_topk(
                hidden_states,
                indexer_q,
                metadata,
                pool,
                self.layer_id,
            )
        else:
            main_stream.wait_stream(aux_stream)
        forward_mode = ctx.forward_mode
        if forward_mode is None:
            raise RuntimeError("DeepSeek V4 NPU attention requires forward mode")
        backend_kwargs = {
            "q": q,
            "positions": positions,
            "token_to_kv_pool": pool,
            "layer_id": self.layer_id,
            "kind": self.attention_kind,
            "compress_ratio": self.compress_ratio,
            "num_local_heads": self.num_local_heads,
            "padded_heads": self.num_local_heads,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "softmax_scale": self.scale,
            "attn_sink": self.attn_sink,
            "topk_indices": topk_indices,
            "npu_metadata": metadata,
        }
        if forward_mode.is_mixed():
            attn_output = ctx.attn_backend.forward_deepseek_v4_mixed(
                **backend_kwargs,
            )
        elif forward_mode.is_decode():
            attn_output = ctx.attn_backend.forward_deepseek_v4_decode(
                **backend_kwargs,
            )
        elif forward_mode.is_extend_or_mixed():
            attn_output = ctx.attn_backend.forward_deepseek_v4_prefill(
                **backend_kwargs,
            )
        else:
            raise RuntimeError(
                f"Unsupported DeepSeek V4 NPU forward mode: {forward_mode}"
            )
        return self._project_output(attn_output, metadata)


class NpuW8A8Experts(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.moe_intermediate_size)
        self.num_local_experts = num_local_experts
        swiglu_limit = getattr(config, "swiglu_limit", None)
        self.swiglu_limit = (
            float(swiglu_limit)
            if swiglu_limit is not None and swiglu_limit > 0
            else None
        )
        self._weights_processed = False
        self.fused_w13_weight_scale = None
        self.fused_w2_weight_scale = None
        self.fused_scale_bias = None
        self.fused_expert_token_nums = None
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.empty(
                num_local_experts, 2 * intermediate_size, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.w13_weight_offset = nn.Parameter(
            torch.empty(
                num_local_experts, 2 * intermediate_size, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.empty(num_local_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        self.w2_weight_offset = nn.Parameter(
            torch.empty(num_local_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )

    def process_weights_after_loading(self, module=None) -> None:
        del module
        if self._weights_processed:
            return
        self.w13_weight.data = self.w13_weight.data.transpose(1, 2).contiguous()
        self.w2_weight.data = self.w2_weight.data.transpose(1, 2).contiguous()
        self.w13_weight.data = _maybe_fractal_nz(self.w13_weight.data)
        self.w2_weight.data = _maybe_fractal_nz(self.w2_weight.data)
        self.w13_weight_scale.data = self.w13_weight_scale.data.view(
            self.num_local_experts, -1
        )
        self.fused_w13_weight_scale = _float32_scale_to_int64(
            self.w13_weight_scale.data
        )
        self.w13_weight_scale.data = self.w13_weight_scale.data.to(torch.bfloat16)
        self.w13_weight_offset.data = self.w13_weight_offset.data.view(
            self.num_local_experts, -1
        )
        self.w2_weight_scale.data = self.w2_weight_scale.data.view(
            self.num_local_experts, -1
        )
        self.fused_w2_weight_scale = _float32_scale_to_int64(
            self.w2_weight_scale.data
        )
        self.w2_weight_scale.data = self.w2_weight_scale.data.to(torch.bfloat16)
        self.w2_weight_offset.data = self.w2_weight_offset.data.view(
            self.num_local_experts, -1
        )
        self.fused_scale_bias = torch.empty(
            (0,),
            dtype=torch.float32,
            device=self.w13_weight.device,
        )
        self.fused_expert_token_nums = torch.zeros(
            (self.num_local_experts,),
            dtype=torch.int32,
            device=self.w13_weight.device,
        )
        self._weights_processed = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        pertoken_scale: torch.Tensor | None = None,
        prequantized: bool = False,
    ) -> torch.Tensor:
        _require_torch_npu()
        if self.swiglu_limit is None:
            raise ValueError("DeepSeek-V4 NPU W8A8 experts require swiglu_limit")
        if prequantized:
            if pertoken_scale is None:
                raise ValueError("Prequantized DeepSeek-V4 experts require a scale")
            quant_hidden = hidden_states
            hidden_scale = pertoken_scale.reshape(-1).to(torch.float32)
        else:
            quant_hidden, hidden_scale = torch_npu.npu_dynamic_quant(
                hidden_states, dst_type=torch.int8
            )
            if hidden_scale.dim() == 2:
                quant_hidden = quant_hidden.squeeze(1)
                hidden_scale = hidden_scale.squeeze(1)
        group_list = group_list.to(torch.int64)
        gate_up = torch_npu.npu_grouped_matmul(
            x=[quant_hidden],
            weight=[self.w13_weight],
            scale=[self.w13_weight_scale],
            bias=None,
            per_token_scale=[hidden_scale],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
        activated = torch_npu.npu_clipped_swiglu(
            gate_up,
            dim=-1,
            alpha=1.0,
            limit=self.swiglu_limit,
            bias=0.0,
            interleaved=False,
        )
        activated, activated_scale = torch_npu.npu_dynamic_quant(
            activated,
            dst_type=torch.int8,
        )
        if activated_scale.dim() == 2:
            activated = activated.squeeze(1)
            activated_scale = activated_scale.squeeze(1)

        expert_output = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_weight],
            scale=[self.w2_weight_scale],
            bias=None,
            per_token_scale=[activated_scale],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
        return expert_output


_SHARED_EXPERT_STREAM: Any = None
_DSA_OVERLAP_STREAM: Any = None


def _get_dsa_overlap_stream() -> Any:
    global _DSA_OVERLAP_STREAM
    if _DSA_OVERLAP_STREAM is None:
        _DSA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSA_OVERLAP_STREAM


def _get_shared_expert_stream() -> Any:
    global _SHARED_EXPERT_STREAM
    if _SHARED_EXPERT_STREAM is None:
        _SHARED_EXPERT_STREAM = torch_npu.npu.Stream()
    return _SHARED_EXPERT_STREAM


def _combine_shared_and_routed(
    shared: torch.Tensor, routed: torch.Tensor, scaling: float
) -> torch.Tensor:
    return torch.add(shared, routed, alpha=scaling)


class NpuMoE(nn.Module):
    MC2_MAX_TOKENS_PER_RANK = 224
    _shared_group_lists: dict[tuple[torch.device, int], torch.Tensor] = {}
    _active_masks: dict[torch.device, torch.Tensor] = {}
    _all_active_masks: dict[torch.device, torch.Tensor] = {}

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.mapping = mapping
        self.config = config
        self.is_hash_moe = layer_id < int(config.num_hash_layers)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.num_local_experts = self.num_experts // mapping.moe.ep_size
        self.local_expert_start = mapping.moe.ep_rank * self.num_local_experts
        self.mega_moe_max_num_tokens = int(
            global_server_args_dict.get(
                "deepseek_v4_mega_moe_max_num_tokens", 0
            )
            or 0
        )
        self.decode_moe_max_num_tokens = int(
            global_server_args_dict["max_num_seqs"]
        )
        self.prefill_moe_max_num_tokens = max(
            int(global_server_args_dict["chunked_prefill_size"]),
            int(global_server_args_dict.get("prefill_graph_max_tokens", 0) or 0),
        )
        self.renormalize = bool(config.norm_topk_prob)
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        swiglu_limit = getattr(config, "swiglu_limit", None)
        self.swiglu_limit = float(swiglu_limit) if swiglu_limit is not None else None
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        if self.is_hash_moe:
            self.e_score_correction_bias = None
            self.register_parameter(
                "tid2eid",
                nn.Parameter(
                    torch.empty(
                        config.vocab_size,
                        self.top_k,
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("tid2eid", None)
        self.experts = NpuW8A8Experts(config, self.num_local_experts)
        intermediate_size = int(config.moe_intermediate_size)
        self.shared_gate_up = NpuDynamicLinear(
            config.hidden_size, 2 * intermediate_size
        )
        self.shared_down = NpuDynamicLinear(intermediate_size, config.hidden_size)
        self._shared_weights_processed = False
        _get_shared_expert_stream()

    def process_weights_after_loading(self, module=None) -> None:
        del module
        if self._shared_weights_processed:
            return
        self.shared_gate_up.process_grouped_weights_after_loading()
        self._ensure_active_mask_buffers(
            self.shared_gate_up.weight.device,
            max(
                self.MC2_MAX_TOKENS_PER_RANK,
                self.decode_moe_max_num_tokens,
                self.prefill_moe_max_num_tokens,
                self.mega_moe_max_num_tokens,
            ),
        )
        self._shared_weights_processed = True

    @classmethod
    def _ensure_active_mask_buffers(
        cls, device: torch.device, capacity: int
    ) -> None:
        capacity = max(capacity, cls.MC2_MAX_TOKENS_PER_RANK)
        active_mask = cls._active_masks.get(device)
        if active_mask is None or active_mask.numel() < capacity:
            cls._active_masks[device] = torch.zeros(
                capacity, dtype=torch.bool, device=device
            )
        all_active_mask = cls._all_active_masks.get(device)
        if all_active_mask is None or all_active_mask.numel() < capacity:
            cls._all_active_masks[device] = torch.ones(
                capacity, dtype=torch.bool, device=device
            )

    def _route(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states.float())[0]
        if self.is_hash_moe:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash MoE routing requires input_ids")
            input_ids = input_ids.reshape(-1).to(torch.int64)
        else:
            input_ids = None
        scores = torch.sqrt(F.softplus(router_logits))
        if self.is_hash_moe:
            topk_ids = self.tid2eid[input_ids].to(torch.int32)
        else:
            scores_for_choice = scores
            if self.e_score_correction_bias is not None:
                scores_for_choice = scores_for_choice + self.e_score_correction_bias
            topk_ids = torch.topk(
                scores_for_choice,
                k=self.top_k,
                dim=-1,
                sorted=True,
            ).indices
        topk_weights = scores.gather(1, topk_ids.long())
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(topk_weights.dtype).tiny)
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def _hccl_group_name(self) -> str:
        group = self._hccl_group()
        local_rank = dist.get_rank(group=group)
        backend = group._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(local_rank)

    def _hccl_group(self) -> Any:
        return pg_manager.get_process_group("hccl", self.mapping.moe.tp_ep_group)

    def _fused_moe(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        topk_weights, topk_ids = self._route(hidden_states, input_ids)
        routed = torch.empty_like(hidden_states)
        dispatch_ffn_combine(
            x=hidden_states,
            weight1=[self.experts.w13_weight],
            weight2=[self.experts.w2_weight],
            expert_idx=topk_ids,
            scale1=[self.experts.fused_w13_weight_scale],
            scale2=[self.experts.fused_w2_weight_scale],
            bias1=[self.experts.fused_scale_bias],
            bias2=[self.experts.fused_scale_bias],
            probs=topk_weights.to(torch.float32),
            group=self._hccl_group_name(),
            max_output_size=self._max_output_size(hidden_states.shape[0]),
            out=routed,
            expert_token_nums=self.experts.fused_expert_token_nums,
            x_active_mask=active_mask,
            swiglu_limit=self.experts.swiglu_limit,
        )
        return routed

    def _padded_moe_input(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        padded_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        pad_size = padded_num_tokens - hidden_states.shape[0]
        if pad_size < 0:
            raise ValueError(
                "DeepSeek-V4 NPU MoE padding is smaller than the local token count"
            )
        if pad_size:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, pad_size))
        active_mask = self._active_mask(
            hidden_states.device, num_tokens, padded_num_tokens
        )
        return hidden_states, input_ids, active_mask

    def _active_mask(
        self, device: torch.device, num_tokens: int, padded_num_tokens: int
    ) -> torch.Tensor:
        if num_tokens == padded_num_tokens:
            all_active_mask = self._all_active_masks.get(device)
            if all_active_mask is None or all_active_mask.numel() < padded_num_tokens:
                all_active_mask = torch.ones(
                    max(padded_num_tokens, self.MC2_MAX_TOKENS_PER_RANK),
                    dtype=torch.bool,
                    device=device,
                )
                self._all_active_masks[device] = all_active_mask
            return all_active_mask[:padded_num_tokens]

        active_mask = self._active_masks.get(device)
        if active_mask is None or active_mask.numel() < padded_num_tokens:
            active_mask = torch.zeros(
                max(padded_num_tokens, self.MC2_MAX_TOKENS_PER_RANK),
                dtype=torch.bool,
                device=device,
            )
            self._active_masks[device] = active_mask
        active_mask = active_mask[:padded_num_tokens]
        if self.layer_id == 0:
            active_mask[:num_tokens] = True
            active_mask[num_tokens:] = False
        return active_mask

    def _max_output_size(self, padded_num_tokens: int) -> int:
        if self.mega_moe_max_num_tokens > 0:
            max_input_tokens = max(
                padded_num_tokens,
                self.mega_moe_max_num_tokens,
            )
        elif padded_num_tokens > self.MC2_MAX_TOKENS_PER_RANK:
            max_input_tokens = max(
                padded_num_tokens,
                self.prefill_moe_max_num_tokens,
            )
        else:
            max_input_tokens = max(
                padded_num_tokens,
                self.decode_moe_max_num_tokens,
            )
        return max_input_tokens * self.top_k

    def _shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit is None:
            raise ValueError("DeepSeek-V4 NPU shared experts require swiglu_limit")
        if (
            self.shared_gate_up.weight.dim() != 3
            or self.shared_gate_up.weight_scale.dim() != 2
        ):
            raise RuntimeError(
                "DeepSeek-V4 NPU shared gate/up weight must be grouped NZ: "
                f"weight={tuple(self.shared_gate_up.weight.shape)}, "
                f"scale={tuple(self.shared_gate_up.weight_scale.shape)}"
            )
        quant_hidden, hidden_scale = torch_npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.int8
        )
        if hidden_scale.dim() == 2:
            quant_hidden = quant_hidden.squeeze(1)
            hidden_scale = hidden_scale.squeeze(1)
        num_tokens = quant_hidden.shape[0]
        group_list_key = (quant_hidden.device, num_tokens)
        group_list = NpuMoE._shared_group_lists.get(group_list_key)
        if group_list is None:
            group_list = torch.full(
                (1,), num_tokens, dtype=torch.int64, device=quant_hidden.device
            )
            NpuMoE._shared_group_lists[group_list_key] = group_list
        gate_up = torch_npu.npu_grouped_matmul(
            x=[quant_hidden],
            weight=[self.shared_gate_up.weight],
            scale=[self.shared_gate_up.weight_scale.to(torch.bfloat16)],
            bias=None,
            per_token_scale=[hidden_scale],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
        activated = torch_npu.npu_clipped_swiglu(
            gate_up,
            dim=-1,
            alpha=1.0,
            limit=self.swiglu_limit,
            bias=0.0,
            interleaved=False,
        )
        activated, activated_scale = torch_npu.npu_dynamic_quant(
            activated,
            dst_type=torch.int8,
        )
        if activated_scale.dim() == 2:
            activated = activated.squeeze(1)
            activated_scale = activated_scale.squeeze(1)
        return self.shared_down.forward_quantized(
            activated,
            activated_scale,
            output_dtype=hidden_states.dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        padded_num_tokens: int,
    ) -> torch.Tensor:
        _require_torch_npu()
        padded_num_tokens = max(1, int(padded_num_tokens))

        shared = None
        shared_ready = None
        if hidden_states.shape[0]:
            hidden_ready = torch.npu.Event()
            hidden_ready.record()
            shared_stream = _get_shared_expert_stream()
            with torch.npu.stream(shared_stream):
                shared_stream.wait_event(hidden_ready)
                shared = self._shared_expert(hidden_states)
                shared_ready = torch.npu.Event()
                shared_ready.record()

        padded_hidden, padded_input_ids, active_mask = self._padded_moe_input(
            hidden_states, input_ids, padded_num_tokens
        )
        routed = self._fused_moe(padded_hidden, padded_input_ids, active_mask)
        if hidden_states.shape[0] < padded_num_tokens:
            routed = routed[: hidden_states.shape[0]]
        if hidden_states.shape[0] == 0:
            return routed

        torch.npu.current_stream().wait_event(shared_ready)
        return _combine_shared_and_routed(
            shared, routed, self.routed_scaling_factor
        )


class NpuDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int,
        prefix: str,
        hadamard: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = int(config.hc_mult)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.hc_eps = float(config.hc_eps)
        self.norm_eps = float(config.rms_norm_eps)
        hc_dim = self.hc_mult * int(config.hidden_size)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.attn = NpuAttention(
            config, mapping, layer_id, add_prefix("self_attn", prefix), hadamard
        )
        self.ffn = NpuMoE(config, mapping, layer_id, add_prefix("mlp", prefix))
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def _pre(
        self,
        hidden_states: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ):
        return npu_hc_pre(
            hidden_states,
            fn,
            scale,
            base,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    @staticmethod
    def _post(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return npu_hc_post(
            hidden_states.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        metadata: NpuDsaRequestMetadata,
        rope: NpuYarnRopeCache,
        compress_rope: NpuYarnRopeCache,
        pool: HybridDeepseekV4NpuTokenToKVPool,
        input_ids: torch.Tensor | None,
        attn_tp_group: Any,
        tp_size: int,
    ) -> torch.Tensor:
        global_num_tokens = ctx.global_num_tokens
        if global_num_tokens:
            max_full_tokens = max(global_num_tokens)
        else:
            max_full_tokens = hidden_states.shape[0]
        moe_num_tokens = max(1, int(max_full_tokens))
        if hidden_states.shape[0] == 0:
            moe_input = hidden_states.new_empty((0, hidden_states.shape[-1]))
            self.ffn(
                moe_input,
                input_ids,
                padded_num_tokens=moe_num_tokens,
            )
            return hidden_states
        residual = hidden_states.clone()
        hidden_states, post, comb = self._pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = self.input_layernorm(hidden_states)
        attn_full = self.attn(
            hidden_states, positions, ctx, metadata, rope, compress_rope, pool
        )
        if attn_tp_group is not None and tp_size > 1:
            attn_reduced = attn_full.float()
            dist.all_reduce(attn_reduced, group=attn_tp_group)
            attn_full = attn_reduced.to(attn_full.dtype)
        hidden_states = self._post(attn_full, residual, post, comb)
        residual = hidden_states.clone()
        hidden_states, post, comb = self._pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output = self.ffn(
            hidden_states,
            input_ids,
            padded_num_tokens=moe_num_tokens,
        )
        hidden_states = self._post(moe_output, residual, post, comb)
        return hidden_states


class NpuDeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: Any | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _require_torch_npu()
        self.config = config
        self.mapping = mapping
        device = torch.device(torch.npu.current_device())
        self._metadata_builder: NpuDsaMetadataBuilder | None = None
        self.hadamard = _build_hadamard(int(config.index_head_dim), device)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                NpuDecoderLayer(
                    config,
                    mapping,
                    layer_id,
                    add_prefix(f"layers.{layer_id}", prefix),
                    self.hadamard,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = int(config.hc_mult) * int(config.hidden_size)
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.hc_eps = float(config.hc_eps)
        self.norm_eps = float(config.rms_norm_eps)
        rope_params = dict(getattr(config, "rope_parameters", {}) or {})
        if not rope_params:
            rope_params = dict(getattr(config, "rope_scaling", {}) or {})
        max_position = int(
            rope_params.get(
                "original_max_position_embeddings", config.max_position_embeddings
            )
        )
        factor = float(rope_params.get("factor", 16.0))
        rope_theta = float(
            rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        )
        self.rope = NpuYarnRopeCache(
            int(config.qk_rope_head_dim),
            int(config.max_position_embeddings),
            max_position,
            rope_theta,
            factor,
            int(rope_params.get("beta_fast", 32)),
            int(rope_params.get("beta_slow", 1)),
            device,
        )
        self.compress_rope = NpuYarnRopeCache(
            int(config.qk_rope_head_dim),
            int(config.max_position_embeddings),
            max_position,
            float(getattr(config, "compress_rope_theta", rope_theta)),
            factor,
            int(rope_params.get("beta_fast", 32)),
            int(rope_params.get("beta_slow", 1)),
            device,
        )

    def _hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        flat = hidden_states.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_head_fn) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        )
        return (pre.unsqueeze(-1) * hidden_states.view(shape)).sum(dim=1).to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden_states = input_embeds
        if hidden_states is None:
            hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(
            1, int(self.config.hc_mult), 1
        )
        metadata = None
        pool = None
        if hidden_states.shape[0] > 0:
            pool = ctx.token_to_kv_pool
            if not isinstance(pool, HybridDeepseekV4NpuTokenToKVPool):
                raise RuntimeError("DeepSeek-V4 NPU requires the NPU DSV4 cache pool")
            backend_metadata = ctx.attn_backend.forward_metadata
            if backend_metadata is None:
                raise RuntimeError("DeepSeek-V4 NPU requires forward metadata")
            if self._metadata_builder is None:
                graph = ctx.attn_backend.graph
                if graph is None:
                    raise RuntimeError("DeepSeek-V4 NPU graph state is not initialized")
                self._metadata_builder = NpuDsaMetadataBuilder(
                    self.config,
                    pool,
                    positions.device,
                    self.layers[0].attn.num_local_heads,
                    max_context_len=ctx.attn_backend.context_len,
                    max_batch_size=graph.max_bs,
                    max_table_pages={
                        group_id: int(table.shape[1])
                        for group_id, table in graph.block_tables.items()
                    },
                )
            builder = self._metadata_builder
            metadata = builder.build(
                backend_metadata, positions, self.rope, self.compress_rope
            )
        attn_tp_group = None
        if self.mapping.attn.tp_size > 1:
            attn_tp_group = pg_manager.get_process_group(
                "hccl", self.mapping.attn.tp_group
            )
        for layer in self.layers:
            hidden_states = layer(
                positions,
                hidden_states,
                ctx,
                metadata,
                self.rope,
                self.compress_rope,
                pool,
                input_ids,
                attn_tp_group,
                self.mapping.attn.tp_size,
            )
        aux_hidden_states = None
        if (
            ctx.capture_hidden_mode is not None
            and ctx.capture_hidden_mode.need_capture()
        ):
            aux_hidden_states = [hidden_states.flatten(1)]
        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, aux_hidden_states


class DeepseekV4ForCausalLM(BaseCausalLM):
    entry_platform = "npu"

    model_cls = NpuDeepseekV4Model

    @staticmethod
    def _map_name(name: str) -> str:
        if name.startswith("layers."):
            name = "model." + name
        elif name.startswith("embed."):
            name = name.replace("embed.", "model.embed_tokens.", 1)
        elif name.startswith("norm."):
            name = "model." + name
        elif name.startswith("hc_head"):
            name = "model." + name
        elif name == "head.weight":
            name = "lm_head.weight"
        if ".ffn_norm." in name:
            name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        if ".attn_norm." in name:
            name = name.replace(".attn_norm.", ".input_layernorm.")
        if ".ffn.shared_experts.w2." in name:
            name = name.replace(".ffn.shared_experts.w2.", ".ffn.shared_down.")
        if ".ffn.shared_experts.w1." in name:
            name = name.replace(
                ".ffn.shared_experts.w1.", ".ffn.shared_gate_up."
            )
        if ".ffn.shared_experts.w3." in name:
            name = name.replace(
                ".ffn.shared_experts.w3.", ".ffn.shared_gate_up."
            )
        if ".ffn.gate.bias" in name:
            name = name.replace(".ffn.gate.bias", ".ffn.e_score_correction_bias")
        if ".ffn.gate.tid2eid" in name:
            name = name.replace(".ffn.gate.tid2eid", ".ffn.tid2eid")
        return name

    def _load_dynamic(
        self,
        param_name: str,
        param: nn.Parameter,
        loaded: torch.Tensor,
        *,
        output_slice: bool = False,
        input_slice: bool = False,
        rank: int = 0,
        size: int = 1,
    ) -> bool:
        if output_slice:
            local_out = param.shape[0]
            param.data.copy_(loaded[rank * local_out : (rank + 1) * local_out])
            return True
        if input_slice:
            local_in = param.shape[1]
            param.data.copy_(loaded[:, rank * local_in : (rank + 1) * local_in])
            return True
        if param.shape == loaded.shape:
            param.data.copy_(loaded)
            return True
        return False

    @staticmethod
    def _load_merged_dynamic(
        param_name: str,
        param: nn.Parameter,
        loaded: torch.Tensor,
        output_offset: int,
    ) -> bool:
        if param_name.endswith(".weight"):
            param.data[
                output_offset : output_offset + loaded.shape[0]
            ].copy_(loaded)
            return True
        if param_name.endswith((".weight_scale", ".weight_offset")):
            param.data[
                output_offset : output_offset + loaded.shape[0], 0
            ].copy_(loaded.reshape(-1))
            return True
        return False

    def _load_expert_weight(
        self,
        name: str,
        loaded: torch.Tensor,
        params: dict[str, nn.Parameter],
        loaded_param_names: set[str],
        *,
        ep_rank: int,
        local_experts: int,
    ) -> None:
        match = re.search(r"\.ffn\.experts\.(\d+)\.", name)
        if match is None:
            raise ValueError(f"invalid DeepSeek-V4 expert checkpoint name: {name}")
        expert_id = int(match.group(1))
        if not (ep_rank * local_experts <= expert_id < (ep_rank + 1) * local_experts):
            return

        local_id = expert_id - ep_rank * local_experts
        prefix, suffix = name.split(f".ffn.experts.{expert_id}.", 1)
        if suffix in ("w1.weight", "w3.weight"):
            target = f"{prefix}.ffn.experts.w13_weight"
        elif suffix in ("w1.weight_scale", "w1.scale"):
            target = f"{prefix}.ffn.experts.w13_weight_scale"
        elif suffix in ("w1.weight_offset", "w1.offset"):
            target = f"{prefix}.ffn.experts.w13_weight_offset"
        elif suffix in ("w3.weight_scale", "w3.scale"):
            target = f"{prefix}.ffn.experts.w13_weight_scale"
        elif suffix in ("w3.weight_offset", "w3.offset"):
            target = f"{prefix}.ffn.experts.w13_weight_offset"
        elif suffix == "w2.weight":
            target = f"{prefix}.ffn.experts.w2_weight"
        elif suffix in ("w2.weight_scale", "w2.scale"):
            target = f"{prefix}.ffn.experts.w2_weight_scale"
        elif suffix in ("w2.weight_offset", "w2.offset"):
            target = f"{prefix}.ffn.experts.w2_weight_offset"
        else:
            raise ValueError(f"unsupported DeepSeek-V4 expert tensor: {name}")

        target_param = params.get(target)
        if target_param is None:
            raise ValueError(f"missing DeepSeek-V4 expert target: {target}")
        loaded_param_names.add(target)
        shard_offset = (
            0 if suffix.startswith("w1.") else int(self.config.moe_intermediate_size)
        )
        if target.endswith(".w13_weight"):
            target_param.data[
                local_id, shard_offset : shard_offset + loaded.shape[0]
            ].copy_(loaded)
        elif target.endswith(".w2_weight"):
            target_param.data[local_id].copy_(loaded)
        elif ".w13_weight_scale" in target or ".w13_weight_offset" in target:
            target_param.data[
                local_id, shard_offset : shard_offset + loaded.shape[0], 0
            ].copy_(loaded.reshape(-1))
        elif ".w2_weight_scale" in target or ".w2_weight_offset" in target:
            target_param.data[local_id, :, 0].copy_(loaded.reshape(-1))
        else:
            raise ValueError(f"unsupported DeepSeek-V4 expert target: {target}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        tp_rank = self.mapping.attn.tp_rank
        tp_size = self.mapping.attn.tp_size
        ep_rank = self.mapping.moe.ep_rank
        local_experts = self.config.n_routed_experts // self.mapping.moe.ep_size
        loaded_param_names = set()
        skipped_checkpoint_patterns = set()
        for raw_name, loaded in weights:
            name = self._map_name(raw_name)
            if name.startswith("mtp."):
                continue
            if name.endswith(".attn.wo_a.weight"):
                name = name[: -len(".weight")]
            if name.endswith(".attn.wo_b.weight"):
                name = name[: -len(".weight")]
            if ".ffn.experts." in name:
                self._load_expert_weight(
                    name,
                    loaded,
                    params,
                    loaded_param_names,
                    ep_rank=ep_rank,
                    local_experts=local_experts,
                )
                continue
            param = params.get(name)
            if param is None:
                pattern = re.sub(r"layers\.\d+", "layers.N", name)
                pattern = re.sub(r"experts\.\d+", "experts.N", pattern)
                skipped_checkpoint_patterns.add(pattern)
                continue
            loaded_param_names.add(name)
            if name == "model.embed_tokens.weight":
                self.model.embed_tokens.weight_loader(param, loaded)
                continue
            if name == "lm_head.weight":
                self.lm_head.weight_loader(param, loaded)
                continue
            if ".attn.attn_sink" in name:
                local_heads = param.shape[0]
                start = tp_rank * local_heads
                param.data.copy_(loaded[start : start + local_heads])
                continue
            if ".attn.wq_b." in name:
                output_slice = name.endswith(
                    (".weight", ".weight_scale", ".weight_offset")
                )
                if not self._load_dynamic(
                    name, param, loaded, output_slice=output_slice, rank=tp_rank
                ):
                    raise ValueError(f"shape mismatch loading {raw_name}")
                continue
            if ".attn.wq_a." in name or ".attn.wkv." in name:
                if not self._load_dynamic(name, param, loaded):
                    raise ValueError(f"unsupported attention tensor: {name}")
                continue
            if ".ffn.shared_gate_up." in name:
                output_offset = (
                    0
                    if ".ffn.shared_experts.w1." in raw_name
                    else int(self.config.moe_intermediate_size)
                )
                if not self._load_merged_dynamic(
                    name, param, loaded, output_offset
                ):
                    raise ValueError(f"unsupported merged shared tensor: {name}")
                continue
            if name.endswith(".attn.wo_a"):
                o_lora_rank = int(self.config.o_lora_rank)
                if tp_size <= int(self.config.o_groups):
                    num_local_groups = int(self.config.o_groups) // tp_size
                    local_out = num_local_groups * o_lora_rank
                    start = tp_rank * local_out
                    if param.ndim != 2 or param.shape != (
                        local_out,
                        loaded.shape[-1],
                    ):
                        param.data = torch.empty(
                            local_out,
                            loaded.shape[-1],
                            dtype=param.dtype,
                            device=param.device,
                        )
                    param.data.copy_(loaded[start : start + local_out])
                    param.data = (
                        param.data.view(num_local_groups, o_lora_rank, -1)
                        .transpose(1, 2)
                        .contiguous()
                    )
                else:
                    ranks_per_group = tp_size // int(self.config.o_groups)
                    if tp_size % int(self.config.o_groups) != 0:
                        raise ValueError(
                            "attn_tp_size must be a divisor or multiple of o_groups"
                        )
                    if loaded.shape[-1] % ranks_per_group != 0:
                        raise ValueError(
                            "wo_a input dimension cannot be evenly divided across ranks"
                        )
                    group_index = tp_rank // ranks_per_group
                    input_part = tp_rank % ranks_per_group
                    input_dim = loaded.shape[-1] // ranks_per_group
                    row_start = group_index * o_lora_rank
                    col_start = input_part * input_dim
                    param.data.copy_(
                        loaded[
                            row_start : row_start + o_lora_rank,
                            col_start : col_start + input_dim,
                        ]
                    )
                    param.data = (
                        param.data.view(1, o_lora_rank, input_dim)
                        .transpose(1, 2)
                        .contiguous()
                    )
                continue
            if name.endswith(".attn.wo_b"):
                if tp_size <= int(self.config.o_groups):
                    local_in = param.shape[1]
                    start = tp_rank * local_in
                    param.data.copy_(loaded[:, start : start + local_in])
                else:
                    ranks_per_group = tp_size // int(self.config.o_groups)
                    group_index = tp_rank // ranks_per_group
                    start = group_index * int(self.config.o_lora_rank)
                    param.data.copy_(
                        loaded[
                            :,
                            start : start + int(self.config.o_lora_rank),
                        ]
                    )
                continue
            if ".shared_down." in name:
                if name.endswith(".weight"):
                    param.data.copy_(loaded)
                else:
                    param.data.copy_(loaded.reshape(-1, 1))
                continue
            if param.shape != loaded.shape:
                raise ValueError(
                    f"shape mismatch loading {raw_name}: "
                    f"expected {tuple(param.shape)}, got {tuple(loaded.shape)}"
                )
            param.data.copy_(loaded)
        intentional_missing = {
            name
            for name in params
            if re.fullmatch(r"model\.layers\.\d+\.attn\.q_head_norm_weight", name)
        }
        missing_params = set(params) - loaded_param_names - intentional_missing
        if missing_params or skipped_checkpoint_patterns:
            raise ValueError(
                "DeepSeek-V4 NPU weight load mismatch: "
                f"missing model parameters={sorted(missing_params)}, "
                f"unmatched checkpoint patterns={sorted(skipped_checkpoint_patterns)}"
            )
        self.post_load_weights()

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, NpuMoE):
                module.process_weights_after_loading()
        for module in self.modules():
            if isinstance(module, NpuDynamicLinear):
                module.process_weights_after_loading()
            elif isinstance(module, NpuW8A8Experts):
                module.process_weights_after_loading()


EntryClass = [DeepseekV4ForCausalLM]
