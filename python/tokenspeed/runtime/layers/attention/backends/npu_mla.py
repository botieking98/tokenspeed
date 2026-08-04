"""NPU MLA attention backend for Ascend using npu_fusion_attention.

Implements decode and extend paths using torch_npu's npu_fusion_attention
with paged KV cache support.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu



def _is_graph_capturing() -> bool:
    """Check if NPU graph capture is in progress."""
    try:
        from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
        return get_is_capture_mode()
    except Exception:
        return False


def _weak_ref_tensor(tensor):
    """Create a weak reference to a tensor (vllm-ascend pattern).

    The tensor shares the same data pointer but does not prevent GC.
    This avoids OOM during graph capture when storing per-layer params.
    """
    if isinstance(tensor, torch.Tensor):
        return torch_npu._C._weak_ref_tensor(tensor)
    return tensor


from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass(kw_only=True)
class NPUMlaDecodeMetadata:
    num_extends: int
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_list: list[int]


@dataclass(kw_only=True)
class NPUMlaPrefillMetadata:
    seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int


class NPUMlaAttnBackend(AttentionBackend):
    """MLA attention backend for Ascend NPU using npu_fusion_attention."""

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size

        self.forward_decode_metadata: NPUMlaDecodeMetadata | None = None
        self.forward_prefill_metadata: NPUMlaPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, NPUMlaDecodeMetadata] = {}
        self.cuda_graph_page_table: torch.Tensor | None = None
        self.cuda_graph_seq_lens: torch.Tensor | None = None
        self._decode_workspace: torch.Tensor | None = None
        self._decode_attn_output: torch.Tensor | None = None
        self._decode_softmax_lse: torch.Tensor | None = None
        self._graph_handles: dict[int, list] = {}
        self._graph_events: dict[int, list] = {}
        self._npu_update_stream = None
        self._capture_layer_params: dict[int, list[tuple]] = {}

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        if forward_mode.is_extend_or_mixed():
            self.forward_prefill_metadata = NPUMlaPrefillMetadata(
                seq_lens=seq_lens[:num_extends],
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                cum_extend_seq_lens=torch.zeros(
                    extend_seq_lens.shape[0] + 1,
                    dtype=torch.int32,
                    device=seq_lens.device,
                ),
                extend_seq_lens_cpu=[int(x) for x in extend_seq_lens_cpu.tolist()],
                max_extend_seq_len=int(extend_seq_lens_cpu.max().item()),
                max_extend_prefix_len=int(
                    extend_prefix_lens_cpu.max().item()
                ) if extend_prefix_lens_cpu is not None and extend_prefix_lens_cpu.numel() > 0 else 0,
            )
            cumsum = 0
            for i, s in enumerate(self.forward_prefill_metadata.extend_seq_lens_cpu):
                cumsum += s
                self.forward_prefill_metadata.cum_extend_seq_lens[i + 1] = cumsum

        if forward_mode.is_decode() or forward_mode.is_mixed():
            page_table = build_page_table(
                req_pool_indices[:bs],
                req_to_page,
                self.page_size,
                self.max_context_len,
            )
            self.forward_decode_metadata = NPUMlaDecodeMetadata(
                num_extends=num_extends,
                page_table=page_table,
                seq_lens=seq_lens[:bs],
                seq_lens_list=seq_lens[:bs].tolist(),
            )

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_seq_lens = seq_lens_buf
        self.decode_cuda_graph_metadata = {}

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        self._graph_handles[bs] = []
        self._graph_events[bs] = []
        metadata = NPUMlaDecodeMetadata(
            num_extends=0,
            page_table=self.cuda_graph_page_table[:bs, :],
            seq_lens=self.cuda_graph_seq_lens[:bs],
            seq_lens_list=self.cuda_graph_seq_lens[:bs].tolist(),
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
            req_to_page[req_pool_indices[:bs], : self.max_num_pages]
        )
        self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        self.forward_decode_metadata = self.decode_cuda_graph_metadata[bs]

    def update_graph_params(self, bs: int, stream):
        """Update actual_seq_kvlen in captured NPU graph task groups.

        NPU graph_task_group regions require graph_task_update_begin/end
        before each replay to rebind host-side parameters (actual_seq_kvlen).
        Without this, graph.replay() hangs.
        """
        if bs not in self._graph_handles:
            return
        handles = self._graph_handles[bs]
        events = self._graph_events[bs]
        layer_params = self._capture_layer_params.get(bs)
        if not handles or not layer_params:
            return
        if self._npu_update_stream is None:
            self._npu_update_stream = torch.npu.Stream()
        update_stream = self._npu_update_stream
        seq_lens_list = self.cuda_graph_seq_lens[:bs].tolist()
        page_table = self.cuda_graph_page_table[:bs, :]
        with torch.npu.stream(update_stream):
            for handle, event, (q_nope, k_nope, q_pe, k_pe,
                                attn_output, softmax_lse) in zip(
                handles, events, layer_params
            ):
                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    q_nope, k_nope, k_nope,
                    query_rope=q_pe,
                    key_rope=k_pe,
                    num_query_heads=self.num_local_heads,
                    num_key_value_heads=1,
                    input_layout="BNSD_NBSD",
                    atten_mask=None,
                    sparse_mode=0,
                    softmax_scale=self.scaling,
                    block_table=page_table,
                    block_size=self.page_size,
                    actual_seq_qlen=None,
                    actual_seq_kvlen=seq_lens_list,
                    workspace=self._decode_workspace,
                    out=[attn_output, softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if save_kv_cache:
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        assert metadata is not None
        num_extends = metadata.num_extends

        q_decode = q[num_extends:] if num_extends > 0 else q
        page_table = metadata.page_table
        seq_lens = metadata.seq_lens
        num_tokens = q_decode.shape[0]

        # Q is [N, H, kv_lora] (q_absorbed) with q_pe passed separately
        q_pe_split = kwargs.get("q_pe_split", None)
        if q_pe_split is not None:
            q_nope = q_decode.reshape(num_tokens, self.num_local_heads, 1, self.kv_lora_rank)
            q_pe = q_pe_split.reshape(num_tokens, self.num_local_heads, 1, self.qk_rope_head_dim)
        else:
            # Fallback: Q is [N, H, kv_lora + rope] (legacy combined format)
            q_nope = q_decode[..., : self.kv_lora_rank]
            q_pe = q_decode[..., self.kv_lora_rank :]
            q_nope = q_nope.reshape(num_tokens, self.num_local_heads, 1, self.kv_lora_rank).contiguous()
            q_pe = q_pe.reshape(num_tokens, self.num_local_heads, 1, self.qk_rope_head_dim)

        # KV cache: separate k_nope and k_pe buffers (already contiguous)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        block_size = self.page_size
        if isinstance(kv_cache, tuple):
            k_nope_full, k_pe_full = kv_cache
        else:
            k_nope_full = kv_cache[..., : self.kv_lora_rank]
            k_pe_full = kv_cache[..., self.kv_lora_rank :]
        num_blocks = k_nope_full.shape[0] // block_size
        k_nope = k_nope_full[: num_blocks * block_size].view(
            num_blocks, 1, block_size, self.kv_lora_rank
        )
        k_pe = k_pe_full[: num_blocks * block_size].view(
            num_blocks, 1, block_size, self.qk_rope_head_dim
        )

        # V decompression weight from kv_b_proj (cached to avoid per-step .contiguous())
        kv_b_proj = getattr(layer, "kv_b_proj", None)
        if kv_b_proj is None:
            raise RuntimeError("layer.kv_b_proj not found for NPU MLA decode")
        w_uv = getattr(layer, "_cached_w_uv", None)
        if w_uv is None:
            w_kv = kv_b_proj.weight.view(
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            w_uv = w_kv[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
            layer._cached_w_uv = w_uv

        # Paged attention via npu_fused_infer_attention_score_v2
        # Pre-allocate workspace and output as instance attributes so they
        # persist across graph capture/replay (torch.empty inside capture
        # does not allocate from the graph pool on NPU).
        if self._decode_attn_output is None or self._decode_attn_output.shape[1] != num_tokens:
            self._decode_attn_output = torch.empty(
                self.num_local_heads, num_tokens, 1, self.kv_lora_rank,
                dtype=q_nope.dtype, device=q_nope.device,
            )
            self._decode_softmax_lse = torch.empty(
                num_tokens, dtype=q_nope.dtype, device=q_nope.device,
            )
        attn_output = self._decode_attn_output
        softmax_lse = self._decode_softmax_lse
        common_kwargs = dict(
            query_rope=q_pe,
            key_rope=k_pe,
            num_query_heads=self.num_local_heads,
            num_key_value_heads=1,
            input_layout="BNSD_NBSD",
            atten_mask=None,
            sparse_mode=0,
            softmax_scale=self.scaling,
            block_table=page_table,
            block_size=block_size,
            actual_seq_qlen=None,
            actual_seq_kvlen=metadata.seq_lens_list,
        )
        if self._decode_workspace is None:
            self._decode_workspace = (
                torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                    q_nope, k_nope, k_nope, **common_kwargs
                )
            )
        if _is_graph_capturing():
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                q_nope, k_nope, k_nope,
                **common_kwargs,
                workspace=self._decode_workspace,
                out=[attn_output, softmax_lse],
            )
            handle = torch.npu.graph_task_group_end(stream)
            if num_tokens in self._graph_handles:
                self._graph_handles[num_tokens].append(handle)
                self._graph_events[num_tokens].append(event)
                if num_tokens not in self._capture_layer_params:
                    self._capture_layer_params[num_tokens] = []
                self._capture_layer_params[num_tokens].append(
                    (_weak_ref_tensor(q_nope), _weak_ref_tensor(k_nope),
                     _weak_ref_tensor(q_pe), _weak_ref_tensor(k_pe),
                     _weak_ref_tensor(attn_output), _weak_ref_tensor(softmax_lse))
                )
        else:
            torch_npu.npu_fused_infer_attention_score_v2.out(
                q_nope, k_nope, k_nope,
                **common_kwargs,
                workspace=self._decode_workspace,
                out=[attn_output, softmax_lse],
            )
        # attn_output: [H, N, 1, kv_lora] (NBSD output layout)

        # V decompression: [H, N, kv_lora] @ [H, kv_lora, v] -> [H, N, v]
        attn_output = attn_output.view(self.num_local_heads, num_tokens, self.kv_lora_rank)
        # npu_transpose_batchmatmul fuses bmm + output transpose (perm_y),
        # eliminating the .permute().contiguous() full copy per layer.
        v_output = torch_npu.npu_transpose_batchmatmul(attn_output, w_uv, perm_y=(1, 0, 2))

        return v_output.reshape(num_tokens, self.num_local_heads * self.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if save_kv_cache:
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.qk_nope_head_dim + self.qk_rope_head_dim :],
            )

        metadata = self.forward_prefill_metadata
        assert metadata is not None

        q = q.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        k = k.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        v = v.view(-1, self.num_local_heads, self.v_head_dim)

        # Use npu_fused_infer_attention_score (8.6x faster than
        # npu_fusion_attention for inference prefill).
        # Group by request (varlen not supported on this NPU).
        extend_seq_lens_cpu = metadata.extend_seq_lens_cpu
        outputs = []
        offset = 0
        for i, seq_len in enumerate(extend_seq_lens_cpu):
            if seq_len == 0:
                continue
            qi = q[offset:offset + seq_len]  # [seq, H, D]
            ki = k[offset:offset + seq_len]
            vi = v[offset:offset + seq_len]

            qi = qi.transpose(0, 1).unsqueeze(0)  # [1, H, seq, D]
            ki = ki.transpose(0, 1).unsqueeze(0)
            vi = vi.transpose(0, 1).unsqueeze(0)

            # Causal mask: upper triangle = True (masked), lower = False
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            out, _ = torch_npu.npu_fused_infer_attention_score(
                qi, ki, vi,
                num_heads=self.num_local_heads,
                input_layout="BNSD",
                scale=self.scaling,
                atten_mask=causal_mask,
                num_key_value_heads=self.num_local_heads,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))  # [seq, H, D]
            offset += seq_len

        if outputs:
            output = torch.cat(outputs, dim=0)
        else:
            output = torch.empty(0, self.num_local_heads, self.v_head_dim,
                                 dtype=q.dtype, device=q.device)
        return output.reshape(-1, self.num_local_heads * self.v_head_dim)


register_backend("npu_mla", {AttentionArch.MLA}, NPUMlaAttnBackend)
