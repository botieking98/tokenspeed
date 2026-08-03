"""GLM-5.1 (GlmMoeDsaForCausalLM) model for Ascend NPU.

MoE with MLA attention, W8A8 quantization, MC2 token dispatch.
TP=16, EP=16 on Ascend910. Uses npu_mla attention backend.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu

# Enable NPU internal format (required for NZ weight format in npu_quant_matmul)
torch_npu.npu.config.allow_internal_format = True
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.rotary_embedding import yarn_get_mscale
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.utils.common import add_prefix
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.distributed.comm_ops import all_reduce

logger = get_colorful_logger(__name__)

ACL_FORMAT_FRACTAL_NZ = 29
ACL_FORMAT_FRACTAL_ND = 2

def _load_ascend_custom_ops():
    """Load _C_ascend custom ops for int8×int8 quantized grouped matmul.

    Provides grouped_matmul_swiglu_quant_weight_nz (fused gmm+swiglu+quant)
    from vllm-ascend's CANN custom op vendor package.
    """
    import os
    vendor = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    if os.path.isdir(vendor):
        cur = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        if vendor not in cur:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor + (":" + cur if cur else "")
        lib = os.path.join(vendor, "op_api", "lib")
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib not in cur_ld:
            os.environ["LD_LIBRARY_PATH"] = lib + (":" + cur_ld if cur_ld else "")
    try:
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        return True
    except Exception:
        return False

_CUSTOM_OPS_LOADED = _load_ascend_custom_ops()


def _maybe_trans_nz(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return torch_npu.npu_format_cast(tensor.contiguous(), ACL_FORMAT_FRACTAL_NZ)
    except Exception:
        return tensor.contiguous()


def trans_rope_weight(weight, rope_dim):
    """Reorder the rope part of a weight to match npu_interleave_rope format."""
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    reordered_rope_part = torch.cat((rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, reordered_rope_part), dim=-2).contiguous()



class NPURotaryEmbedding:
    """Interleaved RoPE for GLM-5.1 (GPT-J style, is_neox_style=False).

    Uses npu_interleave_rope fused kernel (1 op vs 6+ manual ops).
    Output is in split-half layout; attention dot product is layout-
    invariant so both Q and K can use split-half format.
    """

    # Class-level shared indexed cos/sin — all layers use the same rope
    # parameters and the same positions tensor per forward pass, so the
    # gather (cos_cache[positions]) only needs to run once. The first
    # layer's __call__ computes and caches; subsequent layers reuse.
    _shared_cos = None
    _shared_sin = None
    _shared_positions_ptr = None

    def __init__(self, head_size: int, base: float, max_position: int):
        self.head_size = head_size
        self.base = base
        self.max_position = max_position
        self._device = None
        self._cos_cache = None
        self._sin_cache = None

    def _ensure_cache(self, device, dtype):
        if self._cos_cache is not None and self._cos_cache.device == device:
            return
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_size, 2, dtype=torch.float32, device=device) / self.head_size))
        t = torch.arange(self.max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_pos, head_size/2]
        cos_half = freqs.cos().to(dtype)
        sin_half = freqs.sin().to(dtype)
        self._cos_cache = cos_half.repeat(1, 2)  # [max_pos, D] concatenated
        self._sin_cache = sin_half.repeat(1, 2)

    def _apply_rope_op(self, x, cos, sin):
        """Apply npu_interleave_rope to x [N, ..., D].
        cos/sin: [N, D] (concatenated doubled format).
        Returns [N, ..., D] in split-half layout.
        """
        orig_shape = x.shape
        D = self.head_size
        if x.dim() == 3:
            N, H, _ = x.shape
            x_4d = x.unsqueeze(2)  # [N, H, 1, D]
        else:
            N = x.shape[0]
            x_4d = x.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, D]
        cos_4d = cos.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, D] (broadcast)
        sin_4d = sin.unsqueeze(1).unsqueeze(1)
        out = torch_npu.npu_interleave_rope(x_4d, cos_4d, sin_4d)
        return out.reshape(orig_shape)

    def __call__(self, positions, q, k):
        """Apply interleaved RoPE to q and k.
        q: [N, H, D], k: [N, D] or [N, H, D]
        Returns: (q_rotated, k_rotated) in split-half layout.
        """
        self._ensure_cache(q.device, q.dtype)
        ptr = positions.data_ptr()
        n = positions.shape[0]
        if (NPURotaryEmbedding._shared_positions_ptr != ptr
                or NPURotaryEmbedding._shared_cos is None
                or NPURotaryEmbedding._shared_cos.shape[0] != n
                or NPURotaryEmbedding._shared_cos.device != q.device):
            NPURotaryEmbedding._shared_cos = self._cos_cache[positions]
            NPURotaryEmbedding._shared_sin = self._sin_cache[positions]
            NPURotaryEmbedding._shared_positions_ptr = ptr
        cos = NPURotaryEmbedding._shared_cos
        sin = NPURotaryEmbedding._shared_sin
        q_out = self._apply_rope_op(q, cos, sin)
        k_out = self._apply_rope_op(k, cos, sin)
        return q_out, k_out

    def apply_q(self, positions, q):
        """Apply interleaved RoPE to q only."""
        self._ensure_cache(q.device, q.dtype)
        cos = self._cos_cache[positions]
        sin = self._sin_cache[positions]
        return self._apply_rope_op(q, cos, sin)


# ---------------------------------------------------------------------------
# W8A8 quantized layers


# ---------------------------------------------------------------------------
# W8A8 quantized layers
# ---------------------------------------------------------------------------

class W8A8StaticLinear(nn.Module):
    """Static W8A8 quantized linear (attention projections).

    Checkpoint format: weight [out, in] int8, deq_scale [out] f32,
    input_scale [1] f32, input_offset [1] int8, quant_bias [out] i32.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False
        )
        self.deq_scale = nn.Parameter(
            torch.empty(out_features, dtype=torch.float32), requires_grad=False
        )
        self.input_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.input_offset = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.quant_bias = nn.Parameter(
            torch.empty(out_features, dtype=torch.int32), requires_grad=False
        )

    def process_weights_after_loading(self, *args, **kwargs):
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = _maybe_trans_nz(self.weight.data)
        self.deq_scale.data = self.deq_scale.data.to(torch.float32).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quant_x = torch_npu.npu_quantize(
            x, self.input_scale.data, self.input_offset.data.to(torch.int8),
            torch.qint8, axis=1, div_mode=True,
        )
        bias = self.quant_bias
        if getattr(self, "is_row_parallel", False) and getattr(self, "tp_rank", 0) != 0:
            bias = None
        return torch_npu.npu_quant_matmul(
            quant_x, self.weight, self.deq_scale,
            bias=bias, output_dtype=x.dtype,
        )

class W8A8DynamicLinear(nn.Module):
    """Dynamic W8A8 quantized linear (MoE / shared expert weights).

    Checkpoint format: weight [out, in] int8, weight_scale [out, 1] f32,
    weight_offset [out, 1] f32.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False
        )
        self.weight_offset = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False
        )

    def process_weights_after_loading(self, *args, **kwargs):
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = _maybe_trans_nz(self.weight.data)
        self.weight_scale.data = self.weight_scale.data.flatten().contiguous()
        self.weight_offset.data = self.weight_offset.data.flatten().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        if pertoken_scale.dim() == 2:
            quant_x = quant_x.squeeze(1)
            pertoken_scale = pertoken_scale.squeeze(1)
        return torch_npu.npu_quant_matmul(
            quant_x, self.weight, self.weight_scale,
            pertoken_scale=pertoken_scale, output_dtype=x.dtype,
        )


class FusedGateUpMLP(nn.Module):
    """Dense SwiGLU MLP with W8A8 weights, fused gate+up projection."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_weight = nn.Parameter(
            torch.empty(hidden_size, 2 * intermediate_size, dtype=torch.int8),
            requires_grad=False,
        )
        self.gate_up_scale = nn.Parameter(
            torch.empty(2 * intermediate_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.gate_up_offset = nn.Parameter(
            torch.empty(2 * intermediate_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_weight = nn.Parameter(
            torch.empty(intermediate_size, hidden_size, dtype=torch.int8),
            requires_grad=False,
        )
        self.down_scale = nn.Parameter(
            torch.empty(hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_offset = nn.Parameter(
            torch.empty(hidden_size, dtype=torch.float32),
            requires_grad=False,
        )

    def load_from_linears(self, gate, up, down):
        with torch.no_grad():
            self.gate_up_weight.data = torch.cat([gate.weight.data, up.weight.data], dim=1)
            self.gate_up_scale.data = torch.cat([gate.weight_scale.data, up.weight_scale.data])
            self.gate_up_offset.data = torch.cat([gate.weight_offset.data, up.weight_offset.data])
            self.down_weight.data = down.weight.data
            self.down_scale.data = down.weight_scale.data
            self.down_offset.data = down.weight_offset.data

    def forward(self, x):
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        if pertoken_scale.dim() == 2:
            quant_x = quant_x.squeeze(1)
            pertoken_scale = pertoken_scale.squeeze(1)
        gate_up = torch_npu.npu_quant_matmul(
            quant_x, self.gate_up_weight, self.gate_up_scale,
            pertoken_scale=pertoken_scale, output_dtype=x.dtype,
        )
        quant_inter, pertoken_inter = torch_npu.npu_dequant_swiglu_quant(
            gate_up, quant_mode=1, activate_left=True, swiglu_mode=0,
        )
        return torch_npu.npu_quant_matmul(
            quant_inter, self.down_weight, self.down_scale,
            pertoken_scale=pertoken_inter, output_dtype=x.dtype,
        )


# ---------------------------------------------------------------------------
# MLA Attention
# ---------------------------------------------------------------------------

class GlmMoeDsaAttention(nn.Module):
    """MLA attention for GLM-5.1 on Ascend NPU.

    Uses npu_mla attention backend. Model provides q [N, H, qk_head_dim],
    k [N, 1, kv_lora+rope] (compressed), v is derived by backend via kv_b_proj.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float,
        rope_scaling: dict | None,
        max_position_embeddings: int,
        layer_id: int,
        mapping: Mapping,
        prefix: str = "",
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.num_local_heads = num_heads // mapping.attn.tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.prefix = prefix

        tp_size = mapping.attn.tp_size

        # Q projection: hidden -> q_lora -> local_heads * qk_head_dim
        self.q_a_proj = W8A8StaticLinear(hidden_size, q_lora_rank)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = W8A8StaticLinear(q_lora_rank, self.num_local_heads * self.qk_head_dim)
        self.q_b_proj.tp_size = tp_size
        self.q_b_proj.tp_rank = mapping.attn.tp_rank

        # KV projection: hidden -> (kv_lora + rope)
        self.kv_a_proj_with_mqa = W8A8StaticLinear(hidden_size, kv_lora_rank + qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        # kv_b_proj: decompresses kv_lora -> local_heads * (qk_nope + v_head)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, self.num_local_heads * (qk_nope_head_dim + v_head_dim),
            bias=False, dtype=torch.bfloat16,
        )
        self.kv_b_proj.tp_size = tp_size
        self.kv_b_proj.tp_rank = mapping.attn.tp_rank

        # O projection
        self.o_proj = W8A8StaticLinear(self.num_local_heads * v_head_dim, hidden_size)
        self.o_proj.tp_size = tp_size
        self.o_proj.tp_rank = mapping.attn.tp_rank
        self.o_proj.is_row_parallel = True
        self._tp_group_name = None
        self._o_proj_dequant_done = False
        self._qkv_a_fused = False

        # RoPE (interleaved) - use NPU native implementation
        rope_max_pos = min(max_position_embeddings, 40960)
        self.rotary_emb = NPURotaryEmbedding(qk_rope_head_dim, rope_theta, rope_max_pos)
        if rope_scaling and "factor" in rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # PagedAttention wrappers
        # attn_mqa: for decode (absorbed path, KV is compressed [kv_lora+rope])
        self.attn_mqa = PagedAttention(
            self.num_local_heads,
            kv_lora_rank + qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=kv_lora_rank,
        )
        # attn_mha: for prefill (non-absorbed, q/k/v are full)
        self.attn_mha = PagedAttention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
        )

    def _maybe_fuse_qkv_a(self):
        """Fuse q_a_proj + kv_a_proj_with_mqa into a single matmul.

        Both share the same input_scale/input_offset (verified from checkpoint),
        so one quantization + one matmul replaces two separate projections.
        """
        if self._qkv_a_fused:
            return
        with torch.no_grad():
            self._qkv_a_weight = torch.cat([
                self.q_a_proj.weight.data,
                self.kv_a_proj_with_mqa.weight.data,
            ], dim=1).contiguous()
            self._qkv_a_deq_scale = torch.cat([
                self.q_a_proj.deq_scale.data,
                self.kv_a_proj_with_mqa.deq_scale.data,
            ]).contiguous()
            self._qkv_a_quant_bias = torch.cat([
                self.q_a_proj.quant_bias.data,
                self.kv_a_proj_with_mqa.quant_bias.data,
            ]).contiguous()
            self._qkv_a_input_scale = self.q_a_proj.input_scale.data
            self._qkv_a_input_offset = self.q_a_proj.input_offset.data
            self._q_lora_rank = self.q_a_proj.out_features
            # Precompute expanded quant params for npu_add_rms_norm_quant
            # (requires float32 scale and int32 offset, broadcast to [hidden_size])
            hidden_size = self.q_a_proj.in_features
            self._qkv_a_quant_scale_f32 = self._qkv_a_input_scale.expand(hidden_size).contiguous()
            self._qkv_a_quant_offset_i32 = self._qkv_a_input_offset.to(torch.int32).expand(hidden_size).contiguous()
        self._qkv_a_fused = True

    def _maybe_prepare_q_b_quant(self):
        """Precompute expanded quant params for q_b_proj to enable
        npu_add_rms_norm_quant fusion (rms_norm + quantize in one kernel).

        Uses npu_add_rms_norm_quant with x2=zeros to achieve rms_norm+quant
        without precision loss (float32 scale, unlike npu_rms_norm_quant
        which requires bf16 scale). Saves 1 kernel launch per layer.
        """
        if hasattr(self, "_q_b_scale_expanded"):
            return
        with torch.no_grad():
            q_lora = self.q_lora_rank
            self._q_b_scale_expanded = self.q_b_proj.input_scale.data.expand(q_lora).contiguous()
            self._q_b_offset_expanded = self.q_b_proj.input_offset.data.to(torch.int32).expand(q_lora).contiguous()

    def _dequant_o_proj(self):
        """Prepare o_proj params for npu_mm_all_reduce_base antiquant fusion.

        antiquant_scale = deq_scale / input_scale (per-channel, bf16).
        input_offset folded into x: x_adj = x + input_offset * input_scale.
        quant_bias used as bias (rank 0 only, added once before all_reduce).
        """
        if self._o_proj_dequant_done:
            return
        with torch.no_grad():
            ds = self.o_proj.deq_scale.data.float()
            s = self.o_proj.input_scale.data.float()
            self._o_proj_antiquant_scale = (ds / s).to(torch.bfloat16).contiguous()
            self._o_proj_input_adjust = (
                self.o_proj.input_offset.data.float() * s
            ).to(torch.bfloat16).item()
            # npu_mm_all_reduce_base adds bias BEFORE all_reduce (per-NPU),
            # so quant_bias only on rank 0 (added once after all_reduce sum).
            if self.o_proj.tp_rank == 0:
                self._o_proj_quant_bias_or_none = self.o_proj.quant_bias.data
            else:
                self._o_proj_quant_bias_or_none = None
        self._o_proj_dequant_done = True

    def _init_tp_group_name(self):
        """Get HCCL group name for TP all_reduce fusion."""
        if self._tp_group_name is not None:
            return
        try:
            import torch.distributed as dist
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )
            group = self.mapping.attn.tp_group
            device_group = pg_manager.get_process_group("hccl", group)
            local_rank = dist.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self._tp_group_name = backend.get_hccl_comm_name(local_rank)
        except Exception as e:
            logger.warning("TP group name init failed: %s", e)
            self._tp_group_name = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        pre_quantized: bool = False,
    ) -> torch.Tensor:
        orig_dtype = torch.bfloat16 if pre_quantized else hidden_states.dtype
        num_tokens = hidden_states.shape[0]
        num_decodes = ctx.bs - ctx.num_extends
        num_decode_tokens = num_decodes * ctx.attn_backend.spec_num_tokens
        num_prefill_tokens = num_tokens - num_decode_tokens

        # Fused Q + KV projection (one quantize + one matmul)
        self._maybe_fuse_qkv_a()
        if pre_quantized:
            quant_x = hidden_states
        else:
            quant_x = torch_npu.npu_quantize(
                hidden_states, self._qkv_a_input_scale,
                self._qkv_a_input_offset.to(torch.int8),
                torch.qint8, axis=1, div_mode=True,
            )
        qkv_a = torch_npu.npu_quant_matmul(
            quant_x, self._qkv_a_weight, self._qkv_a_deq_scale,
            bias=self._qkv_a_quant_bias, output_dtype=orig_dtype,
        )
        q_a = qkv_a[..., :self._q_lora_rank]
        latent_cache = qkv_a[..., self._q_lora_rank:]

        # Fuse q_a RMSNorm + q_b_proj quantize via npu_add_rms_norm_quant
        # (x2=zeros → add 0 is identity), saving 1 kernel per layer.
        self._maybe_prepare_q_b_quant()
        if not hasattr(self, "_q_a_zeros") or self._q_a_zeros.shape[0] < num_tokens:
            self._q_a_zeros = torch.zeros(
                num_tokens, self.q_lora_rank,
                dtype=torch.bfloat16, device=q_a.device,
            )
        q_quant, _, _ = torch_npu.npu_add_rms_norm_quant(
            q_a, self._q_a_zeros[:num_tokens],
            self.q_a_layernorm.weight.data,
            self._q_b_scale_expanded, self._q_b_offset_expanded,
            None, epsilon=self.q_a_layernorm.variance_epsilon,
            div_mode=True,
        )
        q = torch_npu.npu_quant_matmul(
            q_quant, self.q_b_proj.weight, self.q_b_proj.deq_scale,
            bias=self.q_b_proj.quant_bias, output_dtype=orig_dtype,
        )
        q = q.view(num_tokens, self.num_local_heads, self.qk_head_dim)

        if num_prefill_tokens > 0 and num_decode_tokens > 0:
            attn_output = torch.empty(
                num_tokens, self.num_local_heads * self.v_head_dim,
                dtype=orig_dtype, device=hidden_states.device,
            )
            prefill_out = self._forward_prefill(
                positions[:num_prefill_tokens],
                q[:num_prefill_tokens],
                latent_cache[:num_prefill_tokens],
                ctx,
                out_cache_loc[:num_prefill_tokens],
            )
            attn_output[:num_prefill_tokens] = prefill_out
            decode_out = self._forward_decode(
                positions[num_prefill_tokens:],
                q[num_prefill_tokens:],
                latent_cache[num_prefill_tokens:],
                ctx,
                out_cache_loc[num_prefill_tokens:],
            )
            attn_output[num_prefill_tokens:] = decode_out
        elif num_decode_tokens > 0:
            attn_output = self._forward_decode(
                positions, q, latent_cache,
                ctx, out_cache_loc,
            )
        else:
            attn_output = self._forward_prefill(
                positions, q, latent_cache,
                ctx, out_cache_loc,
            )

        if self.o_proj.tp_size > 1:
            output = self.o_proj(attn_output)
            output = all_reduce(output, self.mapping.attn.tp_group)
        else:
            output = self.o_proj(attn_output)
        return output

    def _forward_prefill(
        self, positions, q, latent_cache, ctx, out_cache_loc
    ) -> torch.Tensor:
        """Prefill: decompress KV, apply RoPE, run full attention."""
        N = q.shape[0]

        # Decompress KV (latent_cache is raw, apply RMSNorm to lora part)
        kv_a = latent_cache[..., :self.kv_lora_rank]
        k_pe = latent_cache[..., self.kv_lora_rank:]
        kv_a_norm, _ = torch_npu.npu_rms_norm(kv_a, self.kv_a_layernorm.weight, epsilon=self.kv_a_layernorm.variance_epsilon)
        kv = self.kv_b_proj(kv_a_norm)  # [N, local_heads*(qk_nope+v)]
        kv = kv.view(N, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., :self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim:]

        # Apply RoPE to q_pe [N, H, D] and k_pe [N, D] (interleaved)
        q_pe = q[..., self.qk_nope_head_dim:]
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe_rotated, k_pe_rotated = self.rotary_emb(positions, q_pe, k_pe)
        # Expand rotated k_pe for attention (shared across heads)
        k_pe_expanded = k_pe_rotated.unsqueeze(1).expand(-1, self.num_local_heads, -1)
        q_full = torch.cat([q_nope, q_pe_rotated], dim=-1)
        k_full = torch.cat([k_nope, k_pe_expanded], dim=-1)

        # Write KV cache (compressed: kv_a_norm + ROTATED k_pe)
        ctx.token_to_kv_pool.set_mla_kv_buffer(
            self.attn_mqa, out_cache_loc,
            cache_k_nope=kv_a_norm,
            cache_k_rope=k_pe_rotated,
        )

        # Run attention via backend
        output = self.attn_mha(
            q_full, k_full, v, ctx, out_cache_loc, save_kv_cache=False,
        )
        return output

    def _forward_decode(
        self, positions, q, latent_cache, ctx, out_cache_loc
    ) -> torch.Tensor:
        """Decode: absorbed MLA path using compressed KV cache."""
        q_pe = q[..., self.qk_nope_head_dim:]
        q_nope = q[..., :self.qk_nope_head_dim]

        # Precomputed q_absorb weight (avoid per-step reshape/slice)
        if not hasattr(self, '_w_kc'):
            w_kv = self.kv_b_proj.weight
            w_kv = w_kv.view(self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            self._w_kc = w_kv[:, :self.qk_nope_head_dim, :].contiguous()
        # npu_transpose_batchmatmul fuses bmm + input/output transpose,
        # producing contiguous [N, H, kv_lora] without intermediate copy.
        q_absorbed = torch_npu.npu_transpose_batchmatmul(
            q_nope, self._w_kc, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))

        # Apply RMSNorm to kv_a
        kv_a = latent_cache[..., :self.kv_lora_rank]
        k_pe = latent_cache[..., self.kv_lora_rank:]
        kv_a_norm, _ = torch_npu.npu_rms_norm(
            kv_a, self.kv_a_layernorm.weight,
            epsilon=self.kv_a_layernorm.variance_epsilon,
        )
        q_pe_rotated, k_pe_rotated = self.rotary_emb(positions, q_pe, k_pe)

        # Write KV cache (ROTATED k_pe)
        ctx.token_to_kv_pool.set_mla_kv_buffer(
            self.attn_mqa, out_cache_loc,
            cache_k_nope=kv_a_norm,
            cache_k_rope=k_pe_rotated,
        )

        # Run attention via backend (absorbed path)
        # Pass q_absorbed [N, H, kv_lora] and q_pe_rotated [N, H, rope]
        # separately to avoid cat+slice+contiguous round-trip.
        self.attn_mqa.kv_b_proj = self.kv_b_proj
        output = self.attn_mqa(
            q_absorbed, None, None,
            ctx, out_cache_loc, save_kv_cache=False,
            q_pe_split=q_pe_rotated,
        )
        return output


# ---------------------------------------------------------------------------
# Dense MLP (first 3 layers)
# ---------------------------------------------------------------------------

class GlmMoeDsaMLP(nn.Module):
    """Dense SwiGLU MLP with W8A8 weights, fused gate+up."""

    def __init__(self, hidden_size: int, intermediate_size: int, mapping: Mapping, prefix: str = ""):
        super().__init__()
        self.mapping = mapping
        self._fused = FusedGateUpMLP(hidden_size, intermediate_size)
        # Keep individual layers for weight loading, then fuse
        self.gate_proj = W8A8DynamicLinear(hidden_size, intermediate_size)
        self.up_proj = W8A8DynamicLinear(hidden_size, intermediate_size)
        self.down_proj = W8A8DynamicLinear(intermediate_size, hidden_size)
        self._fused_loaded = False

    def _maybe_fuse(self):
        if not self._fused_loaded:
            self._fused.load_from_linears(self.gate_proj, self.up_proj, self.down_proj)
            del self.gate_proj
            del self.up_proj
            del self.down_proj
            self._fused_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_fuse()
        return self._fused(x)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class GlmMoeExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = W8A8DynamicLinear(hidden_size, intermediate_size)
        self.up_proj = W8A8DynamicLinear(hidden_size, intermediate_size)
        self.down_proj = W8A8DynamicLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        return self.down_proj(x)


class GlmMoeDsaMoE(nn.Module):
    """MoE with W8A8 experts, MC2 dispatch (decode) + AllGather (prefill)."""

    _weights_prepared_logged = False

    def __init__(
        self, config: PretrainedConfig, mapping: Mapping, layer_index: int, prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.prefix = prefix

        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        ep_size = mapping.moe.ep_size
        self.ep_size = ep_size
        self.ep_rank = mapping.moe.ep_rank
        self.num_local_experts = self.num_experts // ep_size if ep_size > 0 else self.num_experts

        # Router
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=torch.bfloat16)
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32), requires_grad=False
        )

        # Local experts
        self.experts = nn.ModuleList([
            GlmMoeExpert(self.hidden_size, self.moe_intermediate_size)
            for _ in range(self.num_local_experts)
        ])

        # Shared expert
        if self.n_shared_experts > 0:
            self.shared_experts = GlmMoeDsaMLP(
                self.hidden_size, self.moe_intermediate_size, mapping
            )
        else:
            self.shared_experts = None

        self._mc2_group_name = None
        self._mc2_initialized = False

    def _init_mc2(self):
        if self._mc2_initialized:
            return
        import torch.distributed as dist
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )
        try:
            # Try attn.tp_group first (known to work with HCCL all_reduce).
            # Both tp_group and tp_ep_group contain all 16 ranks with TP=1,EP=16.
            group = self.mapping.attn.tp_group
            device_group = pg_manager.get_process_group("hccl", group)
            local_rank = dist.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self._mc2_group_name = backend.get_hccl_comm_name(local_rank)
            self._ep_world_size = self.mapping.attn.tp_size
            self._ep_rank_id = self.mapping.attn.tp_rank
            logger.info("MC2 init using attn.tp_group: group=%s ep_world_size=%d ep_rank_id=%d",
                       self._mc2_group_name, self._ep_world_size, self._ep_rank_id)
        except Exception as e:
            logger.warning("MC2 init failed: %s, using all_reduce", e)
            self._mc2_group_name = None
        self._mc2_initialized = True

    def _prepare_grouped_weights(self):
        """Stack expert weights into 3D ND tensors for npu_grouped_matmul antiquant.

        Fuses gate+up into w13 [E, in, 2*out] to reduce gmm calls from 3 to 2.
        Individual int8 weights [in, out] (NZ format) are converted to ND format
        and stacked into [E, in, out]. Scales/offsets stacked as [E, out] bf16.
        Uses antiquant path: BF16 activations x INT8 weights with on-the-fly
        dequantization (no extra memory for dequantized weights).
        """
        if hasattr(self, "_grouped_w13_weight"):
            return

        E = self.num_local_experts

        def stack_nd(experts, proj_name):
            """Convert NZ int8 weights to ND and stack: [E, in, out] int8 ND."""
            weights = []
            for e in experts:
                w = getattr(e, proj_name)
                w_nd = torch.empty(w.weight.shape, dtype=w.weight.dtype, device=w.weight.device)
                w_nd.copy_(w.weight)  # NZ -> ND copy
                weights.append(w_nd)
            return torch.stack(weights, dim=0).contiguous()

        gate_weight = stack_nd(self.experts, "gate_proj")  # [E, in, out]
        up_weight = stack_nd(self.experts, "up_proj")      # [E, in, out]
        down_weight = stack_nd(self.experts, "down_proj")   # [E, inter, hidden]

        # Fuse gate + up: [E, in, 2*inter] ND, then convert to NZ format
        nd_w13 = torch.cat([gate_weight, up_weight], dim=2).contiguous()
        del gate_weight, up_weight
        nd_w2 = down_weight

        # Convert to NZ format (saves memory: only one copy, used for both
        # quant gmm and antiquant paths, matching vllm-ascend approach)
        self._grouped_w13_weight = torch_npu.npu_format_cast(nd_w13, ACL_FORMAT_FRACTAL_NZ)
        del nd_w13
        self._grouped_w2_weight = torch_npu.npu_format_cast(nd_w2, ACL_FORMAT_FRACTAL_NZ)
        del nd_w2

        # Float32 scales for quant gmm path: [E, out]
        gate_scale = torch.stack(
            [e.gate_proj.weight_scale for e in self.experts], dim=0).contiguous()
        up_scale = torch.stack(
            [e.up_proj.weight_scale for e in self.experts], dim=0).contiguous()
        down_scale = torch.stack(
            [e.down_proj.weight_scale for e in self.experts], dim=0).contiguous()
        self._quant_w13_scale = torch.cat([gate_scale, up_scale], dim=1).contiguous()
        self._quant_w2_scale = down_scale
        del gate_scale, up_scale

        # bf16 scales for antiquant path (derived from float32)
        self._grouped_w13_scale = self._quant_w13_scale.to(torch.bfloat16).contiguous()
        self._grouped_w2_scale = self._quant_w2_scale.to(torch.bfloat16).contiguous()
        # Offsets are zero (symmetric quantization - verified from checkpoint)
        self._grouped_w13_offset = torch.zeros_like(self._grouped_w13_scale)
        self._grouped_w2_offset = torch.zeros_like(self._grouped_w2_scale)

        # Free individual expert weights to reclaim memory
        import gc
        device = self._grouped_w13_weight.device
        for e in self.experts:
            for proj in [e.gate_proj, e.up_proj, e.down_proj]:
                proj.weight = nn.Parameter(
                    torch.empty(1, dtype=torch.int8, device=device), requires_grad=False)
                proj.weight_scale = nn.Parameter(
                    torch.empty(1, dtype=torch.float32, device=device), requires_grad=False)
                proj.weight_offset = nn.Parameter(
                    torch.empty(1, dtype=torch.float32, device=device), requires_grad=False)
        gc.collect()
        torch.npu.empty_cache()

        if self.mapping.rank == 0 and not GlmMoeDsaMoE._weights_prepared_logged:
            GlmMoeDsaMoE._weights_prepared_logged = True
            logger.info("Prepared grouped weights (ND+NZ, custom_ops=%s): w13=%s w2=%s",
                         _CUSTOM_OPS_LOADED,
                         str(self._grouped_w13_weight.shape),
                         str(self._grouped_w2_weight.shape))


    @staticmethod
    def _dequant_grouped(int32_out, stacked_scales, pertoken_scale, group_list, num_experts):
        """Graph-safe dequant: int32 * w_scale * pertoken_scale."""
        N = int32_out.shape[0]
        if N == 0:
            return torch.zeros(0, int32_out.shape[1], dtype=torch.bfloat16,
                               device=int32_out.device)
        positions = torch.arange(N, device=int32_out.device)
        expert_idx = torch.searchsorted(group_list, positions, right=True)
        expert_idx = expert_idx.clamp(0, num_experts - 1)
        gathered_scales = stacked_scales[expert_idx]
        out = (int32_out.to(torch.float32)
               * gathered_scales
               * pertoken_scale.unsqueeze(1)).to(torch.bfloat16)
        return out

    def _select_experts(self, hidden_states, router_logits):
        """Fused MoE gating top-k via npu_moe_gating_top_k.

        Replaces 8 separate kernels (sigmoid + bias + topk + norm + scale +
        casts) with a single fused NPU kernel. The fused kernel uses unbiased
        sigmoid scores for routing weights (bias only affects expert selection),
        matching vllm-ascend grouped-topk behavior.
        """
        if not hasattr(self, "_bias_cast"):
            self._bias_cast = self.e_score_correction_bias.to(router_logits.dtype)
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits,
            k=self.top_k,
            bias=self._bias_cast,
            k_group=1,
            group_count=1,
            group_select_mode=1,
            renorm=0,
            norm_type=1,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
        )
        return topk_weights, topk_ids

    _shared_expert_stream = None

    def forward(self, hidden_states: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        if _CUSTOM_OPS_LOADED and self.shared_experts is not None:
            # Launch shared_experts early on se_stream to overlap with the
            # entire MoE pipeline (gate + routing + gmm + unpermute + all_reduce).
            # shared_experts only needs hidden_states, not moe_output.
            if GlmMoeDsaMoE._shared_expert_stream is None:
                GlmMoeDsaMoE._shared_expert_stream = torch_npu.npu.Stream(
                    device=hidden_states.device)
            se_stream = GlmMoeDsaMoE._shared_expert_stream
            hs_ready = torch.npu.current_stream().record_event()
            se_stream.wait_event(hs_ready)
            with torch_npu.npu.stream(se_stream):
                shared_output = self.shared_experts(hidden_states)

        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self._select_experts(hidden_states, router_logits)

        if _CUSTOM_OPS_LOADED:
            moe_output = self._forward_quant_gmm(
                hidden_states, topk_weights, topk_ids, ctx, skip_allreduce=True)
            if self.shared_experts is not None:
                if self.ep_size > 1:
                    moe_output = all_reduce(moe_output, self.mapping.attn.tp_group)
                torch.npu.current_stream().wait_stream(se_stream)
                output = moe_output + shared_output
            else:
                if self.ep_size > 1:
                    moe_output = all_reduce(moe_output, self.mapping.attn.tp_group)
                output = moe_output
        else:
            output = self._forward_grouped_matmul(hidden_states, topk_weights, topk_ids, ctx)
            if self.shared_experts is not None:
                output = output + self.shared_experts(hidden_states)
        return output

    def _mc2_available(self):
        """Check if MC2 dispatch/combine ops are available and initialized."""
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            return False
        if not hasattr(torch_npu, "npu_moe_distribute_combine_v2"):
            return False
        self._init_mc2()
        return self._mc2_group_name is not None

    def _forward_mc2(self, hidden_states, topk_weights, topk_ids, ctx):
        """MC2 MoE with int8 comm (quant_mode=2) + custom op gmm.

        Eliminates all_reduce: dispatch+combine replaces init_routing+unpermute+all_reduce.
        quant_mode=2 enables int8 communication (4x less data than bf16).
        Dispatch returns int8 expand_x + dynamic_scale, fed directly to custom op.
        """
        N = hidden_states.shape[0]

        self._prepare_grouped_weights()

        if not hasattr(self, "_mc2_mask") or self._mc2_mask.shape[0] != N:
            self._mc2_mask = torch.ones(N, dtype=torch.bool, device=hidden_states.device)

        # MC2 dispatch with int8 communication
        dispatch_out = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states,
            expert_ids=topk_ids.to(torch.int32),
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_experts,
            global_bs=0,
            expert_token_nums_type=0,
            x_active_mask=self._mc2_mask,
            scales=None,
            quant_mode=2,
            group_ep=self._mc2_group_name,
            ep_world_size=self._ep_world_size,
            ep_rank_id=self._ep_rank_id,
            group_tp=self._mc2_group_name,
            tp_world_size=1,
            tp_rank_id=0,
        )
        expand_x = dispatch_out[0]       # int8 [N_exp, hidden]
        dynamic_scale = dispatch_out[1]  # per-token scale [N_exp]
        assist_info_for_combine = dispatch_out[2]
        expert_token_nums = dispatch_out[3]
        ep_recv_counts = dispatch_out[4]
        expand_scales = dispatch_out[6]

        # Build cumsum group_list padded to num_local_experts
        group_list = expert_token_nums.to(torch.int64)
        if group_list.numel() < self.num_local_experts:
            pad_len = self.num_local_experts - group_list.numel()
            if group_list.numel() > 0:
                group_list = torch.cat([group_list, group_list[-1:].repeat(pad_len)])
            else:
                group_list = torch.zeros(self.num_local_experts,
                    dtype=torch.int64, device=hidden_states.device)

        # Squeeze dynamic_scale if 2D
        if dynamic_scale.dim() == 2:
            dynamic_scale = dynamic_scale.squeeze(1)

        # gmm1: fused gate_up + swiglu + quant (INT8*INT8->INT8 + scale)
        gate_up_out, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(
            x=expand_x,
            weight=self._grouped_w13_weight,
            weight_scale=self._quant_w13_scale,
            x_scale=dynamic_scale,
            group_list=group_list,
            bias=None,
            swiglu_limit=0.0,
        )

        # gmm2: down_proj (INT8*INT8->BF16, dequant via scale + per_token_scale)
        down_out = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[self._grouped_w2_weight],
            scale=[self._grouped_w2_scale],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=group_list,
            output_dtype=hidden_states.dtype,
        )[0]

        # MC2 combine with int8 communication
        output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=down_out,
            expert_ids=topk_ids.to(torch.int32),
            expert_scales=topk_weights.to(torch.float32),
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_experts,
            global_bs=0,
            x_active_mask=self._mc2_mask,
            ep_send_counts=ep_recv_counts,
            group_ep=self._mc2_group_name,
            ep_world_size=self._ep_world_size,
            ep_rank_id=self._ep_rank_id,
            expand_scales=expand_scales,
            comm_quant_mode=2,
            assist_info_for_combine=assist_info_for_combine,
            tp_send_counts=ep_recv_counts,
            group_tp=self._mc2_group_name,
            tp_world_size=1,
            tp_rank_id=0,
        )

        return output

    def _forward_fused_mc2(self, hidden_states, topk_weights, topk_ids, ctx):
        """Fully fused MC2: dispatch+gmm+swiglu+gmm+combine in ONE kernel.

        Uses _C_ascend.dispatch_gmm_combine_decode which eliminates:
        - init_routing, dynamic_quant, gmm1, swiglu, gmm2, unpermute, all_reduce
        All replaced by a single fused C++ kernel with integrated HCCL communication.
        This is vllm-ascend's enable_fused_mc2=2 path.
        """
        N = hidden_states.shape[0]
        self._prepare_grouped_weights()
        self._init_mc2()

        if not hasattr(self, "_mc2_mask") or self._mc2_mask.shape[0] != N:
            self._mc2_mask = torch.ones(N, dtype=torch.bool, device=hidden_states.device)

        output, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=hidden_states,
            expert_ids=topk_ids.to(torch.int32),
            gmm1_permuted_weight=[self._grouped_w13_weight],
            gmm1_permuted_weight_scale=[self._quant_w13_scale],
            gmm2_weight=[self._grouped_w2_weight],
            gmm2_weight_scale=[self._quant_w2_scale],
            expert_scales=topk_weights.to(torch.float32),
            expert_smooth_scales=None,
            group_ep=self._mc2_group_name,
            ep_rank_size=self._ep_world_size,
            ep_rank_id=self._ep_rank_id,
            moe_expert_num=self.num_experts,
            global_bs=0,
        )

        return output

    def _forward_quant_gmm(self, hidden_states, topk_weights, topk_ids, ctx, skip_allreduce=False):
        """Quantized MoE: INT8×INT8 via _C_ascend custom ops.

        Routes bf16 tokens, quantizes to int8, then uses:
        - gmm1: grouped_matmul_swiglu_quant_weight_nz (fused gmm+swiglu+quant -> int8)
        - gmm2: npu_grouped_matmul quant path (int8*int8->bf16 via scale+per_token_scale)
        Both use full int8 cube throughput, ~2x faster than antiquant (bf16*int8).
        """
        N = hidden_states.shape[0]
        K = self.top_k
        E = self.num_local_experts
        first_expert = self.ep_rank * E
        global_experts = self.num_experts

        self._prepare_grouped_weights()

        # Mask topk_weights for non-local experts (pre-cast to bf16 for unpermute)
        local_mask = (topk_ids >= first_expert) & (topk_ids < first_expert + E)
        topk_weights_masked = topk_weights * local_mask.to(topk_weights.dtype)

        # Token dispatch with fused dynamic quant (quant_mode=1):
        # bf16 → int8 expanded_x + per-token scale in one kernel
        expanded_x, expanded_row_idx, expert_tokens, expanded_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=N * K,
            expert_num=global_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert, first_expert + E],
            quant_mode=1,
        )
        group_list_cumsum = expert_tokens.cumsum(0)

        # expanded_x is already int8, expanded_scale is per-token scale
        quant_x = expanded_x
        pertoken_scale = expanded_scale

        # gmm1: fused gate_up + swiglu + quant (INT8*INT8->INT8 + scale)
        gate_up_out, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(
            x=quant_x,
            weight=self._grouped_w13_weight,
            weight_scale=self._quant_w13_scale,
            x_scale=pertoken_scale,
            group_list=group_list_cumsum,
            bias=None,
            swiglu_limit=0.0,
        )

        # gmm2: down_proj (INT8*INT8->BF16, dequant via scale + per_token_scale)
        down_out = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[self._grouped_w2_weight],
            scale=[self._grouped_w2_scale],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=group_list_cumsum,
            output_dtype=hidden_states.dtype,
        )[0]

        # Unpermute: scatter back to [N, H] with topk weights
        output = torch_npu.npu_moe_token_unpermute(
            down_out,
            expanded_row_idx.abs(),
            probs=topk_weights_masked,
        )

        if self.ep_size > 1 and not skip_allreduce:
            output = all_reduce(output, self.mapping.attn.tp_group)

        return output

    def _prepare_quant_weights(self):
        """Quant weights are prepared in _prepare_grouped_weights (NZ + float32 scales)."""
        pass

    def _forward_grouped_matmul(self, hidden_states, topk_weights, topk_ids, ctx):
        """Fused MoE with token dispatch via npu_moe_init_routing_v2.

        Only tokens routing to local experts are processed (eliminates E×
        redundant compute). Uses fused gate_up (w13) weight for 2 gmm calls
        instead of 3. Antiquant path: BF16 activations x INT8 weights.
        """
        N = hidden_states.shape[0]
        K = self.top_k
        E = self.num_local_experts
        first_expert = self.ep_rank * E
        global_experts = self.num_experts

        self._prepare_grouped_weights()

        # Mask topk_weights for non-local experts (zero contribution)
        local_mask = (topk_ids >= first_expert) & (topk_ids < first_expert + E)
        topk_weights_masked = topk_weights * local_mask.to(topk_weights.dtype)

        # Token dispatch: sort tokens by local expert
        expanded_x, expanded_row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=N * K,
            expert_num=global_experts,
            expert_tokens_num_type=1,   # count mode
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert, first_expert + E],
            quant_mode=-1,              # no quantization
        )

        group_list = expert_tokens.to(torch.int64)

        # gmm1: fused gate_up_proj (antiquant: BF16 x INT8 ND)
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[self._grouped_w13_weight],
            antiquant_scale=[self._grouped_w13_scale],
            antiquant_offset=[self._grouped_w13_offset],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=hidden_states.dtype,
        )[0]  # [N*K, 2*inter]

        # Fused swiglu activation
        inter = torch_npu.npu_swiglu(gate_up_out)  # [N*K, inter]

        # gmm2: down_proj
        down_out = torch_npu.npu_grouped_matmul(
            x=[inter],
            weight=[self._grouped_w2_weight],
            antiquant_scale=[self._grouped_w2_scale],
            antiquant_offset=[self._grouped_w2_offset],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=hidden_states.dtype,
        )[0]  # [N*K, hidden]

        # Unpermute: scatter back to [N, H] with topk weights
        # abs() handles -1 invalid entries (their probs are 0)
        output = torch_npu.npu_moe_token_unpermute(
            down_out,
            torch.abs(expanded_row_idx),
            probs=topk_weights_masked,
        )

        # all_reduce on attn.tp_group
        if self.ep_size > 1:
            output = all_reduce(output, self.mapping.attn.tp_group)

        return output

    def _forward_allgather(self, hidden_states, topk_weights, topk_ids, ctx):
        import torch.distributed as dist
        from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend

        N = hidden_states.shape[0]
        backend = get_global_backend()
        group = self.mapping.moe.tp_ep_group if self.ep_size > 1 else None

        if self.ep_size > 1:
            gathered = backend.all_gather(hidden_states, group, dim=0)
            gathered_weights = backend.all_gather(topk_weights, group, dim=0)
            gathered_ids = backend.all_gather(topk_ids, group, dim=0)
        else:
            gathered = hidden_states
            gathered_weights = topk_weights
            gathered_ids = topk_ids

        total_tokens = gathered.shape[0]
        first_expert = self.ep_rank * self.num_local_experts
        E = self.num_local_experts

        # Shared dynamic quantization for gate/up (same input, avoid 16x redundant quant)
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(gathered, dst_type=torch.int8)
        if pertoken_scale.dim() == 2:
            quant_x = quant_x.squeeze(1)
            pertoken_scale = pertoken_scale.squeeze(1)

        all_inter = []
        for expert in self.experts:
            gate = torch_npu.npu_quant_matmul(
                quant_x, expert.gate_proj.weight, expert.gate_proj.weight_scale,
                pertoken_scale=pertoken_scale, output_dtype=gathered.dtype,
            )
            up = torch_npu.npu_quant_matmul(
                quant_x, expert.up_proj.weight, expert.up_proj.weight_scale,
                pertoken_scale=pertoken_scale, output_dtype=gathered.dtype,
            )
            all_inter.append(F.silu(gate) * up)

        all_outs = torch.stack(
            [expert.down_proj(inter) for expert, inter in zip(self.experts, all_inter)]
        )  # [E, total_tokens, hidden]

        # Optimized masking: one-hot + bmm (top_k iterations instead of top_k * E)
        output = torch.zeros_like(gathered)
        all_outs_perm = all_outs.permute(1, 0, 2)  # [total_tokens, E, hidden]
        for k in range(self.top_k):
            local_id = gathered_ids[:, k] - first_expert
            valid = (local_id >= 0) & (local_id < E)
            local_id_clamped = local_id.clamp(0, E - 1).long()
            one_hot = torch.zeros(total_tokens, E, device=gathered.device, dtype=gathered.dtype)
            one_hot.scatter_(1, local_id_clamped.unsqueeze(1), valid.unsqueeze(1).to(gathered.dtype))
            selected = torch.bmm(one_hot.unsqueeze(1), all_outs_perm).squeeze(1)
            output += selected * gathered_weights[:, k:k+1]

        if self.ep_size > 1:
            output = backend.reduce_scatter(output, group)

        return output if self.ep_size <= 1 else output


    def _forward_prefill_allreduce(self, hidden_states, topk_weights, topk_ids, ctx):
        """Prefill path using all_reduce on attn.tp_group.

        Avoids moe.tp_ep_group HCCL (all_gather/reduce_scatter) which deadlock
        after graph capture. Each rank computes its local experts on all tokens
        (TP-replicated input), then all_reduce combines results.
        """
        N = hidden_states.shape[0]
        first_expert = self.ep_rank * self.num_local_experts
        E = self.num_local_experts

        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
        if pertoken_scale.dim() == 2:
            quant_x = quant_x.squeeze(1)
            pertoken_scale = pertoken_scale.squeeze(1)

        # Compute only local experts
        all_inter = []
        for expert in self.experts:
            gate = torch_npu.npu_quant_matmul(
                quant_x, expert.gate_proj.weight, expert.gate_proj.weight_scale,
                pertoken_scale=pertoken_scale, output_dtype=hidden_states.dtype,
            )
            up = torch_npu.npu_quant_matmul(
                quant_x, expert.up_proj.weight, expert.up_proj.weight_scale,
                pertoken_scale=pertoken_scale, output_dtype=hidden_states.dtype,
            )
            all_inter.append(F.silu(gate) * up)

        all_outs = torch.stack(
            [expert.down_proj(inter) for expert, inter in zip(self.experts, all_inter)]
        )  # [E, N, hidden]

        # Mask: only keep local expert contributions
        output = torch.zeros_like(hidden_states)
        all_outs_perm = all_outs.permute(1, 0, 2)  # [N, E, hidden]
        for k in range(self.top_k):
            local_id = topk_ids[:, k] - first_expert
            valid = (local_id >= 0) & (local_id < E)
            local_id_clamped = local_id.clamp(0, E - 1).long()
            one_hot = torch.zeros(N, E, device=hidden_states.device, dtype=hidden_states.dtype)
            one_hot.scatter_(1, local_id_clamped.unsqueeze(1), valid.unsqueeze(1).to(hidden_states.dtype))
            selected = torch.bmm(one_hot.unsqueeze(1), all_outs_perm).squeeze(1)
            output += selected * topk_weights[:, k:k+1]

        # all_reduce on attn.tp_group (works after graph capture, unlike moe.tp_ep_group)
        if self.ep_size > 1:
            output = all_reduce(output, self.mapping.attn.tp_group)

        return output

# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, mapping, prefix=""):
        super().__init__()
        self.layer_id = layer_id
        self.mapping = mapping
        self.hidden_size = config.hidden_size

        rope_params = getattr(config, "rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", get_rope_theta(config))
        rope_scaling = getattr(config, "rope_scaling", None)
        self.is_moe_layer = (
            layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )

        self.self_attn = GlmMoeDsaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            layer_id=layer_id,
            mapping=mapping,
            prefix=add_prefix("self_attn", prefix),
        )

        if self.is_moe_layer:
            self.mlp = GlmMoeDsaMoE(
                config=config, mapping=mapping, layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = GlmMoeDsaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mapping=mapping,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, ctx, out_cache_loc, residual):
        pre_quantized = False
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif not ctx.forward_mode.is_idle():
            # Fuse add + rms_norm + quant: produces int8 for attention qkv_a
            # and new residual in one kernel, saving 2 kernel launches/layer.
            self.self_attn._maybe_fuse_qkv_a()
            hidden_states, _, residual = torch_npu.npu_add_rms_norm_quant(
                hidden_states, residual, self.input_layernorm.weight.data,
                self.self_attn._qkv_a_quant_scale_f32,
                self.self_attn._qkv_a_quant_offset_i32,
                None, epsilon=self.input_layernorm.variance_epsilon,
            )
            pre_quantized = True
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if not ctx.forward_mode.is_idle():
            hidden_states = self.self_attn(positions, hidden_states, ctx, out_cache_loc, pre_quantized=pre_quantized)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            if self.is_moe_layer:
                hidden_states = self.mlp(hidden_states, ctx)
            else:
                hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GlmMoeDsaModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, mapping, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.prefix = prefix
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GlmMoeDsaDecoderLayer(
                config, layer_id, mapping=mapping,
                prefix=add_prefix(f"layers.{layer_id}", prefix),
            )
            for layer_id in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: set = set()

    def forward(self, input_ids, positions, ctx, out_cache_loc, **kwargs):
        # Invalidate shared cos/sin cache — positions may reuse the same
        # buffer across forward passes with different values.
        NPURotaryEmbedding._shared_positions_ptr = None
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, ctx, out_cache_loc, residual)
        if not ctx.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, None


class GlmMoeDsaForCausalLM(BaseCausalLM):
    model_cls = GlmMoeDsaModel

    def __init__(self, config, mapping, quant_config=None, prefix=""):
        super().__init__(config, mapping, quant_config, prefix)

    def get_skip_weight_names(self):
        return ["rotary_emb.inv_freq", "indexer"]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs):
        params_dict = dict(self.named_parameters())
        ep_size = self.mapping.moe.ep_size
        ep_rank = self.mapping.moe.ep_rank
        tp_size = self.mapping.attn.tp_size
        tp_rank = self.mapping.attn.tp_rank
        num_local = self.config.n_routed_experts // ep_size if ep_size > 0 else self.config.n_routed_experts
        first_expert = ep_rank * num_local

        # TP-sharded weight names (output dim = dim 0 for ColumnParallel, dim 1 for RowParallel)
        # q_b_proj: weight [out, in] -> shard dim 0 (ColumnParallel)
        # kv_b_proj: weight [out, in] -> shard dim 0 (ColumnParallel)
        # o_proj: weight [out, in] -> shard dim 1 (RowParallel)
        # deq_scale [out] -> shard dim 0
        # quant_bias [out] -> shard dim 0
        column_shard_suffixes = (
            "q_b_proj.weight", "q_b_proj.deq_scale", "q_b_proj.quant_bias",
            "kv_b_proj.weight",
        )
        row_shard_suffixes = (
            "o_proj.weight",
        )
        # o_proj: deq_scale [out] is NOT sharded (full output)
        # o_proj: quant_bias [out] is NOT sharded

        for name, loaded_weight in weights:
            if "rotary_emb" in name or "indexer" in name:
                continue

            # Map global expert index to local
            if ".experts." in name:
                parts = name.split(".experts.")
                expert_idx = int(parts[1].split(".")[0])
                local_idx = expert_idx - first_expert
                if local_idx < 0 or local_idx >= num_local:
                    continue
                name = parts[0] + f".experts.{local_idx}." + ".".join(parts[1].split(".")[1:])

            if name not in params_dict:
                continue

            param = params_dict[name]

            # Apply TP sharding for attention projections
            if tp_size > 1:
                shard_dim = None
                for suffix in column_shard_suffixes:
                    if name.endswith(suffix):
                        shard_dim = 0
                        break
                if shard_dim is None:
                    for suffix in row_shard_suffixes:
                        if name.endswith(suffix):
                            shard_dim = 1
                            break

                if shard_dim is not None and loaded_weight.shape[shard_dim] == param.shape[shard_dim] * tp_size:
                    # Shard the loaded weight
                    chunk_size = loaded_weight.shape[shard_dim] // tp_size
                    start = tp_rank * chunk_size
                    end = start + chunk_size
                    if shard_dim == 0:
                        loaded_weight = loaded_weight[start:end]
                    else:
                        loaded_weight = loaded_weight[:, start:end]

            # Use the param's weight_loader if it has one (handles TP sharding
            # for ParallelLMHead, VocabParallelEmbedding, etc.)
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            elif param.shape == loaded_weight.shape:
                param.data.copy_(loaded_weight)
            else:
                # Fallback: shard based on output_dim attribute (e.g. lm_head)
                output_dim = getattr(param, "output_dim", None)
                if (output_dim is not None
                        and loaded_weight.shape[output_dim]
                        == param.shape[output_dim] * tp_size):
                    chunk = loaded_weight.shape[output_dim] // tp_size
                    start = tp_rank * chunk
                    end = start + chunk
                    if output_dim == 0:
                        loaded_weight = loaded_weight[start:end]
                    else:
                        loaded_weight = loaded_weight[:, start:end]
                    param.data.copy_(loaded_weight)
                else:
                    logger.warning("Shape mismatch for %s: param %s vs loaded %s",
                                   name, param.shape, loaded_weight.shape)


EntryClass = [GlmMoeDsaForCausalLM]
