"""Ascend W8A8 quantization layers for GLM-5.1 on NPU.

Two formats:
- Static W8A8 (attention projections): deq_scale, input_scale, input_offset, quant_bias
- Dynamic W8A8 (MoE experts): weight_scale, weight_offset + per-token dynamic quant
"""

from __future__ import annotations

import torch
import torch_npu
from torch import nn


def _maybe_trans_nz(tensor: torch.Tensor) -> torch.Tensor:
    """Cast weight to FRACTAL_NZ format for faster grouped matmul on NPU."""
    try:
        return torch_npu.npu_format_cast(tensor, 29)
    except Exception:
        return tensor


class AscendW8A8StaticLinear(nn.Module):
    """Static W8A8 quantized linear for attention projections.

    Weight format: int8 [out, in] with deq_scale [out], input_scale [1],
    input_offset [1], quant_bias [out] (int32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.deq_scale = nn.Parameter(
            torch.empty(out_features, dtype=torch.float32), requires_grad=False
        )
        self.input_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.input_offset = nn.Parameter(
            torch.empty(1, dtype=torch.int8), requires_grad=False
        )
        self.quant_bias = nn.Parameter(
            torch.empty(out_features, dtype=torch.int32), requires_grad=False
        )

    def process_weights_after_loading(self):
        import numpy as np

        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = _maybe_trans_nz(self.weight.data)

        scale = self.deq_scale.data.to(torch.float32)
        scale_int64 = torch.from_numpy(
            np.frombuffer(scale.cpu().numpy().tobytes(), dtype=np.int32).astype(np.int64)
        ).to(scale.device)
        self.deq_scale = nn.Parameter(scale_int64, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quant_x = torch.ops.vllm.quantize(
            x,
            self.input_scale.data,
            1.0 / self.input_scale.data,
            self.input_offset.data,
        )
        output = torch_npu.npu_quant_matmul(
            quant_x,
            self.weight,
            self.deq_scale,
            bias=self.quant_bias,
            output_dtype=x.dtype,
        )
        return output


class AscendW8A8DynamicLinear(nn.Module):
    """Dynamic W8A8 quantized linear for MoE expert weights.

    Weight format: int8 [out, in] with weight_scale [out, 1], weight_offset [out, 1].
    Activation is dynamically quantized per-token.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False
        )
        self.weight_offset = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False
        )

    def process_weights_after_loading(self):
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = _maybe_trans_nz(self.weight.data)
        self.weight_scale.data = self.weight_scale.data.flatten()
        self.weight_offset.data = self.weight_offset.data.flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        if pertoken_scale.dim() == 2:
            quant_x = quant_x.squeeze(1)
            pertoken_scale = pertoken_scale.squeeze(1)
        output = torch_npu.npu_quant_matmul(
            quant_x,
            self.weight,
            self.weight_scale,
            pertoken_scale=pertoken_scale,
            output_dtype=x.dtype,
        )
        return output
