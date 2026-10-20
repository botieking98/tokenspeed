# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.deepseek_v4_npu import NpuMoE
from tokenspeed.runtime.models.deepseek_v4_npu import NpuDecoderLayer
from tokenspeed.runtime.models.deepseek_v4_npu import _mask_null_compressor_slots
from tokenspeed.runtime.models.deepseek_v4_npu import _moe_padded_num_tokens
from tokenspeed.runtime.models.deepseek_v4_npu import _scatter_token_counts
from tokenspeed.runtime.models.deepseek_v4_npu import _valid_tokens_per_request


@pytest.fixture(autouse=True)
def clear_active_masks():
    NpuMoE._active_masks.clear()
    NpuMoE._all_active_masks.clear()
    yield
    NpuMoE._active_masks.clear()
    NpuMoE._all_active_masks.clear()


def _moe(layer_id: int) -> NpuMoE:
    module = object.__new__(NpuMoE)
    module.layer_id = layer_id
    return module


def test_all_active_mask_is_reused_across_layers():
    device = torch.device("cpu")
    first_mask = _moe(layer_id=0)._active_mask(
        device=device, num_tokens=4, padded_num_tokens=4
    )
    second_mask = _moe(layer_id=1)._active_mask(
        device=device, num_tokens=4, padded_num_tokens=4
    )

    assert first_mask.tolist() == [True, True, True, True]
    assert second_mask.tolist() == [True, True, True, True]
    assert second_mask.data_ptr() == first_mask.data_ptr()


def test_partial_mask_is_updated_once_and_reused_by_later_layers():
    device = torch.device("cpu")
    first_layer_mask = _moe(layer_id=0)._active_mask(
        device=device, num_tokens=3, padded_num_tokens=8
    )
    later_layer_mask = _moe(layer_id=1)._active_mask(
        device=device, num_tokens=3, padded_num_tokens=8
    )

    assert first_layer_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert later_layer_mask.tolist() == first_layer_mask.tolist()
    assert later_layer_mask.data_ptr() == first_layer_mask.data_ptr()


def test_partial_mask_clears_stale_padding_for_smaller_batches():
    device = torch.device("cpu")
    module = _moe(layer_id=0)

    first_mask = module._active_mask(
        device=device, num_tokens=4, padded_num_tokens=8
    )
    first_mask_snapshot = first_mask.clone()
    second_mask = module._active_mask(
        device=device, num_tokens=2, padded_num_tokens=8
    )

    assert first_mask_snapshot.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert second_mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_padded_moe_input_uses_pointer_stable_valid_token_mask():
    module = _moe(layer_id=0)
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    input_ids = torch.tensor([11, 12, 13], dtype=torch.int64)
    valid_token_mask = torch.tensor([True, True, True, False])

    padded_hidden, padded_input_ids, active_mask = module._padded_moe_input(
        hidden_states=hidden_states,
        input_ids=input_ids,
        padded_num_tokens=4,
        active_mask=valid_token_mask,
    )

    assert padded_hidden.shape == (4, 2)
    assert padded_input_ids.tolist() == [11, 12, 13, 0]
    assert active_mask.data_ptr() == valid_token_mask.data_ptr()
    assert active_mask.tolist() == [True, True, True, False]

    valid_token_mask[2] = False
    assert active_mask.tolist() == [True, True, False, False]


def test_valid_tokens_per_request_excludes_graph_padding_rows():
    token_to_req_indices = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    is_valid_token = torch.tensor(
        [True, True, False, True, False, False], dtype=torch.bool
    )

    valid_counts = _valid_tokens_per_request(
        token_to_req_indices=token_to_req_indices,
        is_valid_token=is_valid_token,
        num_reqs=2,
    )

    assert valid_counts.tolist() == [2, 1]


def test_mask_null_compressor_slots_skips_null_pages():
    slot_mapping = torch.tensor([[1, 3], [0, 0], [-1, -1], [2, 7]], dtype=torch.int32)

    masked = _mask_null_compressor_slots(slot_mapping=slot_mapping)

    assert masked is slot_mapping
    assert masked.tolist() == [[1, 3], [-1, -1], [-1, -1], [2, 7]]


def test_decoder_layer_does_not_clone_residual(monkeypatch):
    layer = object.__new__(NpuDecoderLayer)
    layer.hc_mult = 1
    layer.hc_attn_fn = None
    layer.hc_attn_scale = None
    layer.hc_attn_base = None
    layer.hc_ffn_fn = None
    layer.hc_ffn_scale = None
    layer.hc_ffn_base = None
    monkeypatch.setattr(
        NpuDecoderLayer,
        "_pre",
        lambda self, hidden_states, fn, scale, base: (hidden_states, None, None),
    )
    monkeypatch.setattr(
        NpuDecoderLayer,
        "_post",
        lambda self, hidden_states, residual, post, comb: hidden_states + residual,
    )
    layer.input_layernorm = lambda hidden_states: hidden_states
    layer.post_attention_layernorm = lambda hidden_states: hidden_states
    layer.attn = lambda hidden_states, positions, ctx, metadata, rope, compress_rope, pool: hidden_states
    layer.ffn = (
        lambda hidden_states, input_ids, padded_num_tokens, active_mask: hidden_states
    )

    hidden_states = torch.ones((2, 3))
    ctx = SimpleNamespace(global_num_tokens=[])
    original_clone = torch.Tensor.clone
    torch.Tensor.clone = lambda self: (_ for _ in ()).throw(AssertionError)
    try:
        output = layer.forward(
            positions=torch.zeros(2),
            hidden_states=hidden_states,
            ctx=ctx,
            metadata=None,
            rope=None,
            compress_rope=None,
            pool=None,
            input_ids=None,
            attn_tp_group=None,
            tp_size=1,
        )
    finally:
        torch.Tensor.clone = original_clone

    assert output.tolist() == [
        [4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0],
    ]


def test_scatter_token_counts_distributes_remainder():
    assert _scatter_token_counts(10, 4) == [3, 3, 2, 2]
    assert _scatter_token_counts(8, 4) == [2, 2, 2, 2]


def test_moe_padded_num_tokens_uses_largest_dp_shard():
    ctx = SimpleNamespace(
        collective_global_num_tokens=None,
        global_num_tokens=[10, 10, 10, 10] + [0] * 12,
    )
    mapping = SimpleNamespace(
        attn=SimpleNamespace(dp_size=4, tp_size=4, cp_size=1, has_dp=True),
        moe=SimpleNamespace(has_tp_ep=True),
    )

    assert _moe_padded_num_tokens(ctx=ctx, mapping=mapping, num_tokens=10) == 3


def test_decoder_layer_prefill_shards_residual_and_input_ids(monkeypatch):
    layer = object.__new__(NpuDecoderLayer)
    layer.mapping = Mapping(
        rank=1,
        world_size=16,
        attn_tp_size=4,
        attn_dp_size=4,
        dense_tp_size=4,
        dense_dp_size=4,
        moe_tp_size=1,
        moe_ep_size=16,
        moe_dp_size=1,
    )
    layer.attn_tp_rank = 1
    layer.hc_mult = 2
    layer.hc_attn_fn = None
    layer.hc_attn_scale = None
    layer.hc_attn_base = None
    layer.hc_ffn_fn = None
    layer.hc_ffn_scale = None
    layer.hc_ffn_base = None

    calls = {}

    def fake_pre(self, hidden_states, fn, scale, base):
        del self
        del fn, scale, base
        if hidden_states.ndim == 3:
            return (
                hidden_states[:, 0, :],
                torch.arange(hidden_states.shape[0]),
                torch.arange(hidden_states.shape[0]),
            )
        return (
            hidden_states,
            torch.arange(hidden_states.shape[0]),
            torch.arange(hidden_states.shape[0]),
        )

    def fake_post(self, hidden_states, residual, post, comb):
        del self
        calls.setdefault(
            "post",
            [],
        ).append((hidden_states.clone(), residual.clone(), post.clone()))
        del comb
        return residual

    def fake_reduce_scatter(tensor, group, scattered_num_tokens):
        calls["reduce"] = (tensor.clone(), group, list(scattered_num_tokens))
        return tensor[3:6]

    def fake_all_gather(tensor, group, scattered_num_tokens):
        assert tensor.ndim == 2
        calls["gather"] = (tensor.clone(), group, list(scattered_num_tokens))
        return torch.ones((10, 6), dtype=tensor.dtype)

    monkeypatch.setattr(NpuDecoderLayer, "_pre", fake_pre)
    monkeypatch.setattr(NpuDecoderLayer, "_post", fake_post)
    monkeypatch.setattr(
        "tokenspeed.runtime.models.deepseek_v4_npu.token_reduce_scatter",
        fake_reduce_scatter,
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.models.deepseek_v4_npu.token_all_gather",
        fake_all_gather,
    )
    layer.input_layernorm = lambda hidden_states: hidden_states
    layer.post_attention_layernorm = lambda hidden_states: hidden_states
    layer.attn = lambda hidden_states, positions, ctx, metadata, rope, compress_rope, pool: hidden_states
    layer.ffn = lambda hidden_states, input_ids, padded_num_tokens, active_mask: (
        calls.setdefault("ffn", []).append(
            (input_ids, padded_num_tokens)
        )
        or hidden_states
    )

    ctx = SimpleNamespace(
        collective_global_num_tokens=None,
        global_num_tokens=[10, 10, 10, 10] + [0] * 12,
        forward_mode=SimpleNamespace(is_extend_or_mixed=lambda: True),
    )
    output = layer.forward(
        positions=torch.zeros(10),
        hidden_states=torch.ones((10, 2, 3)),
        ctx=ctx,
        metadata=None,
        rope=None,
        compress_rope=None,
        pool=None,
        input_ids=torch.arange(10),
        attn_tp_group=object(),
        tp_size=4,
    )

    assert calls["reduce"][1] == (0, 1, 2, 3)
    assert calls["reduce"][2] == [3, 3, 2, 2]
    assert calls["post"][0][2].tolist() == [3, 4, 5]
    assert calls["ffn"][0][0].tolist() == [3, 4, 5]
    assert calls["ffn"][0][1] == 3
    assert calls["gather"][1] == (0, 1, 2, 3)
    assert calls["gather"][2] == [3, 3, 2, 2]
    assert output.shape == (10, 2, 3)


def test_decoder_layer_empty_prefill_shard_participates_in_collectives(monkeypatch):
    layer = object.__new__(NpuDecoderLayer)
    layer.mapping = Mapping(
        rank=2,
        world_size=16,
        attn_tp_size=4,
        attn_dp_size=4,
        dense_tp_size=4,
        dense_dp_size=4,
        moe_tp_size=1,
        moe_ep_size=16,
        moe_dp_size=1,
    )
    layer.attn_tp_rank = 2
    layer.hc_mult = 2
    layer.hc_attn_fn = None
    layer.hc_attn_scale = None
    layer.hc_attn_base = None
    layer.hc_ffn_fn = None
    layer.hc_ffn_scale = None
    layer.hc_ffn_base = None
    calls = {}

    monkeypatch.setattr(
        NpuDecoderLayer,
        "_pre",
        lambda self, hidden_states, fn, scale, base: (
            hidden_states[:, 0, :],
            torch.arange(hidden_states.shape[0]),
            torch.arange(hidden_states.shape[0]),
        ),
    )
    monkeypatch.setattr(
        NpuDecoderLayer,
        "_post",
        lambda self, *args: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        "tokenspeed.runtime.models.deepseek_v4_npu.token_reduce_scatter",
        lambda tensor, group, scattered_num_tokens: tensor[0:0],
    )

    def fake_all_gather(tensor, group, scattered_num_tokens):
        assert tensor.shape[0] == 0
        calls["gather"] = (group, list(scattered_num_tokens))
        return torch.ones((1, 6), dtype=tensor.dtype)

    monkeypatch.setattr(
        "tokenspeed.runtime.models.deepseek_v4_npu.token_all_gather",
        fake_all_gather,
    )
    layer.input_layernorm = lambda hidden_states: hidden_states
    layer.post_attention_layernorm = lambda hidden_states: hidden_states
    layer.attn = lambda hidden_states, positions, ctx, metadata, rope, compress_rope, pool: hidden_states
    layer.ffn = lambda hidden_states, input_ids, padded_num_tokens, active_mask: (
        calls.setdefault("ffn", []).append(
            (hidden_states.shape[0], input_ids, padded_num_tokens)
        )
        or hidden_states
    )

    ctx = SimpleNamespace(
        collective_global_num_tokens=None,
        global_num_tokens=[1, 1, 1, 1] + [0] * 12,
        forward_mode=SimpleNamespace(is_extend_or_mixed=lambda: True),
    )
    output = layer.forward(
        positions=torch.zeros(1),
        hidden_states=torch.ones((1, 2, 3)),
        ctx=ctx,
        metadata=None,
        rope=None,
        compress_rope=None,
        pool=None,
        input_ids=torch.arange(1),
        attn_tp_group=object(),
        tp_size=4,
    )

    assert calls["ffn"][0][0] == 0
    assert calls["ffn"][0][1].tolist() == []
    assert calls["ffn"][0][2] == 1
    assert calls["gather"] == ((0, 1, 2, 3), [1, 0, 0, 0])
    assert output.shape == (1, 2, 3)
