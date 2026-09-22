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

from tokenspeed.runtime.models.deepseek_v4_npu import NpuMoE


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
