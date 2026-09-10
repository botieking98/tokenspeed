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

"""Ascend 910C cache view for DeepSeek V4."""

from __future__ import annotations

from typing import ClassVar

from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    HybridDeepseekV4TokenToKVPool,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    DeepseekV4CacheLayout,
)

_NPU_DSV4_MAX_ZERO_RANGES_PER_LAUNCH = 2047


class HybridDeepseekV4NpuTokenToKVPool(HybridDeepseekV4TokenToKVPool):
    """Bind ACLNN-shaped DSV4 fields to the upstream shared arena."""

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **HybridDeepseekV4TokenToKVPool.layer_plane_bindings,
        "indexer_scale": "indexer_scale_buffer",
    }

    def __init__(
        self,
        arena: CacheArena,
        layout: DeepseekV4CacheLayout,
        layer_num: int,
        rank: int,
        field_layer_offset: int = 0,
    ) -> None:
        super().__init__(
            arena=arena,
            layout=layout,
            layer_num=layer_num,
            rank=rank,
            field_layer_offset=field_layer_offset,
        )
        self.layout = layout
        prefix_granularity = self.arena.prefix_granularity
        self.compressed_block_sizes = tuple(
            32 if ratio > 1 else prefix_granularity
            for ratio in layout.layer_ratio
        )
        self.indexer_block_sizes = tuple(
            32 if ratio == 4 else 0 for ratio in layout.layer_ratio
        )

    def get_indexer_scale_buffer(self, layer_id: int):
        return self._require(
            self.indexer_scale_buffer, layer_id, "indexer scale"
        )

    def zero_new_blocks(self, new_page_ids: dict[str, list[int]]) -> None:
        """Zero field payloads in NPU-safe range batches.

        Ascend validates a Triton launch's flattened two-dimensional grid as
        ``coreDim`` and rejects more than 65,535 programs. Batching to 2,047
        ranges keeps ``zero_byte_ranges`` below that limit for every range
        count it may receive.
        """
        segments = [
            segment
            for group_id, page_ids in new_page_ids.items()
            for segment in self.arena.block_byte_segments(group_id, page_ids)
        ]
        for start in range(
            0, len(segments), _NPU_DSV4_MAX_ZERO_RANGES_PER_LAUNCH
        ):
            zero_byte_ranges(
                self.arena.buffer,
                segments[
                    start : start + _NPU_DSV4_MAX_ZERO_RANGES_PER_LAUNCH
                ],
            )
