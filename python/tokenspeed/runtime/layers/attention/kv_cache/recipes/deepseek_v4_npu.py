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

"""DeepSeek V4 cache geometry for Ascend 910C kernels.

The scheduler still sees the upstream cache-group contract.  This recipe only
changes the physical geometry consumed by ACLNN: 32-row kernel pages, BF16 KV,
INT8 indexer values with FP16 scales, and ACLNN's padded state-page strides.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing_extensions import override

from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    DeepseekV4CacheLayout,
    deepseek_v4_cache_layout_from_config,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4PoolOptions,
    DeepseekV4Recipe,
    v4_c4_state_window,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheFieldSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)

NPU_DSV4_BLOCK_SIZE = 32
NPU_DSV4_C4_STATE_ROWS_PER_PAGE = 2
NPU_DSV4_C128_STATE_ROWS_PER_PAGE = 8
NPU_DSV4_STATE_PAGE_STRIDE_BYTES = 32768
NPU_DSV4_INDEXER_STATE_PAGE_STRIDE_BYTES = 4160
NPU_DSV4_SHARED_STRIDE_PLANE = "npu.dsv4.stride_32768"
NPU_DSV4_INDEXER_STATE_PLANE = "npu.dsv4.indexer_state_4160"


def _nearest_power_of_two(value: float) -> int:
    if value < 1:
        raise ValueError(f"NPU DSV4 packing ratio must be >= 1, got {value}")
    return 1 << max(0, round(math.log2(value)))


@dataclass(frozen=True)
class DeepseekV4NpuPoolOptions(DeepseekV4PoolOptions):
    """The NPU cache layout bound to one compute view."""

    @override
    def create_pool(
        self,
        *,
        arena,
        layer_num: int,
        rank: int,
        field_layer_offset: int,
    ):
        """Create the Ascend ACLNN compute view over ``arena``."""
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4_npu import (
            HybridDeepseekV4NpuTokenToKVPool,
        )

        return HybridDeepseekV4NpuTokenToKVPool(
            arena,
            layout=self.layout,
            layer_num=layer_num,
            rank=rank,
            field_layer_offset=field_layer_offset,
        )


class DeepseekV4NpuRecipe(DeepseekV4Recipe):
    """DSV4 W8A8 cache plan for Ascend ACLNN kernels."""

    @cached_property
    def _layer_ratios(self) -> tuple[int, ...]:
        source = (
            self.draft_model_config.hf_config
            if self.draft_attn_config is not None
            else self.model_config.hf_config
        )
        return tuple(
            max(1, int(ratio)) for ratio in source.compress_ratios
        )

    @property
    @override
    def prefix_granularity(self) -> int:
        return NPU_DSV4_BLOCK_SIZE * max(
            1,
            max(self._layer_ratios, default=1),
        )

    @cached_property
    def _cache_layout(self) -> DeepseekV4CacheLayout:
        source = (
            self.draft_model_config.hf_config
            if self.draft_attn_config is not None
            else self.model_config.hf_config
        )
        return deepseek_v4_cache_layout_from_config(
            source,
            page_size=self.prefix_granularity,
            use_fp4_indexer_cache=False,
            layer_indices=range(self.num_target_layers + self.num_draft_layers),
        )

    def _swa_spec(self) -> CacheGroupSpec:
        return CacheGroupSpec(
            group_id=V4_SWA_KV_GROUP_ID,
            retention="sliding_window",
            rows_per_page=NPU_DSV4_BLOCK_SIZE,
            entry_stride_tokens=1,
            sliding_window_tokens=int(self.model_config.hf_config.sliding_window),
            family="history",
        )

    @staticmethod
    def _compressed_spec(ratio: int) -> CacheGroupSpec:
        return CacheGroupSpec(
            group_id=v4_compressed_kv_group_id(ratio),
            retention="full_history",
            rows_per_page=NPU_DSV4_BLOCK_SIZE,
            entry_stride_tokens=ratio,
            sliding_window_tokens=None,
            family="history",
        )

    @staticmethod
    def _state_spec(ratio: int, *, window: int) -> CacheGroupSpec:
        return CacheGroupSpec(
            group_id=v4_compressor_state_group_id(ratio),
            retention="sliding_window",
            rows_per_page=(
                NPU_DSV4_C4_STATE_ROWS_PER_PAGE
                if ratio == 4
                else NPU_DSV4_C128_STATE_ROWS_PER_PAGE
            ),
            entry_stride_tokens=1,
            sliding_window_tokens=window,
            family="history",
        )

    @staticmethod
    def _indexer_state_spec(*, window: int) -> CacheGroupSpec:
        return CacheGroupSpec(
            group_id=V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
            retention="sliding_window",
            rows_per_page=NPU_DSV4_C4_STATE_ROWS_PER_PAGE,
            entry_stride_tokens=1,
            sliding_window_tokens=window,
            family="history",
        )

    @override
    def groups(self):
        layout = self._cache_layout
        ratios = tuple(int(ratio) for ratio in layout.layer_ratio)
        if any(ratio not in (1, 4, 128) for ratio in ratios):
            raise ValueError("DeepSeek V4 NPU layer ratios must be 1, 4, or 128")

        sliding_window = int(self.model_config.hf_config.sliding_window)
        if sliding_window <= 0 or self.prefix_granularity % sliding_window:
            raise ValueError(
                "DeepSeek V4 NPU sliding_window must divide the prefix granularity"
            )

        c4_window = v4_c4_state_window(self.decode_input_tokens)
        declared: dict[str, tuple[CacheGroupSpec, tuple[CacheFieldSpec, ...]]] = {}

        def declare(
            spec: CacheGroupSpec, *fields: CacheFieldSpec
        ) -> None:
            existing = declared.get(spec.group_id)
            declared[spec.group_id] = (
                spec if existing is None else existing[0],
                (() if existing is None else existing[1]) + fields,
            )

        swa_spec = self._swa_spec()
        for layer_id, ratio in enumerate(ratios):
            declare(
                swa_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.swa",
                    NPU_DSV4_SHARED_STRIDE_PLANE,
                    (NPU_DSV4_BLOCK_SIZE, 1, layout.head_dim),
                    "bfloat16",
                    exact_page_stride=False,
                ),
            )
            if ratio == 1:
                continue

            compressed_spec = self._compressed_spec(ratio)
            state_window = c4_window if ratio == 4 else 128
            state_spec = self._state_spec(ratio, window=state_window)
            state_rows = state_spec.rows_per_page
            state_width = layout.head_dim * (4 if ratio == 4 else 2)

            declare(
                compressed_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.compressed_kv",
                    NPU_DSV4_SHARED_STRIDE_PLANE,
                    (NPU_DSV4_BLOCK_SIZE, 1, layout.head_dim),
                    "bfloat16",
                    exact_page_stride=False,
                ),
            )
            declare(
                state_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.compressor_state",
                    NPU_DSV4_SHARED_STRIDE_PLANE,
                    (state_rows, 1, state_width),
                    "float32",
                    exact_page_stride=False,
                    page_stride_alignment_bytes=NPU_DSV4_STATE_PAGE_STRIDE_BYTES,
                ),
            )
            if ratio != 4:
                continue

            indexer_state_spec = self._indexer_state_spec(window=c4_window)
            declare(
                compressed_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.indexer_kv",
                    NPU_DSV4_SHARED_STRIDE_PLANE,
                    (NPU_DSV4_BLOCK_SIZE, 1, layout.index_head_dim),
                    "int8",
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.indexer_scale",
                    NPU_DSV4_SHARED_STRIDE_PLANE,
                    (NPU_DSV4_BLOCK_SIZE, 1, 1),
                    "float16",
                    exact_page_stride=False,
                ),
            )
            declare(
                indexer_state_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.indexer_state",
                    NPU_DSV4_INDEXER_STATE_PLANE,
                    (
                        NPU_DSV4_C4_STATE_ROWS_PER_PAGE,
                        1,
                        4 * layout.index_head_dim,
                    ),
                    "float32",
                    exact_page_stride=False,
                    page_stride_alignment_bytes=(
                        NPU_DSV4_INDEXER_STATE_PAGE_STRIDE_BYTES
                    ),
                ),
            )

        return tuple(declared.values())

    @override
    def packing(self, groups):
        """Balance groups over the two ACLNN stride-class planes.

        The upstream byte-ratio policy assumes each group can own per-layer
        planes. ACLNN state pages instead require one common 32768-byte stride
        class and one 4160-byte indexer-state class, so fields share planes and
        each group needs enough cache blocks to occupy a comparable slice of
        the common plane.
        """
        raw_bytes = {
            spec.group_id: sum(field.payload_bytes for field in fields)
            for spec, fields in groups
        }
        largest_bytes = max(raw_bytes.values())
        return {
            group_id: _nearest_power_of_two(largest_bytes / group_bytes)
            for group_id, group_bytes in raw_bytes.items()
        }

    @override
    def pool_options(self):
        return DeepseekV4NpuPoolOptions(layout=self._cache_layout)
