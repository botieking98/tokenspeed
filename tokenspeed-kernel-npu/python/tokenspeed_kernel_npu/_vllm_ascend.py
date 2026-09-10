# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Load the vllm-ascend-derived custom operator extension."""

from __future__ import annotations

from pathlib import Path

import torch

from tokenspeed_kernel_npu._ascend import register_cann_vendors


_LOADED = False


def _extension_paths() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parent
    build_root = Path(__file__).resolve().parents[2] / "build" / "ascend"
    paths: list[Path] = []
    paths.extend(package_root.glob("vllm_ascend_C*.so"))
    paths.extend((build_root / "vllm-ascend-extension").glob("vllm_ascend_C*.so"))
    return tuple(paths)


def ensure_vllm_ascend_ops() -> None:
    """Load the bundled extension that registers the ``_C_ascend`` namespace."""
    global _LOADED
    if _LOADED:
        return

    registered_vendors = register_cann_vendors()
    if not any(
        vendor_path.name == "custom_transformer"
        for vendor_path in registered_vendors
    ):
        raise FileNotFoundError(
            "TokenSpeed custom_transformer CANN vendor not found. Build "
            "tokenspeed-kernel-npu with CANN and torch_npu installed, or "
            "install a wheel containing "
            "tokenspeed_kernel_npu/_cann_ops_custom/vendors/custom_transformer."
        )
    extension_path = next(
        (candidate for candidate in _extension_paths() if candidate.is_file()),
        None,
    )
    if extension_path is None:
        paths = "\n".join(str(candidate) for candidate in _extension_paths())
        raise FileNotFoundError(
            "TokenSpeed Ascend C extension not found. Expected one of:\n"
            f"{paths}"
        )

    torch.ops.load_library(str(extension_path))
    _LOADED = True


__all__ = ["ensure_vllm_ascend_ops"]
