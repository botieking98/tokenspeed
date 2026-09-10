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

"""Register CANN operator vendors bundled with TokenSpeed."""

from __future__ import annotations

import os
from collections.abc import MutableMapping, Sequence
from pathlib import Path


_VENDOR_NAMES = ("custom_transformer",)
_PACKAGE_VENDOR_PATH = (
    Path(__file__).resolve().parent / "_cann_ops_custom" / "vendors"
)
_SOURCE_VENDOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "build"
    / "ascend"
    / "cann_ops"
    / "vendors"
)


def _is_valid_vendor_path(candidate: Path) -> bool:
    required_file = candidate / "op_api" / "lib" / "libcust_opapi.so"
    return required_file.is_file()


def _register_cann_vendors(
    vendor_paths: Sequence[Path],
    environment: MutableMapping[str, str],
) -> list[Path]:
    registered_paths: list[Path] = []
    for vendor_name in _VENDOR_NAMES:
        vendor_path = next(
            (
                candidate / vendor_name
                for candidate in vendor_paths
                if _is_valid_vendor_path(candidate / vendor_name)
            ),
            None,
        )
        if vendor_path is None:
            continue

        existing_value = environment.get("ASCEND_CUSTOM_OPP_PATH", "")
        entries = existing_value.split(os.pathsep) if existing_value else []
        if str(vendor_path) not in entries:
            environment["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
                [str(vendor_path), *entries]
            )
        registered_paths.append(vendor_path)
    return registered_paths



def register_cann_vendors() -> list[Path]:
    """Prepend bundled CANN vendors to ``ASCEND_CUSTOM_OPP_PATH``.

    Returns:
        Registered vendor paths. Missing vendors are skipped, and existing
        user-provided paths are preserved.
    """
    return _register_cann_vendors(
        (_PACKAGE_VENDOR_PATH, _SOURCE_VENDOR_PATH),
        os.environ,
    )


__all__ = ["register_cann_vendors"]
