from __future__ import annotations

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

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build" / "ascend"
LIB_SOURCE = BUILD / "tokenspeed-extension" / "libtokenspeed_ascend_c.so"
CANN_OPS_SCRIPT = ROOT / "build_cann_ops.sh"
CANN_OPS_SOURCE = BUILD / "cann_ops" / "vendors"
VLLM_ASCEND_EXTENSION_SOURCE = BUILD / "vllm-ascend-extension"


def _can_build_aclnn() -> bool:
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return True


class BuildPyWithAscend(build_py):
    def run(self) -> None:
        super().run()
        if not _can_build_aclnn():
            return

        soc_version = os.environ.get("SOC_VERSION", "ascend910_93")
        jobs = os.environ.get("MAX_JOBS")
        if jobs is None:
            jobs = str(min(16, os.cpu_count() or 1))
        subprocess.check_call(
            [str(CANN_OPS_SCRIPT), soc_version, jobs, sys.executable]
        )

        if self.build_lib:
            package_root = Path(self.build_lib) / "tokenspeed_kernel_npu"
            library_destination = package_root / "lib"
            library_destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LIB_SOURCE, library_destination / LIB_SOURCE.name)
            for extension_source in VLLM_ASCEND_EXTENSION_SOURCE.glob(
                "vllm_ascend_C*.so"
            ):
                shutil.copy2(extension_source, package_root / extension_source.name)
            kernel_source = VLLM_ASCEND_EXTENSION_SOURCE / "libvllm_ascend_kernels.so"
            shutil.copy2(kernel_source, package_root / kernel_source.name)
            vendor_destination = (
                Path(self.build_lib)
                / "tokenspeed_kernel_npu"
                / "_cann_ops_custom"
                / "vendors"
            )
            vendor_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                CANN_OPS_SOURCE,
                vendor_destination,
                dirs_exist_ok=True,
        )


class BDistWheelWithNpuBinaries(bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        return python_tag, python_tag, platform_tag


setup(
    cmdclass={
        "bdist_wheel": BDistWheelWithNpuBinaries,
        "build_py": BuildPyWithAscend,
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "tokenspeed_kernel_npu": [
            "*.so",
            "lib/*.so",
            "_cann_ops_custom/vendors/**/*",
        ]
    },
)
