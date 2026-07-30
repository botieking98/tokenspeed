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

"""HCCL communication backend for Ascend NPU.

Delegates to torch.distributed with the hccl backend. Process groups are
looked up from pg_manager using the "hccl" key.
"""

import torch
import torch.distributed

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend


class HcclBackend(NcclBackend):
    """Backend using HCCL via torch.distributed.

    Identical to NcclBackend except process groups are looked up with the
    "hccl" key and PyNccl is never used.
    """

    def _get_or_create_resources(self, group: Group):
        if group in self._resources:
            return self._resources[group]

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        device_group = pg_manager.get_process_group("hccl", group)
        world_size = len(group)

        self._resources[group] = {
            "pynccl_comm": None,
            "device_group": device_group,
            "world_size": world_size,
        }
        return self._resources[group]
