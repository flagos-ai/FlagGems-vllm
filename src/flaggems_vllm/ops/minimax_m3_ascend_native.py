# SPDX-License-Identifier: Apache-2.0
"""Load operator-owned M3 CANN payload; never import the framework plugin."""

import os
from pathlib import Path
import threading

import torch

_lock = threading.Lock()
_loaded = False


def load_minimax_m3_ascend_native():
    global _loaded
    with _lock:
        if _loaded:
            return torch.ops._flaggems_m3_ascend.npu_sparse_attention_score
        root = Path(__file__).resolve().parents[1] / "_minimax_m3_ascend"
        bridge = root / "_flaggems_m3_ascend.so"
        vendors = sorted((root / "opp" / "vendors").glob("flaggems_m3*"))
        if (
            not bridge.is_file()
            or len(vendors) != 1
            or not (vendors[0] / "op_api" / "lib").is_dir()
        ):
            raise NotImplementedError(
                "Build the operator-owned M3 CANN payload using "
                "tools/build_minimax_m3_ascend.sh for the target SOC"
            )
        old = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        entries = old.split(os.pathsep) if old else []
        path = str(vendors[0])
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
            [path] + [x for x in entries if x != path]
        )
        try:
            torch.ops.load_library(str(bridge))
            native = torch.ops._flaggems_m3_ascend.npu_sparse_attention_score
        except BaseException:
            if old is None:
                os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
            else:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = old
            raise
        _loaded = True
        return native
