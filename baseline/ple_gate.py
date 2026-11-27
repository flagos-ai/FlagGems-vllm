# SPDX-License-Identifier: Apache-2.0
import torch

OP_NAME = "ple_gate"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False


def native():
    try:
        from vllm._custom_ops import ple_gate

        return ple_gate
    except ImportError:
        return None


def grid():
    return [(T, 4, 2560) for T in [128, 512, 2048, 4096, 8192]]


def build_inputs(binding, dtype, device):
    T, HC, H = binding
    key = torch.randn(T, HC * H, dtype=dtype, device=device)
    hidden = torch.randn(T, HC * H, dtype=dtype, device=device)
    value = torch.randn(T, H, dtype=dtype, device=device)
    nk = torch.randn(HC * H, dtype=dtype, device=device)
    nq = torch.randn(HC * H, dtype=dtype, device=device)
    ncw = torch.randn(HC * H, dtype=dtype, device=device)
    return key, value, hidden, nk, nq, ncw


def key_shape(binding):
    return binding


def config(binding, dtype):
    return {"T": binding[0], "HC": binding[1], "H": binding[2], "dtype": str(dtype)}
