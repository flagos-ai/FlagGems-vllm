# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import math

import torch
import torch.nn.functional as F

OP_NAME = "chunk_kda"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False

LOWER_BOUND = -5.0
DEFAULT_H = 96
DEFAULT_D = 128

FIXED_CASES = [
    [8192],
]
VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]
FLASHKDA_CASES = FIXED_CASES + VARLEN_CASES


# chunk_kda moved across vLLM releases; probe each known location in turn.
_KDA_MODULES = [
    "vllm.third_party.flash_linear_attention.ops.kda",  # 0.26+
    "vllm.model_executor.layers.fla.ops.kda",  # 0.25
]


def _kda_module():
    for module_name in _KDA_MODULES:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _supports_lower_bound(op):
    """The 0.27+ fused gate supports lower_bound; 0.25/0.26 only softplus gate."""
    try:
        return "lower_bound" in inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False


def _chunk_kda_entry():
    mod = _kda_module()
    if mod is None:
        return None, None
    plain = getattr(mod, "chunk_kda", None)
    gate = getattr(mod, "fused_kda_gate", None)
    return (
        plain if callable(plain) else None,
        gate if callable(gate) else None,
    )


def native():
    plain, gate = _chunk_kda_entry()
    if plain is None or gate is None:
        return None

    def chunk_kda(q, k, v, raw_g, beta, A_log, dt_bias, **kwargs):
        head_k_dim = dt_bias.shape[-1]
        g_flat = raw_g.reshape(raw_g.shape[0], raw_g.shape[1], -1)
        gate_kwargs = {"g_bias": dt_bias}
        lower_bound = kwargs.get("lower_bound")
        if lower_bound is not None:
            gate_kwargs["lower_bound"] = lower_bound
        g_activated = gate(g_flat, A_log, head_k_dim, **gate_kwargs)
        return plain(
            q,
            k,
            v,
            g_activated,
            beta,
            scale=kwargs.get("scale"),
            initial_state=kwargs.get("initial_state"),
            output_final_state=kwargs.get("output_final_state", False),
            use_qk_l2norm_in_kernel=kwargs.get("use_qk_l2norm_in_kernel", False),
            cu_seqlens=kwargs.get("cu_seqlens"),
        )

    return chunk_kda


def grid():
    return [
        {"shape": list(seq_lens) + [DEFAULT_H, DEFAULT_D]}
        for seq_lens in FLASHKDA_CASES
    ]


def _unpack(binding):
    shape = list(binding["shape"])
    *seq_lens, H, D = shape
    return seq_lens, H, D


def build_inputs(binding, dtype, device):
    seq_lens, H, D = _unpack(binding)
    T_total = sum(seq_lens)
    N = len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = F.normalize(
        torch.randn((1, T_total, H, D), dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    ).to(dtype)
    k = F.normalize(
        torch.randn((1, T_total, H, D), dtype=torch.float32, device=device),
        p=2.0,
        dim=-1,
    ).to(dtype)
    v = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
    raw_g = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
    beta = torch.randn((1, T_total, H), dtype=dtype, device=device).float().sigmoid()
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
    initial_state = torch.arange(
        N * H * D * D, dtype=torch.float32, device=device
    ).reshape(N, H, D, D)

    cu_seqlens = None
    if N > 1:
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.long,
            device=device,
        )

    args = (q, k, v, raw_g, beta, A_log, dt_bias)
    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": cu_seqlens,
    }
    _, gate = _chunk_kda_entry()
    if gate is not None and _supports_lower_bound(gate):
        kwargs["lower_bound"] = LOWER_BOUND
    return args, kwargs


def key_shape(binding):
    """Return the JSON primary-key shape.

    The first tensor q has shape [1, T_total, H, D], but all three cases share
    the same total length T_total == 8192, so fixed/varlen cannot be told apart
    that way. Append the number of sequences N as a compound key, together with
    config's initial_state/cu_seqlens (which differ by N) matched via consts.py's
    signature fallback.
    """
    seq_lens, H, D = _unpack(binding)
    return [1, sum(seq_lens), H, D, len(seq_lens)]


def config(binding, dtype):
    """Real input/output shape description written into JSON's config field.

    q/k/v/raw g use the collected dtype; beta is FP32 (already sigmoided at the
    model layer); A_log/dt_bias/initial_state are FP32; cu_seqlens is int64
    (varlen only).
    """
    seq_lens, H, D = _unpack(binding)
    T_total = sum(seq_lens)
    N = len(seq_lens)
    dt = str(dtype)

    inputs = {
        "q": {"shape": [1, T_total, H, D], "dtype": dt},
        "k": {"shape": [1, T_total, H, D], "dtype": dt},
        "v": {"shape": [1, T_total, H, D], "dtype": dt},
        "g": {
            "shape": [1, T_total, H, D],
            "dtype": dt,
            "note": (
                "raw gate, unactivated; activation runs in the standalone "
                "fused_kda_gate kernel before chunk_kda (cumsum inside)"
            ),
        },
        "beta": {
            "shape": [1, T_total, H],
            "dtype": "torch.float32",
            "note": "already sigmoided at the model layer",
        },
        "A_log": {"shape": [H], "dtype": "torch.float32"},
        "dt_bias": {"shape": [H, D], "dtype": "torch.float32"},
        "initial_state": {"shape": [N, H, D, D], "dtype": "torch.float32"},
    }
    if N > 1:
        inputs["cu_seqlens"] = {"shape": [N + 1], "dtype": "torch.int64"}

    return {
        "inputs": inputs,
        "outputs": {
            "out": {"shape": [1, T_total, H, D], "dtype": dt},
            "final_state": {"shape": [N, H, D, D], "dtype": "torch.float32"},
        },
        "dims": {
            "seq_lens": seq_lens,
            "T_total": T_total,
            "H": H,
            "D": D,
            "N": N,
        },
    }
