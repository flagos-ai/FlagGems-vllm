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

import ctypes
import importlib
import warnings

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.mhc.hc_head_fused_kernel import hc_head_fused_kernel

try:
    from vllm.model_executor.layers.mhc import (
        _hc_head_fused_kernel as _vllm_hc_head_fused,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
from flaggems_vllm.ops.mhc.hc_split_sinkhorn import (
    hc_split_sinkhorn,
    mhc_split_sinkhorn_torch_ref,
)
from flaggems_vllm.ops.mhc.mhc_bwd import mhc_bwd, mhc_bwd_ref, sinkhorn_forward
from flaggems_vllm.ops.mhc.mhc_post import mhc_post
from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre

from . import base

VENDOR = flaggems_vllm.vendor_name

# Vendor-native mhc_post, one entry per vendor (nvidia mirrors
# baseline/mhc_post.py::native()). A vendor missing here has no native kernel and
# the benchmark reports the FlagGems latency only.
_NATIVE_MHC_POST = {
    "nvidia": ("vllm.model_executor.kernels.mhc", "mhc_post_tilelang"),
    "hygon": ("aiter", "mhc_post"),
    "thead": ("vllm.model_executor.layers.mhc", "mhc_post"),
    "mthreads": ("vllm_musa.deepseek_v4_mhc", "mhc_post_musa"),
    "iluvatar": ("ixformer.inference.functions.deepseek.mhc", "ds4_mhc_post"),
    "ascend": ("torch_npu", "npu_mhc_post"),
}


def _get_baseline_mhc_post_op():
    """Vendor-native mhc_post, or None when this vendor ships none."""
    entry = _NATIVE_MHC_POST.get(VENDOR)
    if entry is None:
        return None
    module_name, attr = entry
    try:
        native = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        warnings.warn(
            f"mhc_post benchmark: {VENDOR} declares a native baseline at "
            f"{module_name}.{attr} but it could not be loaded ({exc!r}); the "
            "case runs without a baseline, so latency_base/speedup stay empty.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if VENDOR == "ascend":
        # npu_mhc_post(residual, h_res, h_out, h_post): mix first, (N, 4) post.
        return lambda x, r, p, m: native(r, m, x, p.squeeze(-1))
    return native


# Ascend: CANN ships aclnnMhcPreSinkhorn, which torch_npu does not bind (and
# npu_mhc_pre has no kernel on 910B4).  Only argument marshalling happens
# below, so the measured time is the vendor kernel's.

_ASCEND_SYMS = None
_ASCEND_CACHE = {}
_ASCEND_REQUIRED = (
    "aclnnMhcPreSinkhornGetWorkspaceSize",
    "aclnnMhcPreSinkhorn",
    "aclCreateTensor",
)


def _ascend_handles():
    """Handles to resolve the CANN symbols from, without guessing any directory."""
    yield ctypes.CDLL(None)
    for name in ("libopapi_transformer.so", "libnnopbase.so", "libascendcl.so"):
        try:
            yield ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _ascend_load():
    """Resolve the three CANN entry points, or raise ImportError."""
    global _ASCEND_SYMS
    if _ASCEND_SYMS is not None:
        return _ASCEND_SYMS
    syms = {}
    for handle in _ascend_handles():
        for name in _ASCEND_REQUIRED:
            if name not in syms:
                try:
                    syms[name] = getattr(handle, name)
                except AttributeError:
                    pass
        if len(syms) == len(_ASCEND_REQUIRED):
            break
    missing = sorted(set(_ASCEND_REQUIRED) - set(syms))
    if missing:
        raise ImportError(f"CANN symbols not found: {missing}")

    vp, i64, u64, f64 = (
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_double,
    )
    syms["aclCreateTensor"].restype = vp
    syms["aclCreateTensor"].argtypes = [
        ctypes.POINTER(i64),
        u64,
        ctypes.c_int,
        ctypes.POINTER(i64),
        i64,
        ctypes.c_int,
        ctypes.POINTER(i64),
        u64,
        vp,
    ]
    # x, fn, hc_scale, hc_base, hc, numIters, eps, eps, outFlag, 8 outputs
    syms["aclnnMhcPreSinkhornGetWorkspaceSize"].argtypes = (
        [vp] * 4
        + [i64, i64, f64, f64, i64]
        + [vp] * 8
        + [ctypes.POINTER(u64), ctypes.POINTER(vp)]
    )
    syms["aclnnMhcPreSinkhornGetWorkspaceSize"].restype = ctypes.c_int
    syms["aclnnMhcPreSinkhorn"].argtypes = [vp, u64, vp, vp]
    syms["aclnnMhcPreSinkhorn"].restype = ctypes.c_int
    _ASCEND_SYMS = syms
    return syms


def _ascend_ptr(tensor):
    create = _ascend_load()["aclCreateTensor"]
    shape = list(tensor.shape)
    dims = (ctypes.c_int64 * len(shape))(*shape)
    strides = (ctypes.c_int64 * len(shape))(*tensor.stride())
    dtype = 27 if tensor.dtype == torch.bfloat16 else 0  # ACL_BF16 / ACL_FLOAT
    ptr = create(
        dims,
        len(shape),
        dtype,
        strides,
        0,
        2,
        dims,
        len(shape),
        ctypes.c_void_p(tensor.data_ptr()),
    )
    return ctypes.c_void_p(ptr)


def _ascend_state(residual, fn, hc_scale, hc_base, hc_pre_eps, rms_eps):
    """Descriptors and scratch for one shape: everything except the executor."""
    n, hc, h = residual.shape
    dev = residual.device
    # Always 20 iterations, two normalisations each: 40 rows.
    f32, bf16, it = torch.float32, torch.bfloat16, 40
    shapes = [
        (bf16, (1, n, h)),
        (f32, (1, n, hc)),
        (f32, (1, n, hc * hc)),
        (f32, (1, n, hc)),
        (f32, (1, n, hc * hc + 2 * hc)),
        (f32, (1, n, 1)),
        (f32, (it, 1, n, hc)),
        (f32, (it, 1, n, hc, hc)),
    ]
    tensors = [
        residual.reshape(1, n, hc, h).contiguous(),
        fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
    ]
    tensors += [torch.zeros(*shape, dtype=dt, device=dev) for dt, shape in shapes]
    return {
        "n": n,
        "hc": hc,
        "h": h,
        "ptrs": [_ascend_ptr(t) for t in tensors],
        "out": tensors[4:],
        "eps": (hc_pre_eps, rms_eps),
        "ws": None,
        "stream": ctypes.c_void_p(torch.npu.current_stream().npu_stream),
    }


def _ascend_native_mhc_pre():
    """Ascend native mhc_pre baseline.

    Binds CANN's aclnnMhcPreSinkhorn through ctypes because torch_npu does not
    expose it. The operator takes only numIters=20 (sinkhorn_repeat is ignored)
    and has no hc_post_mult_value.
    """
    _ascend_load()

    def npu_mhc_pre_ascend_native(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    ):
        if hc_post_mult_value != 1.0:
            # Not a vendor parameter; scaling here would time our own arithmetic.
            raise ValueError("aclnnMhcPreSinkhorn has no hc_post_mult_value")
        # Always 20 iterations; sinkhorn_repeat is accepted and ignored.
        state = _ASCEND_CACHE.get(tuple(residual.shape))
        if state is None:
            state = _ascend_state(residual, fn, hc_scale, hc_base, hc_pre_eps, rms_eps)
            _ASCEND_CACHE[tuple(residual.shape)] = state

        syms = _ascend_load()
        get_ws = syms["aclnnMhcPreSinkhornGetWorkspaceSize"]
        run = syms["aclnnMhcPreSinkhorn"]
        ws, ex = ctypes.c_uint64(0), ctypes.c_void_p(None)
        rc = get_ws(
            *state["ptrs"][:4],
            state["hc"],
            20,  # numIters: the operator's only accepted value
            *state["eps"],
            0,
            *state["ptrs"][4:],
            ctypes.byref(ws),
            ctypes.byref(ex),
        )
        if rc:
            raise RuntimeError(f"aclnnMhcPreSinkhornGetWorkspaceSize rc={rc}")
        if state["ws"] is None or state["ws"].numel() < ws.value:
            dev = residual.device
            state["ws"] = torch.empty(max(ws.value, 1), dtype=torch.uint8, device=dev)
        rc = run(ctypes.c_void_p(state["ws"].data_ptr()), ws.value, ex, state["stream"])
        if rc:
            raise RuntimeError(f"aclnnMhcPreSinkhorn rc={rc}")

        n, hc, h = state["n"], state["hc"], state["h"]
        layer_input, _, hres, *_ = state["out"]
        return (
            torch.zeros(n, hc, 1, dtype=torch.float32, device=residual.device),
            hres.reshape(n, hc, hc),
            layer_input.reshape(n, h),
        )

    return npu_mhc_pre_ascend_native


# Every entry is a (module, attribute) pair; ascend is resolved by a special case
# in _get_baseline_mhc_pre_op because torch_npu does not bind aclnnMhcPreSinkhorn.
_NATIVE_MHC_PRE = {
    "nvidia": ("vllm._custom_ops", "mhc_pre"),
    "hygon": ("aiter", "mhc_pre"),
    "thead": ("vllm.model_executor.layers.mhc", "mhc_pre"),
    "mthreads": ("mate", "mhc_pre"),
    "iluvatar": ("ixformer.inference.functions.deepseek.mhc", "ds4_mhc_pre_v3"),
}


def _get_baseline_mhc_pre_op():
    """Vendor-native mhc_pre, or None when this vendor ships none."""
    if VENDOR == "ascend":
        try:
            return _ascend_native_mhc_pre()
        except (ImportError, OSError) as exc:
            warnings.warn(
                "mhc_pre benchmark: ascend declares a native baseline at "
                f"aclnnMhcPreSinkhorn but it could not be loaded ({exc!r}); the "
                "case runs without a baseline, so latency_base/speedup stay empty.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
    entry = _NATIVE_MHC_PRE.get(VENDOR)
    if entry is None:
        return None
    module_name, attr = entry
    try:
        native = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        warnings.warn(
            f"mhc_pre benchmark: {VENDOR} declares a native baseline at "
            f"{module_name}.{attr} but it could not be loaded ({exc!r}); the "
            "case runs without a baseline, so latency_base/speedup stay empty.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if module_name == "mate":
        # mate's trailing parameters are keyword-only and named differently.
        return lambda x, f, s, b, rms, pre, sink, mult, rep: native(
            x,
            f,
            s,
            b,
            rms_eps=rms,
            mhc_pre_eps=pre,
            mhc_sinkhorn_eps=sink,
            mhc_post_mult_value=mult,
            sinkhorn_repeat=rep,
        )
    return native


class MHCPostBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4096, 1280),
            (4096, 2560),
            (4096, 7168),
        ]

    def get_input_iter(self, dtype):
        for n, h in self.shapes:
            hc_mult = 4
            x = torch.randn((n, h), dtype=torch.bfloat16, device=self.device)
            residual = torch.randn(
                (n, hc_mult, h), dtype=torch.bfloat16, device=self.device
            )
            post_layer_mix = torch.randn(
                (n, hc_mult, 1), dtype=torch.float32, device=self.device
            )
            comb_res_mix = torch.randn(
                (n, hc_mult, hc_mult), dtype=torch.float32, device=self.device
            )
            yield x, residual, post_layer_mix, comb_res_mix


@pytest.mark.mhc_post
def test_mhc_post():
    bench = MHCPostBenchmark(
        op_name="mhc_post",
        torch_op=_get_baseline_mhc_post_op(),
        gems_op=mhc_post,
        dtypes=[torch.bfloat16],
    )
    bench.run()


class MHCPreBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def __init__(self, *args, hc_mult=4, sinkhorn_repeat=10, **kwargs):
        self.hc_mult = hc_mult
        self.sinkhorn_repeat = sinkhorn_repeat
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (512, 1280),
            (512, 2560),
            (512, 4096),
            (1024, 1280),
            (1024, 2560),
            (1024, 4096),
            (2048, 1280),
            (2048, 2560),
            (2048, 4096),
            (8192, 1280),
            (8192, 2560),
            (8192, 4096),
        ]

    def get_input_iter(self, dtype):
        for n, hidden_size in self.shapes:
            hc_mult = self.hc_mult
            hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
            device = self.device

            torch.manual_seed(42)
            residual = (
                torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
                .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
                .bfloat16()
            )
            fn = (
                torch.randn(
                    (hc_mult3, hc_mult, hidden_size),
                    dtype=torch.float,
                    device=device,
                )
                * 1e-4
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
            ).flatten(1, 2)
            hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
            hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

            yield (
                residual,
                fn,
                hc_scale,
                hc_base,
                1e-6,  # rms_eps
                1e-6,  # hc_pre_eps
                1e-6,  # hc_sinkhorn_eps
                1.0,  # hc_post_mult_value
                self.sinkhorn_repeat,
            )


@pytest.mark.mhc_pre
def test_mhc_pre():
    bench = MHCPreBenchmark(
        op_name="mhc_pre",
        torch_op=_get_baseline_mhc_pre_op(),
        gems_op=mhc_pre,
        dtypes=[torch.bfloat16],
    )
    bench.run()


class MHCSplitSinkhornBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch, seqlen"

    def __init__(self, *args, hc_mult=4, sinkhorn_iters=20, eps=1e-6, **kwargs):
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 16, 4),
            (32, 64, 4),
            (128, 128, 4),
            (256, 256, 4),
            (8, 16, 2),
            (32, 64, 2),
            (128, 128, 2),
            (256, 256, 2),
        ]

    def get_input_iter(self, dtype):
        for batch, seqlen, hc_mult in self.shapes:
            mix_hc = (2 + hc_mult) * hc_mult
            device = self.device

            torch.manual_seed(42)
            mixes = torch.randn(
                (batch, seqlen, mix_hc), dtype=torch.float32, device=device
            )
            hc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
            hc_base = torch.randn((mix_hc,), dtype=torch.float32, device=device) * 0.1

            yield mixes, hc_scale, hc_base, hc_mult, self.sinkhorn_iters, self.eps


@pytest.mark.hc_split_sinkhorn_forward
def test_hc_split_sinkhorn_forward():
    bench = MHCSplitSinkhornBenchmark(
        op_name="hc_split_sinkhorn_forward",
        torch_op=mhc_split_sinkhorn_torch_ref,
        dtypes=[torch.float32],
    )
    bench.set_gems(hc_split_sinkhorn)
    bench.run()


class MHCBwdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "seqlen, n_stream"

    def __init__(self, *args, n_stream=4, sinkhorn_iters=20, **kwargs):
        self.n_stream = n_stream
        self.sinkhorn_iters = sinkhorn_iters
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (256, 4),
            (1024, 4),
            (4096, 4),
            (8192, 4),
            (16384, 4),
            (65536, 4),
        ]

    def get_input_iter(self, dtype):
        for seqlen, n_stream in self.shapes:
            device = self.device
            torch.manual_seed(42)

            dist = torch.distributions.uniform.Uniform(0.0, 4.0)
            M = dist.sample((seqlen, n_stream, n_stream)).to(device)
            R, _P = sinkhorn_forward(M, iters=self.sinkhorn_iters)
            dR = torch.randn_like(R)

            yield R.detach(), dR


@pytest.mark.mhc_bwd
def test_mhc_bwd():
    bench = MHCBwdBenchmark(
        op_name="mhc_bwd",
        torch_op=mhc_bwd_ref,
        gems_op=mhc_bwd,
        dtypes=[torch.float32],
    )
    bench.run()


class HCHeadFusedBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1280, 4),
            (4, 2560, 4),
            (16, 4096, 4),
            (64, 7168, 4),
            (256, 1280, 2),
            (256, 1280, 4),
            (512, 1280, 2),
            (512, 1280, 4),
            (512, 2560, 2),
            (512, 2560, 4),
            (1024, 2560, 2),
            (1024, 2560, 4),
            (2048, 4096, 2),
            (2048, 4096, 4),
            (4096, 4096, 2),
            (4096, 4096, 4),
        ]

    def get_input_iter(self, dtype):
        for n, hidden_size, hc_mult in self.shapes:
            device = self.device
            torch.manual_seed(42)
            hs_flat = torch.randn((n, hc_mult, hidden_size), dtype=dtype, device=device)
            fn = torch.randn(
                (hc_mult, hc_mult * hidden_size),
                dtype=torch.float32,
                device=device,
            )
            hc_scale = torch.randn((1,), dtype=torch.float32, device=device) * 0.1
            hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device) * 0.1
            out = torch.empty((n, hidden_size), dtype=dtype, device=device)

            yield hs_flat, fn, hc_scale, hc_base, out, hidden_size, 1e-6, 1e-6, hc_mult


def _hc_head_fused_kernel_ref(
    hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
):
    _vllm_hc_head_fused(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out


@pytest.mark.hc_head_fused_kernel
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not available")
def test_hc_head_fused_kernel():
    bench = HCHeadFusedBenchmark(
        op_name="hc_head_fused_kernel",
        torch_op=_hc_head_fused_kernel_ref,
        gems_op=hc_head_fused_kernel,
        dtypes=[torch.bfloat16],
    )
    bench.run()
