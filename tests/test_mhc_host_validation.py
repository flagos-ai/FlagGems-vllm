# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Public validation must run before any private mHC launch helper."""

import copy
import importlib

import pytest
import torch


class MetadataTensor:
    """Tensor metadata only: these tests cannot allocate or launch on a GPU."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self.requires_grad = False
        self.contiguous = True

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        result = copy.copy(self)
        result.shape = tuple(shape)
        return result


class ReachedPrivateLaunch(Exception):
    pass


@pytest.mark.parametrize("tokens", (64, 96, 128))
@pytest.mark.parametrize("entry", ("full", "post", "prenorm", "pre"))
def test_public_mhc_validation_precedes_private_launch(monkeypatch, tokens, entry):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    tensor = MetadataTensor
    bf16, fp32 = torch.bfloat16, torch.float32
    common = dict(
        x=tensor((tokens, 4096), bf16),
        residual=tensor((tokens, 4, 4096), bf16),
        post_layer_mix=tensor((tokens, 4, 1), fp32),
        comb_res_mix=tensor((tokens, 4, 4), fp32),
    )
    if entry == "full":
        module = importlib.import_module("flaggems_vllm.ops.mhc.mhc_fused_post_pre")
        function, helper = module.mhc_fused_post_pre, "_mhc_post_impl"
        kwargs = dict(
            common,
            fn=tensor((24, 16384), fp32),
            hc_scale=tensor((3,), fp32),
            hc_base=tensor((24,), fp32),
            norm_weight=tensor((4096,), bf16),
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )
    elif entry == "post":
        module = importlib.import_module("flaggems_vllm.ops.mhc.mhc_post")
        function, helper, kwargs = module.mhc_post, "_mhc_post_impl", common
    elif entry == "prenorm":
        module = importlib.import_module("flaggems_vllm.ops.mhc.mhc_prenorm")
        function, helper = module.mhc_prenorm_gemm, "_mhc_prenorm_gemm_impl"
        # Availability is metadata, not a request to construct a descriptor.
        monkeypatch.setattr(module, "TensorDescriptor", object())
        kwargs = dict(
            residual=tensor((tokens, 16384), bf16), fn=tensor((24, 16384), fp32)
        )
    else:
        module = importlib.import_module("flaggems_vllm.ops.mhc.mhc_pre_with_norm")
        function, helper = module.mhc_pre_with_norm, "_mhc_pre_impl"
        splits = module._SPLIT_COUNTS[tokens]
        kwargs = dict(
            partial=tensor((splits, tokens, 32), fp32),
            partial_sqrsum=tensor((splits, tokens), fp32),
            hc_scale=tensor((3,), fp32),
            hc_base=tensor((24,), fp32),
            residual=common["residual"],
            post_mix=tensor((tokens, 4), fp32),
            comb_mix=tensor((tokens, 16), fp32),
            layer_input=tensor((tokens, 4096), bf16),
            norm_weight=tensor((4096,), bf16),
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            norm_eps=1e-6,
        )

    def stop(*args, **kwargs):
        raise ReachedPrivateLaunch

    monkeypatch.setattr(module, helper, stop)
    with pytest.raises(ReachedPrivateLaunch):
        function(**kwargs)
    for name, value in kwargs.items():
        if not isinstance(value, MetadataTensor):
            continue
        for attribute, invalid in (
            ("shape", (1,)),
            ("dtype", torch.int32),
            ("device", torch.device("cuda", 1)),
            ("contiguous", False),
            ("requires_grad", True),
        ):
            bad = copy.copy(value)
            setattr(bad, attribute, invalid)
            with pytest.raises((ValueError, NotImplementedError)):
                function(**dict(kwargs, **{name: bad}))
