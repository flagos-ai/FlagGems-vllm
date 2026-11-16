# SPDX-License-Identifier: Apache-2.0
"""PLE conv performance. Native baseline is vLLM ple_conv, or None."""

import statistics

import pytest
import torch

from baseline import ple_conv as workloads
from flaggems_vllm import ple_conv

from . import base

# Scalars published in the H20 config. query_start_loc, has_initial_states,
# num_accepted_tokens and spec_query_len are real call arguments, but they are
# not part of that config, so recording them makes --base-data miss.
_CONFIG_SCALARS = ("mode", "dilation")


class PLEConvBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [tuple(binding["shape"]) for binding in workloads.grid()]
        self.shape_desc = "mode,rows,channels,fuse,kernel,dilation"

    def record_shapes(self, *args, **kwargs):
        visible = {key: kwargs[key] for key in _CONFIG_SCALARS if key in kwargs}
        return super().record_shapes(*args, **visible)

    def get_latency(self, op, *args, **kwargs):
        """Same protocol as the H20 collector for this in-place op.

        Device events, five warmups, 25 samples, median. Mutated tensors are
        restored outside each timed interval, and the launch is not captured
        in a CUDA graph.
        """
        saved = []
        for index in workloads.MUTATED_INPUTS:
            tensor = args[index] if isinstance(index, int) else kwargs[index]
            saved.append((tensor, tensor.clone()))

        def restore():
            for tensor, initial in saved:
                tensor.copy_(initial)

        try:
            for _ in range(5):
                restore()
                op(*args, **kwargs)
            torch.cuda.synchronize()
            samples = []
            for _ in range(25):
                restore()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                op(*args, **kwargs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return statistics.median(samples)
        finally:
            restore()
            torch.cuda.synchronize()


def conv_inputs(shape, dtype, device):
    args, kwargs = workloads.build_inputs({"shape": list(shape)}, dtype, device)
    yield (*args, kwargs)


@pytest.mark.ple_fusion
def test_ple_conv_perf():
    native = workloads.native()
    if native is None:
        print(
            "[ple_conv] no vLLM ple_conv on this image "
            f"({workloads.NATIVE_UNAVAILABLE_REASON}); measuring FlagGems only. "
            "Pass --base-data to report the scaled H20 comparison."
        )
    PLEConvBenchmark(
        input_fn=conv_inputs,
        op_name=workloads.OP_NAME,
        # Vendor-native callable. None means there is no on-machine baseline;
        # do not substitute a PyTorch conv1d.
        torch_op=native,
        gems_op=ple_conv,
        dtypes=list(workloads.DTYPES),
        is_inplace=workloads.IS_INPLACE,
    ).run()
