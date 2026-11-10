# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CANN raw bindings, grouped by hardware core rather than optimization history."""
import hashlib
from pathlib import Path

from triton.experimental.tle.raw import dialect

VECTOR_SOURCE = Path(__file__).with_name("vector.cpp")
CUBE_SOURCE = Path(__file__).with_name("cube.cpp")
# Include C++ changes in the Triton cache key; this is not a pinned version.
REVISION = hashlib.sha256(
    VECTOR_SOURCE.read_bytes() + CUBE_SOURCE.read_bytes() + Path(__file__).read_bytes()
).hexdigest()


def _bind(name, symbol, core, pipe):
    def implementation(*args, **kwargs):
        pass

    implementation.__name__ = name
    return dialect(
        name="cann",
        file=CUBE_SOURCE if core == "cube" else VECTOR_SOURCE,
        extern_func_name=symbol,
        pipeline={"core": core, "pipe": pipe},
    )(implementation)


scatter_rows = _bind("scatter_rows", "moe_scatter_rows", "vector", "PIPE_MTE3")
read_epoch = _bind("read_epoch", "moe_read_epoch", "vector", "PIPE_S")
begin = _bind("begin", "moe_begin", "vector", "PIPE_S")
wait_many = _bind("wait_many", "moe_wait_many", "vector", "PIPE_S")
signal = _bind("signal", "moe_signal", "vector", "PIPE_S")
wait_epoch = _bind("wait_epoch", "moe_wait", "vector", "PIPE_S")
dispatch_inputs = _bind("dispatch_inputs", "direct_dispatch", "vector", "PIPE_V")
plan_experts = _bind("plan_experts", "scalar_plan_dma", "vector", "PIPE_S")
schedule_ready = _bind("schedule_ready", "queue_schedule", "vector", "PIPE_S")
acquire_vector = _bind("acquire_vector", "queue_acquire_vector", "vector", "PIPE_S")
schedule_output = _bind("schedule_output", "stream_dispatch_output", "vector", "PIPE_S")
wait_output_task = _bind(
    "wait_output_task", "stream_wait_assignment", "vector", "PIPE_S"
)
cube_read_epoch = _bind("cube_read_epoch", "moe_cube_read_epoch", "cube", "PIPE_S")
wait_cube = _bind("wait_cube", "ready_wait_cube", "cube", "PIPE_S")
acquire_cube = _bind("acquire_cube", "queue_acquire_cube", "cube", "PIPE_S")

prepare_memory = _bind("prepare_memory", "moe_prepare_memory", "vector", "PIPE_MTE3")
prepare_weights16 = _bind(
    "prepare_weights16", "moe_prepare_weights16", "vector", "PIPE_MTE3"
)

advance_return_chunk = _bind(
    "advance_return_chunk", "stream_advance_chunk", "vector", "PIPE_S"
)
wait_chunk_complete = _bind(
    "wait_chunk_complete", "stream_wait_chunk", "vector", "PIPE_S"
)
