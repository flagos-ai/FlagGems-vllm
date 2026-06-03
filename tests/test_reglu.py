import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

REGLU_SHAPES = [
    (),
    (2,),
    (512, 512),
    (1, 2048),
    (2048, 2),
    (1024, 1024),
    (20, 320, 16),
    (4096, 1024),
    (2048, 2048),
    (1024, 4096),
    (512, 512, 512),
    (512, 256, 512),
]


@pytest.mark.reglu
@pytest.mark.parametrize("shape", REGLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reglu(shape, dtype):
    if len(shape) == 0:
        # reglu does not support 0-dim scalar tensors.
        return

    if shape[-1] % 2 != 0:
        # reglu requires the last dimension to be even, but got shape {shape}.
        return

    input_tensor = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)

    x1, x2 = input_tensor.chunk(2, dim=-1)
    ref_out = torch.relu(x1) * x2
    ref_out = utils.to_reference(ref_out)
    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.reglu(input_tensor)

    utils.gems_assert_close(res_out, ref_out, dtype)
