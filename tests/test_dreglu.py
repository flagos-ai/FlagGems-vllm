import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

DREGLU_SHAPES = [
    (),
    (1,),
    (512, 512),
    (1, 2048),
    (2048, 1),
    (1024, 1024),
    (20, 320, 15),
    (4096, 1024),
    (2048, 2048),
    (1024, 4096),
    (512, 512, 512),
    (512, 256, 512),
]


@pytest.mark.dreglu
@pytest.mark.parametrize("shape", DREGLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dreglu(shape, dtype):
    if len(shape) == 0:
        # dreglu does not support 0-dim scalar tensors.
        return

    if shape[-1] % 2 != 0:
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    input_tensor = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flaggems_vllm.device
    )

    x1, x2 = input_tensor.chunk(2, dim=-1)
    grad_x1 = grad_output * (x1 > 0).to(x1.dtype) * x2
    grad_x2 = grad_output * torch.relu(x1)
    ref_out = torch.cat((grad_x1, grad_x2), dim=-1)
    ref_out = utils.to_reference(ref_out)
    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.dreglu(grad_output, input_tensor, None)
    utils.gems_assert_close(res_out, ref_out, dtype)
