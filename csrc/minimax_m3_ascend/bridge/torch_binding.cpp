// SPDX-License-Identifier: Apache-2.0
// Independent MiniMax-only dispatcher. No vllm-plugin-FL runtime dependency.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/flaggems_m3_sparse_attention_score/flaggems_m3_sparse_attention_score_torch_adpt.h"

TORCH_LIBRARY(_flaggems_m3_ascend, ops) {
  ops.def("npu_sparse_attention_score(Tensor query, Tensor key, Tensor value, "
          "Tensor selectIdx, Tensor blockTable, Tensor? selectNumIdx=None, "
          "Tensor? qDequantScale=None, Tensor? kDequantScale=None, "
          "Tensor? vDequantScale=None, Tensor? actualSeqLengths=None, "
          "Tensor? actualSeqLengthsKv=None, str qInputLayout=\"TND\", "
          "str kvInputLayout=\"TND\", int numKeyValueHeads=0, "
          "float scaleValue=0.0, int blockSize=128, int topK=0, "
          "int innerPrecise=4) -> Tensor");
}
TORCH_LIBRARY_IMPL(_flaggems_m3_ascend, PrivateUse1, ops) {
  ops.impl("npu_sparse_attention_score", &flaggems_m3_native::npu_flaggems_m3_sparse_attention_score);
}
TORCH_LIBRARY_IMPL(_flaggems_m3_ascend, Meta, ops) {
  ops.impl("npu_sparse_attention_score", &flaggems_m3_native::meta::npu_flaggems_m3_sparse_attention_score);
}
