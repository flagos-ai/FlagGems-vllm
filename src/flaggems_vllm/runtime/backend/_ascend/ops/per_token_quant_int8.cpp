// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
// Rank-one memref descriptor used by the FlagTree raw-call ABI.
template <typename T> struct VectorView {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

[aicore] __attribute__((always_inline)) float
write_scale(VectorView<__ubuf__ float> *partials,
            VectorView<__ubuf__ float> *scales, int64_t row, int32_t pieces) {
  float maximum = 1e-10f;
  for (int32_t piece = 0; piece < pieces; ++piece) {
    const float partial =
        partials->aligned[partials->offset +
                          (row * pieces + piece) * partials->strides[0]];
    if (partial > maximum)
      maximum = partial;
  }
  // Both divisions use scalar FP32 to keep the reference rounding.
  scales->aligned[scales->offset + row * scales->strides[0]] = maximum / 127.0f;
  return 127.0f / maximum;
}

[aicore] __attribute__((always_inline)) void
multiply_span(__ubuf__ float *values, int64_t count, float inverse) {
  set_vector_mask(0, count);
  const int64_t repeat = (count + 63) / 64;
  if (inverse == 0.0f)
    vector_dup(values, 0.0f, repeat, 1, 1, 8, 0);
  else
    vmuls(values, values, inverse, repeat, 1, 1, 8, 8);
}

[aicore] __attribute__((always_inline)) void
narrow_span(__ubuf__ int8_t *output, __ubuf__ half *rounded,
            __ubuf__ float *values, int64_t count) {
  // The INT8 half-integer boundaries are all representable in FP16.
  // Truncating toward zero preserves those boundaries for the half-away cast.
  set_vector_mask(0, count);
  vconv_f322f16z(rounded, values, (count + 63) / 64, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
  vconv_f162s8a(output, rounded, (count + 127) / 128, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
}

extern "C" [aicore]
__attribute__((always_inline)) void _mlir_ciface_per_token_quant_tile(
    VectorView<__ubuf__ float> *partials, int32_t columns, int32_t pieces,
    VectorView<__ubuf__ int8_t> *output, VectorView<__ubuf__ float> *scales,
    VectorView<__ubuf__ float> *source, VectorView<__ubuf__ half> *work) {
  __ubuf__ float *scratch = source->aligned + source->offset;
  __ubuf__ int8_t *result = output->aligned + output->offset;
  __ubuf__ half *rounded = work->aligned + work->offset;
  const int64_t rows = scales->sizes[0];
  const int64_t active = rows * columns;
  constexpr int64_t kFp32Span = 255 * 64;
  pipe_barrier(PIPE_ALL);
  for (int64_t row = 0; row < rows; ++row) {
    const float inverse = write_scale(partials, scales, row, pieces);
    for (int64_t offset = 0; offset < columns; offset += kFp32Span) {
      const int64_t count =
          columns - offset < kFp32Span ? columns - offset : kFp32Span;
      __ubuf__ float *values = scratch + row * columns + offset;
      set_mask_count();
      multiply_span(values, count, inverse);
      set_mask_norm();
    }
  }
  constexpr int64_t kCastSpan = 8192;
  pipe_barrier(PIPE_V);
  for (int64_t offset = 0; offset < active; offset += kCastSpan) {
    const int64_t count =
        active - offset < kCastSpan ? active - offset : kCastSpan;
    set_mask_count();
    narrow_span(result + offset, rounded, scratch + offset, count);
    set_mask_norm();
  }
  pipe_barrier(PIPE_ALL);
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

extern "C" [aicore]
__attribute__((always_inline)) void _mlir_ciface_per_token_quant_half_tile(
    VectorView<__ubuf__ float> *partials, int32_t columns, int32_t pieces,
    VectorView<__ubuf__ half> *source, VectorView<__ubuf__ int8_t> *output,
    VectorView<__ubuf__ float> *scales, VectorView<__ubuf__ float> *work,
    VectorView<__ubuf__ half> *cast_work) {
  auto *input = source->aligned + source->offset;
  auto *result = output->aligned + output->offset;
  auto *scratch = work->aligned + work->offset;
  auto *rounded = cast_work->aligned + cast_work->offset;
  const int64_t rows = scales->sizes[0];
  const int64_t rows_per_span = work->sizes[0] / columns;
  pipe_barrier(PIPE_ALL);
  // Keep the full tile in FP16. Only whole rows that fit the scratch span
  // are promoted together for the FP32 multiply.
  for (int64_t first = 0; first < rows; first += rows_per_span) {
    const int64_t batch =
        rows - first < rows_per_span ? rows - first : rows_per_span;
    const int64_t count = batch * columns;
    set_mask_count();
    set_vector_mask(0, count);
    vconv_f162f32(scratch, input + first * columns, (count + 63) / 64, 1, 1, 8,
                  4);
    pipe_barrier(PIPE_V);
    for (int64_t row = 0; row < batch; ++row) {
      const float inverse = write_scale(partials, scales, first + row, pieces);
      multiply_span(scratch + row * columns, columns, inverse);
    }
    pipe_barrier(PIPE_V);
    narrow_span(result + first * columns, rounded, scratch, count);
    set_mask_norm();
  }
  pipe_barrier(PIPE_ALL);
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
