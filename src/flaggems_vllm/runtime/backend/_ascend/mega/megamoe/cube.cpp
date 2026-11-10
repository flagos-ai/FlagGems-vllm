// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

__aicore__ __attribute__((always_inline)) void cube_invalidate(__gm__ int64_t *flag) {
  __asm__ __volatile__("");
  dcci(reinterpret_cast<__gm__ uint64_t *>(flag), cache_line_t::SINGLE_CACHE_LINE,
       dcci_dst_t::CACHELINE_OUT);
  __asm__ __volatile__("");
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_moe_cube_read_epoch(memref_t<__gm__ int64_t, 1> *epoch,
                                memref_t<__gm__ int64_t, 1> *control) {
  auto src = epoch->aligned + epoch->offset;
  auto dst = control->aligned + control->offset;
  cube_invalidate(src);
  *reinterpret_cast<volatile __gm__ int64_t *>(dst) =
      *reinterpret_cast<volatile __gm__ int64_t *>(src);
  cube_invalidate(dst);
}

template <typename T>
__aicore__ __attribute__((always_inline)) void ready_cube_invalidate(__gm__ T *ptr) {
  dcci(reinterpret_cast<__gm__ uint64_t *>(ptr), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
}

__aicore__ __attribute__((always_inline)) int64_t ready_cube_await_epoch(__gm__ int64_t *ptr, int64_t epoch) {
  while (true) {
    ready_cube_invalidate(ptr);
    int64_t observed = *reinterpret_cast<volatile __gm__ int64_t *>(ptr);
    if (observed == epoch) return observed;
  }
}

extern "C" __aicore__ __attribute__((always_inline)) void _mlir_ciface_ready_wait_cube(
    memref_t<__gm__ int64_t, 1> *flag, int64_t epoch) {
  pipe_barrier(PIPE_ALL);
  ready_cube_await_epoch(flag->aligned + flag->offset, epoch);
  pipe_barrier(PIPE_ALL);
}

extern "C" __aicore__ __attribute__((always_inline)) void _mlir_ciface_queue_acquire_cube(
    memref_t<__gm__ int64_t,1> *states, int64_t epoch, int32_t job,
    memref_t<__gm__ int64_t,1> *control) {
  pipe_barrier(PIPE_ALL);
  auto flag=states->aligned+states->offset+job*16;
  while (true) {
    dcci(reinterpret_cast<__gm__ uint64_t *>(flag), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    if (*reinterpret_cast<volatile __gm__ int64_t *>(flag)==epoch) break;
  }
  auto out=control->aligned+control->offset;
  out[0]=flag[5] ? -1 : flag[2];
  out[1]=flag[1];
  out[2]=flag[3];
  out[3]=flag[4];
  dcci(reinterpret_cast<__gm__ uint64_t *>(out), cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
  pipe_barrier(PIPE_ALL);
}
