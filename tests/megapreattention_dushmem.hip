// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Requires vendor RDC/device-library linkage.
// Test-only vendor-runtime bridge. No custom GPU compute/synchronization kernel.
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>
#include <cstring>

extern "C" size_t mpa_uid_size() { return sizeof(dushmemx_uniqueid_t); }
extern "C" int mpa_uid(void *bytes) {
  dushmemx_uniqueid_t uid = DUSHMEMX_UNIQUEID_INITIALIZER;
  int status = dushmemx_get_uniqueid(&uid);
  if (!status) std::memcpy(bytes, &uid, sizeof(uid));
  return status;
}
extern "C" int mpa_init(int rank, int world, int device, const void *bytes) {
  auto error = hipSetDevice(device);
  if (error != hipSuccess) return static_cast<int>(error);
  dushmemx_uniqueid_t uid = DUSHMEMX_UNIQUEID_INITIALIZER;
  std::memcpy(&uid, bytes, sizeof(uid));
  dushmemx_init_attr_t attr{};
  int status = dushmemx_set_attr_uniqueid_args(rank, world, &uid, &attr);
  if (status) return status;
  return dushmemx_init_attr(DUSHMEMX_INIT_WITH_UNIQUEID, &attr);
}
extern "C" int mpa_rank() { return dushmem_my_pe(); }
extern "C" int mpa_world() { return dushmem_n_pes(); }
extern "C" int mpa_local_world() { return dushmem_team_n_pes(DUSHMEMX_TEAM_NODE); }
extern "C" void *mpa_alloc(size_t bytes) { return dushmem_malloc(bytes); }
extern "C" void *mpa_peer(void *allocation, int peer) {
  return dushmem_ptr(allocation, peer);
}
extern "C" int mpa_copy(void *dst, const void *src, size_t bytes, hipStream_t stream) {
  return static_cast<int>(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
}
extern "C" int mpa_barrier(hipStream_t stream) {
  dushmemx_barrier_all_on_stream(stream);
  return static_cast<int>(hipGetLastError());
}
extern "C" void mpa_free(void *allocation) { dushmem_free(allocation); return; }
extern "C" void mpa_finalize() { dushmem_finalize(); return; }
