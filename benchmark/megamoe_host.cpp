// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Caller-side communication bridge; compile the device part with
// -DMEGAMOE_DEVICE_LIBRARY. The operator receives the prepared libraries.
#if defined(MEGAMOE_DEVICE_LIBRARY)
#include <mcr/mc_runtime.h>

extern "C" __device__ void mxshmem_fence();

// Triton's elementwise extern interface requires a return value.
extern "C" __device__ __attribute__((used)) int flaggems_mxshmem_fence() {
  mxshmem_fence();
  return 0;
}

#else

#include <mpi.h>
#include <mxshmem/host/mxshmem_api.h>
#include <cstdlib>
#include <climits>

static bool owns_mpi = false;
static bool initialized_shmem = false;
static MPI_Comm bootstrap_comm;
static mxshmemx_init_attr_t bootstrap_attr{};
extern "C" void mxshmem_init_wrapper() {
  if (initialized_shmem) return;
  int initialized = 0, provided = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
    owns_mpi = true;
  }
  bootstrap_comm = MPI_COMM_WORLD;
  bootstrap_attr.mpi_comm = &bootstrap_comm;
  mxshmemx_init_attr(MXSHMEMX_INIT_WITH_MPI_COMM, &bootstrap_attr);
  initialized_shmem = true;
}
extern "C" int mxshmem_mype_wrapper() { return mxshmem_my_pe(); }
extern "C" int mxshmem_npes_wrapper() { return mxshmem_n_pes(); }
extern "C" void *mxshmem_alloc_bytes_wrapper(size_t size) { return mxshmem_malloc(size); }
extern "C" void *mxshmem_ptr_wrapper(const void *ptr, int pe) { return mxshmem_ptr(ptr, pe); }
extern "C" void mxshmemx_barrier_wrapper(mcStream_t stream) { mxshmemx_barrier_all_on_stream(stream); }
extern "C" void mxshmem_barrier_all_wrapper() { mxshmem_barrier_all(); }
extern "C" void mxshmem_free_wrapper(void *ptr) { mxshmem_free(ptr); }
extern "C" void mxshmem_finalize_wrapper() {
  if (std::getenv("MEGAMOE_REUSE_SHMEM")) return;
  mxshmem_finalize();
  initialized_shmem = false;
  if (owns_mpi) MPI_Finalize();
}
extern "C" void mxshmem_session_finalize_wrapper() {
  if (initialized_shmem) mxshmem_finalize();
  initialized_shmem = false;
  if (owns_mpi) MPI_Finalize();
}
extern "C" int mxshmem_all_success_wrapper(int success) {
  int all = 0;
  MPI_Allreduce(&success, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all;
}
static int mpi_byte_count(size_t bytes) {
  if (bytes > INT_MAX) MPI_Abort(MPI_COMM_WORLD, 1);
  return static_cast<int>(bytes);
}
extern "C" void mxshmem_allgather_bytes_wrapper(const void *send, void *receive, size_t bytes) {
  int count = mpi_byte_count(bytes);
  MPI_Allgather(send, count, MPI_BYTE, receive, count, MPI_BYTE, MPI_COMM_WORLD);
}
extern "C" void mxshmem_broadcast_bytes_wrapper(void *data, size_t bytes, int root) {
  MPI_Bcast(data, mpi_byte_count(bytes), MPI_BYTE, root, MPI_COMM_WORLD);
}

#endif
