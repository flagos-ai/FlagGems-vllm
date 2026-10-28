// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Benchmark-only DUSHMEM lifecycle shim for the Hygon MegaMoE harness.
// The production operator accepts caller-owned symmetric tensors and never
// initializes, allocates, or finalizes DUSHMEM itself.

#include <cstring>
#include <dushmem.h>
#include <dushmemx.h>
#include <hip/hip_runtime.h>
#include <stddef.h>

#define HIP_CHECK(statement)                                                   \
  do {                                                                         \
    const hipError_t status = (statement);                                     \
    if (status != hipSuccess)                                                  \
      return -1;                                                               \
  } while (0)

extern "C" size_t flaggems_megamoe_unique_id_nbytes(void) {
  return sizeof(dushmemx_uniqueid_t);
}

extern "C" int flaggems_megamoe_get_unique_id(void *buffer,
                                                size_t buffer_size) {
  if (buffer == nullptr || buffer_size < sizeof(dushmemx_uniqueid_t))
    return -1;
  dushmemx_uniqueid_t unique_id = DUSHMEMX_UNIQUEID_INITIALIZER;
  const int status = dushmemx_get_uniqueid(&unique_id);
  if (status != 0)
    return status;
  std::memcpy(buffer, &unique_id, sizeof(unique_id));
  return 0;
}

extern "C" int flaggems_megamoe_init(int rank, int world_size,
                                      int local_device,
                                      const void *unique_id_bytes,
                                      size_t unique_id_nbytes) {
  if (rank < 0 || world_size <= 0 || rank >= world_size || local_device < 0 ||
      unique_id_bytes == nullptr ||
      unique_id_nbytes < sizeof(dushmemx_uniqueid_t))
    return -1;
  HIP_CHECK(hipSetDevice(local_device));

  dushmemx_uniqueid_t unique_id = DUSHMEMX_UNIQUEID_INITIALIZER;
  std::memcpy(&unique_id, unique_id_bytes, sizeof(unique_id));
  dushmemx_init_attr_t attributes;
  std::memset(&attributes, 0, sizeof(attributes));
  int status = dushmemx_set_attr_uniqueid_args(rank, world_size, &unique_id,
                                                &attributes);
  if (status != 0)
    return status;
  return dushmemx_init_attr(DUSHMEMX_INIT_WITH_UNIQUEID, &attributes);
}

extern "C" int flaggems_megamoe_rank_info(int *rank, int *world_size,
                                           int *local_rank) {
  if (rank == nullptr || world_size == nullptr || local_rank == nullptr)
    return -1;
  *rank = dushmem_my_pe();
  *world_size = dushmem_n_pes();
  *local_rank = dushmem_team_my_pe(DUSHMEMX_TEAM_NODE);
  return 0;
}

extern "C" int flaggems_megamoe_is_single_node(void) {
  return dushmem_team_n_pes(DUSHMEMX_TEAM_NODE) == dushmem_n_pes();
}

extern "C" int flaggems_megamoe_symmetric_alloc(size_t nbytes,
                                                 void **allocation) {
  if (nbytes == 0 || allocation == nullptr)
    return -1;
  dushmem_barrier_all();
  *allocation = dushmem_malloc(nbytes);
  if (*allocation == nullptr)
    return -2;
  HIP_CHECK(hipMemset(*allocation, 0, nbytes));
  dushmem_barrier_all();
  return 0;
}

extern "C" void *flaggems_megamoe_peer_pointer(void *allocation, int peer) {
  if (allocation == nullptr || peer < 0 || peer >= dushmem_n_pes())
    return nullptr;
  return dushmem_ptr(allocation, peer);
}

extern "C" int flaggems_megamoe_symmetric_free(void *allocation) {
  if (allocation == nullptr)
    return -1;
  dushmem_barrier_all();
  dushmem_free(allocation);
  dushmem_barrier_all();
  return 0;
}

extern "C" int flaggems_megamoe_finalize(void) {
  dushmem_finalize();
  return 0;
}
