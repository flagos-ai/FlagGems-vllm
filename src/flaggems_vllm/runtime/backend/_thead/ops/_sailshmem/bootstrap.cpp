// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#define NVSHMEM_HOSTLIB_ONLY 1

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstring>

extern "C" size_t mega_uid_size() {
    return sizeof(nvshmemx_uniqueid_t);
}

extern "C" int mega_make_uid(void *destination) {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    const int status = nvshmemx_get_uniqueid(&id);
    if (status == 0) {
        std::memcpy(destination, &id, sizeof(id));
    }
    return status;
}

extern "C" int mega_init(const void *source, int rank, int world) {
    nvshmemx_uniqueid_t id;
    std::memcpy(&id, source, sizeof(id));
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    const int status = nvshmemx_set_attr_uniqueid_args(rank, world, &id, &attr);
    if (status != 0) {
        return status;
    }
    return nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}
