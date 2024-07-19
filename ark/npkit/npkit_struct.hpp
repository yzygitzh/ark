// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_NPKIT_STRUCT_HPP_
#define ARK_NPKIT_STRUCT_HPP_

#include <cstdint>

#pragma pack(push, 1)

union NpKitEvent {
    uint64_t bits[2];
    struct {
        uint32_t type;
        uint32_t rsvd;
        uint64_t timestamp;
    } fields;
};

struct NpKitEventCollectContext {
    NpKitEvent* event_buffer;
    uint64_t event_buffer_head;
};

#pragma pack(pop)

#endif
