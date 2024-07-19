// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_NPKIT_H_
#define ARK_NPKIT_H_

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
    NpKitEvent* npkit_event_buffer;
    uint64_t npkit_event_buffer_head;
};

#pragma pack(pop)

#if defined(ARK_TARGET_CUDA_ARCH)
#define NPKIT_GET_GPU_TIMESTAMP clock64
#define __syncshm() __syncthreads();
#elif defined(ARK_TARGET_ROCM_ARCH)
#define NPKIT_GET_GPU_TIMESTAMP wall_clock64
#define __syncshm() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#endif

#define NPKIT_CPU_TIMESTAMP_SLOT_SIZE 128
#define NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(buf, blk) \
    *(buf + NPKIT_CPU_TIMESTAMP_SLOT_SIZE * blk / sizeof(uint64_t))

// Flush to global memory per 64 events
#define NPKIT_SHM_NUM_EVENTS 64

namespace ark {

class NpKit {
   public:
    static __forceinline__ __device__ void CollectGpuEventShm(
        uint32_t type, uint32_t rsvd, uint64_t timestamp,
        NpKitEvent* npkit_event_buffer, uint64_t* npkit_event_buffer_head) {
        if (*npkit_event_buffer_head < NPKIT_SHM_NUM_EVENTS) {
            if (threadIdx.x == 0) {
                NpKitEvent& event =
                    npkit_event_buffer[*npkit_event_buffer_head];
                event.fields.type = type;
                event.fields.rsvd = rsvd;
                event.fields.timestamp = timestamp;
            }
            (*npkit_event_buffer_head)++;
        }
    }

    static __forceinline__ __device__ void StoreGpuEventShm(
        NpKitEventCollectContext* npkit_contexts,
        NpKitEvent* npkit_event_buffer, uint64_t* npkit_event_buffer_head) {
        __syncshm();
        NpKitEventCollectContext* npkit_context = npkit_contexts + blockIdx.x;
        NpKitEvent* global_event_buffer = npkit_context->npkit_event_buffer;
        uint64_t global_event_buffer_head =
            npkit_context->npkit_event_buffer_head;
        *npkit_event_buffer_head =
            min(*npkit_event_buffer_head,
                kMaxNumGpuEventsPerBlock - global_event_buffer_head);
        for (size_t i = threadIdx.x;
             i < (*npkit_event_buffer_head) * sizeof(NpKitEvent) / sizeof(int4);
             i += blockDim.x) {
            ((int4*)(global_event_buffer + global_event_buffer_head))[i] =
                ((int4*)npkit_event_buffer)[i];
        }
        if (threadIdx.x == 0) {
            npkit_context->npkit_event_buffer_head += *npkit_event_buffer_head;
        }
        *npkit_event_buffer_head = 0;
        __syncshm();
    }

   private:
    static const uint64_t kMaxNumGpuEventsPerBlock = 1ULL << 16;
};

}  // namespace ark

#endif
