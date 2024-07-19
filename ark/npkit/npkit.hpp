// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_NPKIT_HPP_
#define ARK_NPKIT_HPP_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gpu/gpu_manager.hpp"
#include "npkit/npkit_event.hpp"
#include "npkit/npkit_struct.hpp"

#define NPKIT_CPU_TIMESTAMP_SLOT_SIZE 128
#define NPKIT_STORE_CPU_TIMESTAMP_PER_BLOCK(buf, val, blk) \
    *reinterpret_cast<volatile uint64_t*>(                 \
        buf + NPKIT_CPU_TIMESTAMP_SLOT_SIZE * blk / sizeof(uint64_t)) = val

// Flush to global memory per 64 events
#define NPKIT_SHM_NUM_EVENTS 64

namespace ark {

class NpKit {
   public:
    static void Init(int rank, int num_blocks);

    static void Dump(const std::string& dump_dir);

    static void Shutdown();

    static NpKitEventCollectContext* GetGpuEventCollectContexts();

    static uint64_t* GetCpuTimestamp();

    static std::string GetCodeForNpKitEntryParams();

    static std::string GetCodeForNpKitArkBodyParams();

    static std::string GetCodeForNpKitArkBodyArgs();

    static std::string GetCodeForNpKitTaskSeqParams();

    static std::string GetCodeForNpKitTaskSeqArgs(int npkit_event_entry_id);

    static std::string GetCodeForNpKitInit();

    static std::string GetCodeForNpKitCollectEventEntry();

    static std::string GetCodeForNpKitCollectEventExit();

    static std::string GetCodeForNpKitFlushEvents();

   private:
    static void CpuTimestampUpdateThread();

    static int rank_;
    static int num_blocks_;
    static const uint64_t kMaxNumGpuEventsPerBlock = 1ULL << 16;
    static std::vector<std::shared_ptr<GpuMemory>> gpu_event_buffers_;
    static std::shared_ptr<GpuMemory> gpu_collect_contexts_;
    static std::shared_ptr<GpuHostMemory> cpu_timestamp_;
    static std::unique_ptr<std::thread> cpu_timestamp_update_thread_;
    static volatile bool cpu_timestamp_update_thread_should_stop_;
};

}  // namespace ark

#endif
