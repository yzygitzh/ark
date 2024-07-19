// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "npkit/npkit.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>

#include "gpu/gpu_logging.hpp"

namespace ark {

int NpKit::rank_ = 0;
int NpKit::num_blocks_ = 0;
std::vector<std::shared_ptr<GpuMemory>> NpKit::gpu_event_buffers_;
std::shared_ptr<GpuMemory> NpKit::gpu_collect_contexts_;
std::shared_ptr<GpuHostMemory> NpKit::cpu_timestamp_;
std::unique_ptr<std::thread> NpKit::cpu_timestamp_update_thread_;
volatile bool NpKit::cpu_timestamp_update_thread_should_stop_ = false;

void NpKit::CpuTimestampUpdateThread() {
    uint64_t init_system_clock =
        std::chrono::system_clock::now().time_since_epoch().count();
    uint64_t init_steady_clock =
        std::chrono::steady_clock::now().time_since_epoch().count();
    uint64_t curr_steady_clock = 0;
    while (!cpu_timestamp_update_thread_should_stop_) {
        for (int i = 0; i < num_blocks_; i++) {
            curr_steady_clock =
                std::chrono::steady_clock::now().time_since_epoch().count();
            NPKIT_STORE_CPU_TIMESTAMP_PER_BLOCK(
                cpu_timestamp_.get()->ref<uint64_t>(),
                init_system_clock + (curr_steady_clock - init_steady_clock), i);
        }
    }
}

void NpKit::Init(int rank, int num_blocks) {
    int i = 0;
    NpKitEventCollectContext ctx;
    ctx.event_buffer_head = 0;
    rank_ = rank;
    num_blocks_ = num_blocks;

    auto gpu_manager = GpuManager::get_instance(rank_);

    // Init event data structures
    gpu_collect_contexts_ =
        gpu_manager->malloc(sizeof(NpKitEventCollectContext) * num_blocks_);
    for (i = 0; i < num_blocks_; i++) {
        gpu_event_buffers_.emplace_back(
            gpu_manager->malloc(sizeof(NpKitEvent) * kMaxNumGpuEventsPerBlock));
        ctx.event_buffer = gpu_event_buffers_[i].get()->ref<NpKitEvent>();
        GLOG(gpuMemcpy(
            gpu_collect_contexts_.get()->ref<NpKitEventCollectContext>() + i,
            &ctx, sizeof(NpKitEventCollectContext), gpuMemcpyHostToDevice));
    }

    // Init timestamp. Allocates MAXCHANNELS*128 bytes buffer for GPU
    cpu_timestamp_ = gpu_manager->malloc_host(
        num_blocks_ * NPKIT_CPU_TIMESTAMP_SLOT_SIZE,
        gpuHostAllocMapped | gpuHostAllocWriteCombined);
    for (i = 0; i < num_blocks_; i++) {
        NPKIT_STORE_CPU_TIMESTAMP_PER_BLOCK(
            cpu_timestamp_.get()->ref<uint64_t>(),
            std::chrono::system_clock::now().time_since_epoch().count(), i);
    }
    cpu_timestamp_update_thread_should_stop_ = false;
    cpu_timestamp_update_thread_ =
        std::make_unique<std::thread>(CpuTimestampUpdateThread);
}

static int GetGpuClockRateInKhz() {
    int dev_id;
#if defined(ARK_CUDA)
    cudaDeviceProp dev_prop;
    GLOG(cudaGetDevice(&dev_id));
    GLOG(cudaGetDeviceProperties(&dev_prop, dev_id));
    return dev_prop.clockRate;
#elif defined(ARK_ROCM)
    hipDeviceProp_t dev_prop;
    char gcn_arch[256];
    GLOG(hipGetDevice(&dev_id));
    GLOG(hipGetDeviceProperties(&dev_prop, dev_id));
    char* gcnArchNameToken = strtok(dev_prop.gcnArchName, ":");
    strcpy(gcn_arch, gcnArchNameToken);
    if (strncmp("gfx94", gcn_arch, 5) == 0)
        return 100000;
    else
        return 25000;
#else
    return 0;
#endif
}

void NpKit::Dump(const std::string& dump_dir) {
    int i = 0;
    std::string dump_file_path;

    // Dump CPU clock info
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_clock_period_num_rank_";
    dump_file_path += std::to_string(rank_);
    std::string clock_period_num_str =
        std::to_string(std::chrono::steady_clock::duration::period::num);
    auto clock_period_num_file = std::fstream(dump_file_path, std::ios::out);
    clock_period_num_file.write(clock_period_num_str.c_str(),
                                clock_period_num_str.length());
    clock_period_num_file.close();

    dump_file_path = dump_dir;
    dump_file_path += "/cpu_clock_period_den_rank_";
    dump_file_path += std::to_string(rank_);
    std::string clock_period_den_str =
        std::to_string(std::chrono::steady_clock::duration::period::den);
    auto clock_period_den_file = std::fstream(dump_file_path, std::ios::out);
    clock_period_den_file.write(clock_period_den_str.c_str(),
                                clock_period_den_str.length());
    clock_period_den_file.close();

    // Dump GPU events, reuse CPU struct
    std::unique_ptr<NpKitEvent[]> cpu_event_buffer(
        new NpKitEvent[kMaxNumGpuEventsPerBlock]);
    NpKitEventCollectContext ctx;
    for (i = 0; i < num_blocks_; i++) {
        dump_file_path = dump_dir;
        dump_file_path += "/gpu_events_rank_";
        dump_file_path += std::to_string(rank_);
        dump_file_path += "_buf_";
        dump_file_path += std::to_string(i);
        GLOG(gpuMemcpy(cpu_event_buffer.get(),
                       gpu_event_buffers_[i].get()->ref<NpKitEvent>(),
                       sizeof(NpKitEvent) * kMaxNumGpuEventsPerBlock,
                       gpuMemcpyDeviceToHost));
        GLOG(gpuMemcpy(
            &ctx,
            gpu_collect_contexts_.get()->ref<NpKitEventCollectContext>() + i,
            sizeof(NpKitEventCollectContext), gpuMemcpyDeviceToHost));
        auto gpu_trace_file =
            std::fstream(dump_file_path, std::ios::out | std::ios::binary);
        gpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffer.get()),
                             ctx.event_buffer_head * sizeof(NpKitEvent));
        gpu_trace_file.close();
    }

    // Dump GPU clockRate
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_clock_rate_rank_";
    dump_file_path += std::to_string(rank_);
    std::string clock_rate_str = std::to_string(GetGpuClockRateInKhz());
    auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
    gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
    gpu_clock_rate_file.close();
}

void NpKit::Shutdown() {
    // Stop CPU timestamp updating thread
    cpu_timestamp_update_thread_should_stop_ = true;
    cpu_timestamp_update_thread_->join();

    // Free GPU event data structures
    gpu_event_buffers_.clear();
    gpu_collect_contexts_.reset();

    // Free timestamp
    cpu_timestamp_update_thread_.reset();
    cpu_timestamp_.reset();
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContexts() {
    return gpu_collect_contexts_.get()->ref<NpKitEventCollectContext>();
}

uint64_t* NpKit::GetCpuTimestamp() {
    return cpu_timestamp_.get()->ref<uint64_t>();
}

std::string NpKit::GetCodeForNpKitEntryParams() {
    return ", NpKitEventCollectContext* npkit_context, uint64_t* "
           "npkit_cpu_timestamp";
}

std::string NpKit::GetCodeForNpKitArkBodyParams() {
    return ", NpKitEventCollectContext* npkit_context, NpKitEvent* "
           "npkit_event_buffer, uint64_t* npkit_event_buffer_head";
}

std::string NpKit::GetCodeForNpKitArkBodyArgs() {
    return ", npkit_context, npkit_event_buffer, &npkit_event_buffer_head";
}

std::string NpKit::GetCodeForNpKitTaskSeqParams() {
    return ", NpKitEvent* npkit_event_buffer, uint64_t* "
           "npkit_event_buffer_head, "
           "int npkit_event_entry_id";
}

std::string NpKit::GetCodeForNpKitTaskSeqArgs(int npkit_event_entry_id) {
    std::string code_str = ", npkit_event_buffer, npkit_event_buffer_head, ";
    code_str += std::to_string(npkit_event_entry_id);
    return code_str;
}

std::string NpKit::GetCodeForNpKitInit() {
    std::string code_str =
        "NpKitEvent* npkit_event_buffer = (NpKitEvent*)((char*)shared_mem + "
        "ARK_SMEM_RESERVED_BYTES);\n"
        "  uint64_t npkit_event_buffer_head = 0;\n"
        "  NpKit::CollectGpuEventShm(";
    code_str += std::to_string(NPKIT_EVENT_TIME_SYNC_CPU);
    code_str +=
        ", 0, NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(npkit_cpu_timestamp, "
        "blockIdx.x), npkit_event_buffer, &npkit_event_buffer_head);\n"
        "  NpKit::CollectGpuEventShm(";
    code_str += std::to_string(NPKIT_EVENT_TIME_SYNC_GPU);
    code_str +=
        ", 0, NPKIT_GET_GPU_TIMESTAMP(), npkit_event_buffer, "
        "&npkit_event_buffer_head);\n";
    return code_str;
}

std::string NpKit::GetCodeForNpKitCollectEventEntry() {
    std::string code_str =
        "NpKit::CollectGpuEventShm(npkit_event_entry_id, 0, "
        "NPKIT_GET_GPU_TIMESTAMP(), npkit_event_buffer, "
        "npkit_event_buffer_head);\n";
    return code_str;
}

std::string NpKit::GetCodeForNpKitCollectEventExit() {
    std::string code_str =
        "NpKit::CollectGpuEventShm(npkit_event_entry_id + 1, 0, "
        "NPKIT_GET_GPU_TIMESTAMP(), npkit_event_buffer, "
        "npkit_event_buffer_head);\n";
    return code_str;
}

std::string NpKit::GetCodeForNpKitFlushEvents() {
    return "  NpKit::StoreGpuEventShm(npkit_context, npkit_event_buffer, "
           "npkit_event_buffer_head);\n";
}

}  // namespace ark
