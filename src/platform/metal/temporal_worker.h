#pragma once
#include "renderer/temporal_worker_layout.h"
#include <Metal/Metal.hpp>
#include <atomic>
#include <memory>

namespace phosphor {
class GpuMemory;
// One logical scaler lifetime. The GPU bridge retains this object until its
// last GPU reader finishes. This object never owns that bridge (no cycle).
class TemporalWorker {
  public:
    TemporalWorker(MTL::Device *, u32 width, u32 height);
    ~TemporalWorker();
    TemporalWorker(const TemporalWorker &) = delete;
    TemporalWorker &operator=(const TemporalWorker &) = delete;
    void start(); // PipelineCache utility worker, never the render thread.
    void cancelPending();
    void retire(); // Called by the completed GPU buffer's deallocator.
    bool ready() const;
    bool finished() const;
    bool failed() const;
    u64 enqueue(temporal_worker::Request);
    void submitted(u64 ticket);
    MTL::SharedEvent *inputReady() const;
    MTL::SharedEvent *outputReady() const;
    void *mapping() const;
    const temporal_worker::Layout &layout() const;
    u64 deviceBytes() const;
    u64 physicalFootprint() const;
    u64 gpuAllocations() const;
    static u64 spawnedCount();
    static u64 reapedCount();
    static u64 peakLiveCount();
    static u64 failureCount();
    static u64 mappedBytes();
    static u64 totalGpuAllocations();
    static u64 localPhysicalFootprint();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Internal child mode of the same executable. Uses the inherited private
// socket (3) and unlinked mapping FD (4), never a registered/network service.
int runTemporalWorker();
} // namespace phosphor
