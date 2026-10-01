#pragma once

// F5-S4 / F5-S7 shared code: the 1M-instance scene, the CPU reference of the
// cull test (same formulas and operation order as bench/f5_spike/shaders/
// s4_cull.metal) and the GPU rig (buffers + pipelines + the encoders of the
// compaction variants).  A measurement tool, not engine code.

#include "f5_common.h"

#include <array>
#include <functional>
#include <vector>

namespace f5::cull {

using soc::u8;

constexpr u32 kN        = 1u << 20; // instances
constexpr u32 kBuckets  = 8;        // meshes == buckets
constexpr u32 kGroupSz  = 1024;     // threads per scan group
constexpr u32 kGroups   = kN / kGroupSz;
static_assert(kGroups == 1024, "the group-count scan is one threadgroup of 1024 threads");

struct Instance { // 80 bytes, MSL `Instance`
    float m[16];  // column-major model matrix
    u32 mesh;
    u32 pad[3];
};
static_assert(sizeof(Instance) == 80);

struct CullParams { // MSL `CullParams`
    float planes[20];
    float cam[3];
    float maxDistance;
    float fwd[3];
    float minPixels;
    float projY;
    float halfViewportH;
    float nearPlane;
    u32 count;
};
static_assert(sizeof(CullParams) == 128);

struct Scene {
    std::vector<Instance> inst;
    float spheres[kBuckets][4]; // centre xyz, radius
    u32 bucketFirst[kBuckets + 1];
    u32 mirrored = 0;
};

/// Seeded scene: kN instances in 8 contiguous buckets of random sizes (none a multiple of 32).
Scene makeScene(u64 seed);

/// Camera at the origin looking down -Z, fov 60 deg, aspect 16:9, viewport 3200x1800, near 0.05,
/// reverse-Z infinite projection exactly as src/scene/camera.cpp builds it.  Unit-checks the plane
/// extraction on the CPU (throws soc::BenchError on a wrong plane).
CullParams makeParams(const Scene& scene, float maxDistance = 800.0f, float minPixels = 1.0f);

/// Relative width of the band around a decision boundary inside which GPU and CPU may disagree.
constexpr float kBand = 1e-4f;

struct CpuRef {
    std::vector<u8> vis;       // strict CPU decision
    std::vector<u8> ambiguous; // decision may flip within kBand (see cullCpu)
    u32 visible   = 0;
    u32 ambiguousCount = 0;
    u32 bucketVisible[kBuckets] = {};
};
CpuRef cullCpu(const Scene& scene, const CullParams& p);

struct Pipes {
    MTL::ComputePipelineState* clear;
    MTL::ComputePipelineState* atomic;
    MTL::ComputePipelineState* flags;
    MTL::ComputePipelineState* scan;
    MTL::ComputePipelineState* write;
};

struct Rig {
    MTL::Buffer* inst        = nullptr;
    MTL::Buffer* spheres     = nullptr;
    MTL::Buffer* bucketFirst = nullptr;
    MTL::Buffer* params      = nullptr;
    MTL::Buffer* counters    = nullptr; // (a) 8 x u32
    MTL::Buffer* listAtomic  = nullptr; // (a) N x u32, bucket b in [bucketFirst[b], bucketFirst[b+1])
    MTL::Buffer* flags       = nullptr; // (b) N x u8
    MTL::Buffer* groupCounts = nullptr; // (b) 1024 x u32
    MTL::Buffer* groupOffs   = nullptr; // (b) 1025 x u32
    MTL::Buffer* prefix      = nullptr; // (b) N+1 x u32
    MTL::Buffer* listStable  = nullptr; // (b) N x u32
    Pipes fast{}, safe{};
};

Rig makeRig(soc::Context& ctx, const Scene& scene, const CullParams& p);
void setParams(Rig& rig, const CullParams& p);

/// Dispatches of (a): clear, cull+compact.  `lap` runs after each dispatch (timer lap or just a barrier).
void encodeAtomic(soc::Context& ctx, MTL4::ComputeCommandEncoder* e, const Rig& rig, const Pipes& pipes,
                  const std::function<void()>& lap);
/// Dispatches of (b): flags+group counts, scan of the counts, write list+prefix.
void encodeStable(soc::Context& ctx, MTL4::ComputeCommandEncoder* e, const Rig& rig, const Pipes& pipes,
                  const std::function<void()>& lap);

} // namespace f5::cull
