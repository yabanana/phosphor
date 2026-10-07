#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

class GpuScene;

// Coarse to fine, with exactly the S4 border-locked ladder. These are targets:
// meshoptimizer can retain more triangles when topology/borders require it.
enum class RtProxyLevel : u32 { R10 = 0, R25 = 1, R50 = 2, Full = 3 };
const char* rtProxyLevelName(RtProxyLevel level);
bool rtProxyParseLevel(std::string_view name, RtProxyLevel& level);

struct RtProxyCookedMesh {
    std::vector<u32> indices; // mesh-local indices into the UNMODIFIED GPUVertex range
    float relativeError = 0;
    float objectSpaceError = 0; // meshoptimizer's estimate, not a conservative ray-error bound
};

// Invalid geometry throws std::invalid_argument before entering meshoptimizer.
// Empty/non-triangle inputs are rejected. Small/unsimplifiable meshes stay full.
RtProxyCookedMesh rtCookProxyMesh(std::span<const GPUVertex> vertices,
                                std::span<const u32> indices, RtProxyLevel level);

struct RtProxyThresholds {
    double shadowPercent = 0.5;
    double primaryBadPercent = 0.2; // hit/miss differs or abs(dt) > 5 cm
    double distance95Cm = 1.0;
    double acnePercent = 0.5; // false hits on the receiver's own instance
};

struct RtProxyMeasurements {
    u64 primaryRays = 0;
    u64 shadowReceivers = 0;
    double shadowPercent = 0;
    double primaryBadPercent = 0;
    double distance95Cm = 0;
    double acnePercent = 0;
};

bool rtProxyMeetsThresholds(const RtProxyMeasurements& measurement,
                           const RtProxyThresholds& thresholds = {});

// S4 promotion: eligible meshes with >=2% of eligible blame move one step;
// at least the highest offender moves when any eligible blame exists.
// Call after each full/proxy measurement until thresholds pass. False means no
// progress is possible; callers MUST NOT export an unvalidated configuration.
bool rtProxyPromote(std::span<RtProxyLevel> levels, std::span<const double> blame);

struct RtProxyMesh {
    u32 mesh = 0;
    RtProxyLevel level = RtProxyLevel::Full;
    u32 sourceIndexCount = 0;
    u32 proxyIndexCount = 0;
    std::string indexFingerprint; // checks cooked output against the measured indices
    float relativeError = 0;
    float objectSpaceError = 0;
};

struct RtProxyManifest {
    u32 schema = 1;
    u32 meshoptimizerVersion = 0;
    std::string scene;
    std::string geometryFingerprint;
    // This scope is intentionally explicit: S4 measures opaque geometric rays,
    // in the named offline corpus. It is not an alpha/dynamic-scene guarantee.
    std::string measurementScope = "opaque-geometry-s4";
    std::string measurementCorpus;
    RtProxyThresholds thresholds;
    RtProxyMeasurements measured;
    std::vector<RtProxyMesh> meshes;
};

// Stable endian-independent FNV-1a fingerprints detect asset/cooker drift, not
// adversarial tampering. Covers vertex attributes, mesh ranges and all indices.
std::string rtProxyGeometryFingerprint(const GpuScene& scene);
std::string rtProxyIndexFingerprint(std::span<const u32> indices);
u32 rtProxyMeshoptimizerVersion();

// Regenerates declarations for the measured levels. The caller must supply
// successful measurements (including nonzero ray/receiver populations) and
// compare these index fingerprints with the ACTUAL measured buffers, as S4 does.
// build/load regenerate and compare every index signature before runtime use.
RtProxyManifest rtMakeProxyManifest(const GpuScene& scene, std::span<const RtProxyLevel> levels,
                                   const RtProxyMeasurements& measured, std::string sceneName,
                                   std::string measurementCorpus);

// Reader is bounded (16 MiB), rejects missing/non-finite/malformed fields and
// unsupported versions. Failure clears out and reports a reason; no silent use.
bool rtReadProxyManifest(const std::string& path, RtProxyManifest& out, std::string& error);
bool rtWriteProxyManifest(const std::string& path, const RtProxyManifest& manifest, std::string& error);

struct RtProxyGeometry {
    std::vector<u32> indices;
    // vertexOffset and boundingSphere preserved; RT indexOffset/indexCount
    // point into indices above. Meshlet ranges are cleared (never raster data).
    std::vector<GPUMeshInfo> meshes;
    std::vector<RtProxyMesh> selections;
    u64 fullTriangles = 0;
    u64 proxyTriangles = 0;
    bool manifestApplied = false;
    std::string diagnostic;
};

// All-or-nothing validation: a missing/stale/incompatible manifest falls back to
// the full geometry. Never changes GpuScene or its raster/meshlet index streams.
// Protect alpha-tested/emissive meshes (and unknown material references) until
// an appropriate material-aware corpus validates their simplification. Invalid
// instance mesh IDs are ignored. IDs are sorted, unique, and independent of slots.
std::vector<u32> rtProxyProtectedMeshes(u32 meshCount, std::span<const GPUInstance> instances,
                                      std::span<const GPUMaterial> materials);

// protectedMeshes additionally promotes individual valid manifest selections
// to Full. The caller must rebuild if instance/material assignments change.
RtProxyGeometry rtBuildProxyGeometry(const GpuScene& scene, const RtProxyManifest* manifest = nullptr,
                                    std::span<const u32> protectedMeshes = {});

} // namespace phosphor
