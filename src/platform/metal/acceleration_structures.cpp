#include "platform/metal/acceleration_structures.h"

#include "core/log.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/upload_ring.h"
#include "renderer/gpu_scene.h"
#include "renderer/rt_scene.h"
#include "renderer/scene_store.h"
#include "rendergraph/pass_context.h"

#include <Metal/MTL4AccelerationStructure.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace phosphor {
namespace {
constexpr auto kCategory = MemoryCategory::RayTracing;
NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }
u64 alignUp(u64 v, u64 a) { return (v + a - 1) & ~(a - 1); }
MTL4::BufferRange range(MTL::Buffer* b, u64 offset = 0, u64 bytes = 0) {
    return MTL4::BufferRange::Make(b->gpuAddress() + offset, bytes ? bytes : b->length() - offset);
}
u64 resourceID(MTL::AccelerationStructure* as) { return as->gpuResourceID()._impl; }
pipe::PipelineDesc kernelDesc(const char* name) {
    pipe::PipelineDesc d;
    d.kind = pipe::PipelineKind::Compute;
    d.label = name;
    d.functions = {name, "", ""};
    return d;
}
u32 probeType(RtProbe p) {
    switch (p) {
    case RtProbe::Shadow: return RT_PROBE_SHADOW;
    case RtProbe::AO: return RT_PROBE_AO;
    case RtProbe::Diffuse: return RT_PROBE_DIFFUSE;
    default: return RT_PROBE_PRIMARY;
    }
}
const char* probeName(u32 p) {
    switch (p) {
    case RT_PROBE_SHADOW: return "shadow";
    case RT_PROBE_AO: return "ao";
    case RT_PROBE_DIFFUSE: return "diffuse";
    default: return "primary";
    }
}
} // namespace

struct AccelerationStructures::Impl {
    enum Table : u32 { ClearInstances, Instances, ClearPrimary, ClearProbe, GeneratePrimary,
                       TracePrimary, GenerateSecondary, TraceFinal, Debug, Present, TableCount };
    struct FrameSet {
        MTL::Buffer *descriptors = nullptr, *meshTable = nullptr, *params = nullptr, *probe = nullptr;
        MTL::Buffer *scratch = nullptr, *counters = nullptr, *primaryCounters = nullptr, *probeCounters = nullptr;
        MTL::Buffer *primaryRays = nullptr, *primaryHits = nullptr, *rays = nullptr, *hits = nullptr;
        MTL::Buffer *instanceReadback = nullptr, *materialReadback = nullptr, *querySizes = nullptr;
        std::vector<std::pair<u32, u64>> queried;
        RtTlasAction action = RtTlasAction::None;
        MTL::AccelerationStructure* tlas = nullptr;
        MTL4::InstanceAccelerationStructureDescriptor* descriptor = nullptr;
        MTL::AccelerationStructureSizes sizes{};
        MTL::IntersectionFunctionTable* ift = nullptr;
        MTL::ComputePipelineState* iftPipeline = nullptr; // Borrowed from PipelineCache.
        std::array<MTL4::ArgumentTable*, TableCount> tables{};
        u32 capacity = 0, rayCapacity = 0, instanceReadbackCapacity = 0, materialReadbackCapacity = 0;
        u32 raysUsed = 0, instancesUsed = 0, materialsUsed = 0, width = 0, height = 0, type = RT_PROBE_PRIMARY;
        u64 frame = 0, geometryRevision = 0;
        bool recorded = false, accounted = false, check = false, secondary = false, trace = false;
        std::shared_ptr<const std::vector<GPUVertex>> geometrySnapshot;
    };
    struct Blas {
        MTL4::PrimitiveAccelerationStructureDescriptor* descriptor = nullptr;
        MTL::AccelerationStructureSizes sizes{};
        u32 vertexCount = 0;
    };
    struct Maintenance {
        RtWork work;
        MTL::AccelerationStructure *source = nullptr, *destination = nullptr;
        u64 scratchOffset = 0, scratchBytes = 0;
    };
    struct VertexUpdate { u32 mesh = 0; u64 offset = 0, bytes = 0, uploadOffset = 0; };

    MetalContext& context;
    PipelineCache& pipelines;
    SceneRenderer& renderer;
    LaunchOptions options;
    MetalTextureManager* textures = nullptr;
    RtScene ledger{METAL_FRAMES_IN_FLIGHT};
    RtProxyGeometry proxy;
    std::shared_ptr<std::vector<GPUVertex>> cpuVertices = std::make_shared<std::vector<GPUVertex>>();
    std::vector<GPURtMesh> meshTable;
    std::vector<Blas> blases;
    std::unordered_map<u64, MTL::AccelerationStructure*> resources;
    std::vector<RtWork> work;
    std::vector<RtRetiredBlas> retired;
    std::vector<Maintenance> maintenance;
    std::vector<VertexUpdate> vertexUpdates, pendingVertexUpdates;
    std::array<FrameSet, METAL_FRAMES_IN_FLIGHT> frames{};
    MTL::Buffer *indexBuffer = nullptr, *maintenanceScratch = nullptr;
    UploadRing::Slice vertexUpload;
    u64 graphVersion = 1, geometryVersion = 1, compactEligibleFrame = 0;
    u64 maintenanceScratchCapacity = 0, sourceBlasBytes = 0;
    FrameParams frame{};
    u32 slots = 0, materialCount = 0, rayCount = 0, selectedProbe = RT_PROBE_PRIMARY;
    bool enabled = false, trace = false, secondary = false, diagnostic = false;
    bool debugView = false, sampled = false;
    u64 lastGraphSignature = ~u64{0};
    RtReport stats{};

    pipe::PipelineHandle clearPipeline = pipe::INVALID_PIPELINE, instancePipeline = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle generatePrimary = pipe::INVALID_PIPELINE, tracePipeline = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle generateSecondary = pipe::INVALID_PIPELINE, debugPipeline = pipe::INVALID_PIPELINE;
    std::unordered_map<u32, pipe::PipelineHandle> presentPipelines;
    pipe::PipelineHandle presentPipeline = pipe::INVALID_PIPELINE;

    rg::BufferRef descriptorRef{}, meshTableRef{}, vertexRef{}, indexRef{}, counterRef{}, primaryCounterRef{}, probeCounterRef{};
    rg::BufferRef primaryRaysRef{}, primaryHitsRef{}, raysRef{}, hitsRef{}, snapshotRef{}, materialSnapshotRef{}, uploadRef{}, scratchRef{}, queryRef{}, tlasScratchRef{};
    rg::AccelerationStructureRef tlas{};
    std::vector<rg::AccelerationStructureRef> blasRefs, sourceRefs;
    rg::TextureRef debugTexture{};

    Impl(MetalContext& c, PipelineCache& p, SceneRenderer& r, const LaunchOptions& o)
        : context(c), pipelines(p), renderer(r), options(o) {
        debugView = o.debugView == MeshletDebugView::RT;
        stats.present = o.rtEnabled;
        stats.enabled = o.rtEnabled;
        if (!o.rtEnabled) return;
        clearPipeline = p.request(kernelDesc("rt_clear_counters"));
        instancePipeline = p.request(kernelDesc("rt_write_instances"));
        generatePrimary = p.request(kernelDesc("rt_generate_primary"));
        generateSecondary = p.request(kernelDesc("rt_generate_secondary"));
        auto traceDesc = kernelDesc("rt_trace_rays");
        traceDesc.linkedFunctions = {"rt_alpha_generic"};
        tracePipeline = p.request(traceDesc);
        debugPipeline = p.request(kernelDesc("rt_debug_view"));
        for (auto format : {rg::Format::BGRA8Srgb, rg::Format::RGBA16Float}) requestPresent(format);
        for (auto& f : frames) {
            for (auto*& table : f.tables) {
                auto* d = MTL4::ArgumentTableDescriptor::alloc()->init();
                d->setMaxBufferBindCount(12);
                d->setMaxTextureBindCount(1);
                NS::Error* error = nullptr;
                table = c.device()->newArgumentTable(d, &error);
                d->release();
                if (!table) throw std::runtime_error("RT argument table creation failed");
            }
        }
    }

    pipe::PipelineHandle requestPresent(rg::Format format) {
        const auto key = static_cast<u32>(format);
        if (auto it = presentPipelines.find(key); it != presentPipelines.end()) return it->second;
        pipe::PipelineDesc d;
        d.kind = pipe::PipelineKind::Render;
        d.label = "RT present";
        d.functions = {"rt_present_vs", "rt_present_fs", ""};
        d.output(0, format);
        return presentPipelines.emplace(key, pipelines.request(d)).first->second;
    }
    MTL::Buffer* buffer(u64 bytes, bool shared, const char* label) {
        auto* b = context.memory().newBuffer(std::max<u64>(bytes, 16),
            shared ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate, kCategory, label);
        if (!b) throw std::runtime_error(std::string("RT buffer allocation failed: ") + label);
        return b;
    }
    template<class T> void release(T*& resource) {
        if (resource) context.memory().release(resource, kCategory);
        resource = nullptr;
    }
    void releaseFrame(FrameSet& f) {
        for (auto** b : {&f.descriptors, &f.meshTable, &f.params, &f.probe, &f.scratch, &f.counters,
                        &f.primaryCounters, &f.probeCounters, &f.primaryRays, &f.primaryHits, &f.rays, &f.hits,
                        &f.instanceReadback, &f.materialReadback, &f.querySizes}) release(*b);
        release(f.tlas);
        release(f.ift);
        if (f.descriptor) f.descriptor->release();
        auto tables = f.tables;
        f = {};
        f.tables = tables;
    }
    void clear() {
        context.waitIdle();
        for (auto& f : frames) releaseFrame(f);
        for (auto& [id, as] : resources) context.memory().release(as, kCategory);
        resources.clear();
        for (auto& b : blases) if (b.descriptor) b.descriptor->release();
        blases.clear(); meshTable.clear(); work.clear(); retired.clear(); maintenance.clear(); vertexUpdates.clear(); pendingVertexUpdates.clear();
        release(indexBuffer);
        release(maintenanceScratch);
        maintenanceScratchCapacity = 0;
        ledger = RtScene{METAL_FRAMES_IN_FLIGHT};
        proxy = {};
        cpuVertices = std::make_shared<std::vector<GPUVertex>>();
        enabled = trace = secondary = diagnostic = false;
        sourceBlasBytes = 0;
        textures = nullptr;
        ++graphVersion;
        ++geometryVersion;
        lastGraphSignature = ~u64{0};
        stats = {};
        stats.present = stats.enabled = options.rtEnabled;
    }
    ~Impl() {
        clear();
        for (auto& f : frames) for (auto* table : f.tables) if (table) table->release();
    }

    void load(const GpuScene& scene, const SceneStore& store, MetalTextureManager& textureManager) {
        clear();
        textures = &textureManager;
        if (!options.rtEnabled || scene.meshInfos().empty() || scene.vertices().empty() || !store.slotCapacity()) return;
        if (!renderer.vertexBuffer()) throw std::logic_error("RT load requires uploaded scene geometry");
        const auto protectedMeshes = rtProxyProtectedMeshes(scene.getMeshCount(), store.instances(), store.materials());
        RtProxyManifest manifest;
        bool haveManifest = false;
        if (options.rtProxyManifest) {
            const auto path = options.rtProxyManifestPath.empty() ? "assets/manifests/sponza.rtproxy.json"
                                                                 : options.rtProxyManifestPath;
            std::string error;
            haveManifest = rtReadProxyManifest(path, manifest, error);
            if (!haveManifest) LOG_WARN("RT proxy manifest not used: %s", error.c_str());
        }
        proxy = rtBuildProxyGeometry(scene, haveManifest ? &manifest : nullptr, protectedMeshes);
        if (options.rtProxyManifest && !proxy.manifestApplied)
            LOG_WARN("RT proxy uses full geometry: %s", proxy.diagnostic.c_str());
        *cpuVertices = scene.vertices();
        for (const auto& vertex : *cpuVertices)
            if (!std::isfinite(vertex.px) || !std::isfinite(vertex.py) || !std::isfinite(vertex.pz))
                throw std::invalid_argument("RT geometry contains a nonfinite vertex");
        for (const auto& material : store.materials())
            if (material.baseColorTex != INVALID_TEXTURE_INDEX && material.baseColorTex >= textureManager.textureCount())
                throw std::invalid_argument("RT material base-color texture index is outside bindless table");
        indexBuffer = buffer(proxy.indices.size() * sizeof(u32), false, "RT mesh-local indices");
        if (!proxy.indices.empty()) {
            auto upload = context.stagingAllocate(proxy.indices.size() * sizeof(u32));
            std::memcpy(upload.cpu, proxy.indices.data(), proxy.indices.size() * sizeof(u32));
            context.enqueueUpload([this, upload](auto* e) {
                e->copyFromBuffer(upload.buffer, upload.offset, indexBuffer, 0, proxy.indices.size() * sizeof(u32));
            });
            context.flushUploads();
        }
        blases.resize(proxy.meshes.size());
        meshTable.resize(proxy.meshes.size());
        std::vector<u64> scratchOffsets(blases.size());
        u64 scratchBytes = 0;
        for (u32 m = 0; m < blases.size(); ++m) {
            const auto& mesh = proxy.meshes[m];
            auto& b = blases[m];
            auto& table = meshTable[m];
            table.vertexOffset = mesh.vertexOffset;
            table.indexOffset = mesh.indexOffset;
            table.indexCount = mesh.indexCount;
            table.proxyLevel = static_cast<u32>(proxy.selections[m].level);
            const u32 vertexEnd = m + 1 < scene.meshInfos().size() ? scene.meshInfos()[m + 1].vertexOffset
                                                                 : static_cast<u32>(scene.vertices().size());
            if (vertexEnd < mesh.vertexOffset || mesh.indexCount % 3 ||
                u64(mesh.indexOffset) + mesh.indexCount > proxy.indices.size())
                throw std::invalid_argument("Invalid RT mesh ranges");
            b.vertexCount = vertexEnd - mesh.vertexOffset;
            if (!mesh.indexCount)
                throw std::invalid_argument("RT geometry requires a nonempty triangle list for every mesh");
            for (u32 j = 0; j < mesh.indexCount; ++j)
                if (proxy.indices[mesh.indexOffset + j] >= b.vertexCount)
                    throw std::invalid_argument("RT index escapes its mesh vertex range");
            auto* geo = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
            geo->setVertexBuffer(range(renderer.vertexBuffer(), u64(mesh.vertexOffset) * sizeof(GPUVertex),
                                       u64(b.vertexCount) * sizeof(GPUVertex)));
            geo->setVertexFormat(MTL::AttributeFormatFloat3);
            geo->setVertexStride(sizeof(GPUVertex));
            geo->setIndexBuffer(range(indexBuffer, u64(mesh.indexOffset) * sizeof(u32), u64(mesh.indexCount) * sizeof(u32)));
            geo->setIndexType(MTL::IndexTypeUInt32);
            geo->setTriangleCount(mesh.indexCount / 3);
            geo->setOpaque(false); // Per-instance Opaque/NonOpaque is dynamic.
            geo->setIntersectionFunctionTableOffset(0);
            b.descriptor = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
            b.descriptor->setGeometryDescriptors(NS::Array::array(geo));
            b.descriptor->setUsage(MTL::AccelerationStructureUsageRefit);
            geo->release();
            b.sizes = context.device()->accelerationStructureSizes(b.descriptor);
            auto* as = context.memory().newAccelerationStructure(b.sizes.accelerationStructureSize, kCategory, "RT BLAS");
            if (!as) throw std::runtime_error("RT BLAS allocation failed");
            const u64 id = resourceID(as);
            resources.emplace(id, as);
            ledger.request(m, 1, 1);
            ledger.markBuilt(m, id, b.sizes.accelerationStructureSize, context.frameIndex());
            ++stats.blasBuilds;
            table.blasLo = static_cast<u32>(id);
            table.blasHi = static_cast<u32>(id >> 32);
            scratchOffsets[m] = alignUp(scratchBytes, 256);
            scratchBytes = scratchOffsets[m] + std::max<u64>(b.sizes.buildScratchBufferSize, 16);
            sourceBlasBytes += b.sizes.accelerationStructureSize;
            ++stats.blasCount;
        }
        if (resources.empty()) return;
        auto* scratch = buffer(scratchBytes, false, "RT initial disjoint BLAS scratch");
        auto* compactSizes = buffer(blases.size() * sizeof(u64), true, "RT compacted sizes (64 bit)");
        std::memset(compactSizes->contents(), 0, compactSizes->length());
        const auto start = std::chrono::steady_clock::now();
        context.submitAndWait([&](auto* e) {
            for (u32 m = 0; m < blases.size(); ++m) {
                const auto& b = blases[m];
                if (!b.descriptor) continue;
                e->buildAccelerationStructure(resources.at(ledger.mesh(m).resourceID), b.descriptor,
                    range(scratch, scratchOffsets[m], std::max<u64>(b.sizes.buildScratchBufferSize, 16)));
            }
        }, &stats.blasBuildMs);
        // The completed build submission is measured separately from the
        // compact-size queries; no CPU wall time is labelled as GPU work.
        context.submitAndWait([&](auto* e) {
            for (u32 m = 0; m < blases.size(); ++m)
                if (blases[m].descriptor)
                    e->writeCompactedAccelerationStructureSize(resources.at(ledger.mesh(m).resourceID),
                                                               range(compactSizes, u64(m) * sizeof(u64), sizeof(u64)));
        });
        const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        LOG_INFO("RT BLAS load: %u structures, GPU build %.3f ms, CPU build+query wall %.3f ms", stats.blasCount,
                 static_cast<double>(stats.blasBuildMs), wallMs);
        const auto* sizes = static_cast<const u64*>(compactSizes->contents());
        for (u32 m = 0; m < blases.size(); ++m) {
            if (!blases[m].descriptor) continue;
            const auto bytes = sizes[m];
            if (!bytes || bytes > ledger.mesh(m).bytes)
                throw std::runtime_error("Invalid 64-bit compacted BLAS size");
            ledger.queueCompaction(m, bytes, ledger.mesh(m).version);
        }
        context.memory().release(scratch, kCategory);
        context.memory().release(compactSizes, kCategory);
        compactEligibleFrame = context.frameIndex() + 1;
        enabled = true;
        stats.alphaStrategy = "generic-ift";
        stats.effectiveFamily = context.effectiveFamilyName();
        stats.proxyMode = proxy.manifestApplied ? "manifest" : "off";
        stats.probe = rtProbeName(options.rtProbe);
        stats.fullTriangles = proxy.fullTriangles;
        stats.proxyTriangles = proxy.proxyTriangles;
        stats.uncompactedBytes = sourceBlasBytes;
        stats.blasScratchBytes = scratchBytes;
        for (const auto& selection : proxy.selections)
            if (selection.level != RtProxyLevel::Full) ++stats.proxyMeshes;
        if (proxy.manifestApplied) {
            stats.proxyShadowErrorPct = static_cast<float>(manifest.measured.shadowPercent);
            stats.proxyPrimaryErrorPct = static_cast<float>(manifest.measured.primaryBadPercent);
            stats.proxyDt95 = static_cast<float>(manifest.measured.distance95Cm);
            stats.proxyAcnePct = static_cast<float>(manifest.measured.acnePercent);
        }
        // Requesting/waiting for compilation is allowed only at scene loading.
        for (auto h : {clearPipeline, instancePipeline, generatePrimary, generateSecondary, tracePipeline, debugPipeline})
            pipelines.waitReady(h);
        const bool wantTrace = debugView || options.rtProbe != RtProbe::None || options.debugRt != 0;
        const bool wantFull = debugView || options.rtProbe != RtProbe::None || options.visibility;
        const u64 fullPixels = u64(context.width()) * context.height();
        if (fullPixels > std::numeric_limits<u32>::max()) throw std::invalid_argument("RT ray capacity overflow");
        const u32 initialRays = wantTrace ? static_cast<u32>(wantFull ? fullPixels : std::min<u64>(512, fullPixels)) : 0;
        trace = wantTrace;
        for (auto& f : frames) {
            reserveFrame(f, store.slotCapacity(), initialRays, initialRays, options.debugRt != 0,
                         static_cast<u32>(store.materials().size()));
            updateIft(f);
        }
        trace = false;
        ++graphVersion;
    }

    void reserveFrame(FrameSet& f, u32 capacity, u32 raysNeeded, u32 allocationRays, bool check, u32 materials) {
        if (!f.params) {
            f.params = buffer(sizeof(GPURtParams), true, "RT frame parameters");
            f.probe = buffer(sizeof(GPURtProbeParams), true, "RT probe parameters");
            f.meshTable = buffer(meshTable.size() * sizeof(GPURtMesh), true, "RT frame BLAS table");
            f.counters = buffer(sizeof(GPURtCounters), true, "RT instance counters");
            f.primaryCounters = buffer(sizeof(GPURtCounters), true, "RT primary setup counters");
            f.probeCounters = buffer(sizeof(GPURtCounters), true, "RT final probe counters");
            ++graphVersion;
        }
        if (capacity > f.capacity) {
            release(f.descriptors); release(f.scratch); release(f.tlas);
            if (f.descriptor) f.descriptor->release();
            f.descriptors = buffer(u64(capacity) * sizeof(GPURtInstanceDesc), false, "RT TLAS instance descriptors");
            f.descriptor = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
            f.descriptor->setInstanceCount(capacity);
            f.descriptor->setInstanceDescriptorBuffer(range(f.descriptors));
            f.descriptor->setInstanceDescriptorStride(sizeof(GPURtInstanceDesc));
            f.descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
            f.descriptor->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
            f.descriptor->setUsage(MTL::AccelerationStructureUsageRefit | MTL::AccelerationStructureUsagePreferFastBuild);
            f.sizes = context.device()->accelerationStructureSizes(f.descriptor);
            f.tlas = context.memory().newAccelerationStructure(f.sizes.accelerationStructureSize, kCategory, "RT per-slot TLAS");
            if (!f.tlas) throw std::runtime_error("RT TLAS allocation failed");
            f.scratch = buffer(std::max<u64>({f.sizes.buildScratchBufferSize, f.sizes.refitScratchBufferSize, 16}),
                               false, "RT persistent TLAS scratch");
            f.capacity = capacity;
            ++graphVersion;
        }
        if (raysNeeded && allocationRays > f.rayCapacity) {
            for (auto** b : {&f.primaryRays, &f.primaryHits, &f.rays, &f.hits}) release(*b);
            f.primaryRays = buffer(u64(allocationRays) * sizeof(GPURtRay), true, "RT primary rays");
            f.primaryHits = buffer(u64(allocationRays) * sizeof(GPURtHit), true, "RT primary hits");
            f.rays = buffer(u64(allocationRays) * sizeof(GPURtRay), true, "RT secondary rays");
            f.hits = buffer(u64(allocationRays) * sizeof(GPURtHit), true, "RT secondary hits");
            f.rayCapacity = allocationRays;
            ++graphVersion;
        }
        if (check && capacity > f.instanceReadbackCapacity) {
            release(f.instanceReadback);
            f.instanceReadback = buffer(u64(capacity) * sizeof(GPUInstance), true, "RT same-frame instance readback");
            f.instanceReadbackCapacity = capacity;
            ++graphVersion;
        }
        if (check && materials > f.materialReadbackCapacity) {
            release(f.materialReadback);
            f.materialReadback = buffer(u64(materials) * sizeof(GPUMaterial), true, "RT same-frame material readback");
            f.materialReadbackCapacity = materials;
            ++graphVersion;
        }
    }

    void prepareMaintenance() {
        maintenance.clear();
        vertexUpdates.clear();
        vertexUpdates.swap(pendingVertexUpdates);
        pendingVertexUpdates.clear();
        auto& f = frames[frame.slot];
        f.queried.clear();
        RtWorkBudget budget;
        // Initial compaction happens in a frame following the build/query.
        budget.compactions = frame.index >= compactEligibleFrame ? ~0u : 0;
        ledger.plan(budget, work);
        u64 scratchBytes = 0;
        for (const auto& w : work) {
            Maintenance job;
            job.work = w;
            const auto& old = ledger.mesh(w.mesh);
            job.source = old.resourceID ? resources.at(old.resourceID) : nullptr;
            if (w.kind == RtWorkKind::Compact) {
                job.destination = context.memory().newAccelerationStructure(old.compactBytes, kCategory, "RT compacted BLAS");
            } else if (w.kind == RtWorkKind::Build) {
                job.destination = context.memory().newAccelerationStructure(blases[w.mesh].sizes.accelerationStructureSize,
                                                                            kCategory, "RT rebuilt BLAS");
            } else job.destination = job.source;
            if (!job.destination) throw std::runtime_error("RT maintenance destination missing");
            if (w.kind != RtWorkKind::Refit) resources.emplace(resourceID(job.destination), job.destination);
            if (w.kind != RtWorkKind::Compact) {
                const auto& sizes = blases[w.mesh].sizes;
                job.scratchBytes = std::max<u64>(w.kind == RtWorkKind::Build ? sizes.buildScratchBufferSize
                                                                                            : sizes.refitScratchBufferSize, 16);
                job.scratchOffset = alignUp(scratchBytes, 256);
                scratchBytes = job.scratchOffset + job.scratchBytes;
            }
            maintenance.push_back(job);
            // Publish the destination before descriptor generation. The graph
            // makes it ready before TLAS build; the old AS remains in resources
            // until ALL snapshots abandon its resource ID and readers complete.
            if (w.kind == RtWorkKind::Compact) {
                if (!ledger.markCompacted(w.mesh, w.version, resourceID(job.destination), old.compactBytes, frame.index))
                    throw std::logic_error("RT compaction publication became stale on render thread");
                ++stats.compactions;
            } else if (w.kind == RtWorkKind::Build) {
                ledger.markBuilt(w.mesh, resourceID(job.destination), blases[w.mesh].sizes.accelerationStructureSize, frame.index);
                ++stats.blasBuilds;
                f.queried.emplace_back(w.mesh, ledger.mesh(w.mesh).version);
            } else {
                ledger.markRefitted(w.mesh, frame.index);
                ++stats.blasRefits;
            }
            const auto id = ledger.mesh(w.mesh).resourceID;
            meshTable[w.mesh].blasLo = static_cast<u32>(id);
            meshTable[w.mesh].blasHi = static_cast<u32>(id >> 32);
        }
        if (!f.queried.empty() && !f.querySizes) {
            f.querySizes = buffer(blases.size() * sizeof(u64), true, "RT maintenance compact size queries");
            ++graphVersion;
        }
        if (scratchBytes > maintenanceScratchCapacity) {
            release(maintenanceScratch);
            maintenanceScratch = buffer(scratchBytes, false, "RT maintenance disjoint scratch");
            maintenanceScratchCapacity = scratchBytes;
            ++graphVersion;
        }
        if (!vertexUpdates.empty()) {
            u64 bytes = 0;
            for (auto& update : vertexUpdates) {
                update.uploadOffset = bytes;
                bytes += update.bytes;
            }
            vertexUpload = context.frameUploads().allocate(bytes);
            for (const auto& update : vertexUpdates)
                std::memcpy(static_cast<u8*>(vertexUpload.cpu) + update.uploadOffset,
                            reinterpret_cast<const u8*>(cpuVertices->data()) + update.offset, update.bytes);
        } else vertexUpload = {};
    }

    GPURtCounters frameCounters(u32 slot) const {
        const auto& f = frames.at(slot);
        GPURtCounters out{};
        if (!f.recorded || !f.counters || context.frameEvent()->signaledValue() <= f.frame) return out;
        out = *static_cast<const GPURtCounters*>(f.counters->contents());
        if (f.trace) {
            const auto& c = *static_cast<const GPURtCounters*>(f.probeCounters->contents());
            out.rays = c.rays;
            out.hits = c.hits;
            out.alphaTests = c.alphaTests;
            out.opaqueAlphaTests = c.opaqueAlphaTests;
            if (f.secondary) {
                const auto& primary = *static_cast<const GPURtCounters*>(f.primaryCounters->contents());
                out.alphaTests += primary.alphaTests;
                out.opaqueAlphaTests += primary.opaqueAlphaTests;
            }
        }
        return out;
    }
    void collectCompleted() {
        const u64 value = context.frameEvent()->signaledValue();
        if (value) ledger.completeFrame(value - 1); // Every RT pass is on graphics.
        for (u32 slot = 0; slot < frames.size(); ++slot) {
            auto& f = frames[slot];
            if (!f.recorded || f.accounted || value <= f.frame) continue;
            const auto c = frameCounters(slot);
            stats.probeRays += c.rays;
            stats.alphaTests += c.alphaTests;
            stats.opaqueAlphaTests += c.opaqueAlphaTests;
            if (f.querySizes) {
                const auto* sizes = static_cast<const u64*>(f.querySizes->contents());
                for (auto [mesh, version] : f.queried) {
                    if (ledger.mesh(mesh).version != version) continue;
                    const auto bytes = sizes[mesh];
                    if (!bytes || bytes > ledger.mesh(mesh).bytes)
                        throw std::runtime_error("Invalid asynchronous RT compact size query");
                    ledger.queueCompaction(mesh, bytes, version);
                }
            }
            f.accounted = true;
        }
        ledger.collectRetired(retired);
        for (const auto& old : retired) {
            const auto it = resources.find(old.resourceID);
            if (it == resources.end()) throw std::logic_error("RT retired resource not owned by backend");
            context.memory().release(it->second, kCategory);
            resources.erase(it);
        }
    }

    void updateIft(FrameSet& f) {
        if (!trace) return;
        auto* pipeline = pipelines.compute(tracePipeline);
        if (!pipeline) throw std::runtime_error("RT trace pipeline unavailable after load");
        if (f.iftPipeline != pipeline) {
            release(f.ift);
            f.ift = context.memory().newIntersectionFunctionTable(pipeline, 1, kCategory, "RT per-slot alpha table");
            if (!f.ift) throw std::runtime_error("RT intersection function table allocation failed");
            auto* handle = pipeline->functionHandle(str("rt_alpha_generic"));
            if (!handle) throw std::runtime_error("RT linked alpha function missing from resolved pipeline");
            f.ift->setFunction(handle, 0);
            f.iftPipeline = pipeline;
        }
        // Only the completed frame slot is rebound, never an in-flight IFT.
        f.ift->setBuffer(renderer.buffers().materials(), 0, 0);
        f.ift->setBuffer(textures->tableBuffer(), 0, 1);
        f.ift->setBuffer(renderer.vertexBuffer(), 0, 2);
        f.ift->setBuffer(indexBuffer, 0, 3);
        f.ift->setBuffer(renderer.buffers().instances(), 0, 4);
        f.ift->setBuffer(f.meshTable, 0, 5);
        f.ift->setBuffer(f.params, 0, 6);
    }
    void bindRayTable(FrameSet& f, Table table, MTL::Buffer* rays, MTL::Buffer* hits, MTL::Buffer* counters) {
        auto* t = f.tables[table];
        t->setResource(f.tlas->gpuResourceID(), 0);
        t->setAddress(rays->gpuAddress(), 1);
        t->setAddress(hits->gpuAddress(), 2);
        t->setAddress(f.probe->gpuAddress(), 3);
        t->setResource(f.ift->gpuResourceID(), 4);
        t->setAddress(renderer.buffers().instances()->gpuAddress(), 5);
        t->setAddress(f.meshTable->gpuAddress(), 6);
        t->setAddress(renderer.vertexBuffer()->gpuAddress(), 7);
        t->setAddress(indexBuffer->gpuAddress(), 8);
        t->setAddress(counters->gpuAddress(), 9);
        t->setAddress(f.primaryHits->gpuAddress(), 10);
        t->setAddress(f.primaryRays->gpuAddress(), 11);
    }
    void prepare(const SceneStore& store, const FrameParams& p) {
        frame = p;
        if (!enabled) return;
        if (p.slot >= frames.size() || !p.width || !p.height)
            throw std::invalid_argument("RT frame slot or extent invalid");
        collectCompleted();
        auto& f = frames[p.slot];
        if (f.recorded && context.frameEvent()->signaledValue() <= f.frame)
            throw std::logic_error("RT prepare attempted to overwrite an in-flight slot");
        // Material-aware proxy acceptance is an asset contract. Refuse to keep
        // a simplified mesh after it becomes MASK/emissive without recooking.
        if (proxy.manifestApplied && (!store.materialDeltas().empty() || !store.instanceDeltas().empty() ||
                                     store.stats().fullMaterials || store.stats().fullInstances)) {
            for (u32 mesh : rtProxyProtectedMeshes(static_cast<u32>(meshTable.size()), store.instances(), store.materials()))
                if (proxy.selections[mesh].level != RtProxyLevel::Full)
                    throw std::runtime_error("RT proxy material assignment changed; reload with full/protected geometry");
        }
        if (!store.materialDeltas().empty() || store.stats().fullMaterials)
            for (const auto& material : store.materials())
                if (material.baseColorTex != INVALID_TEXTURE_INDEX && material.baseColorTex >= textures->textureCount())
                    throw std::invalid_argument("RT dynamic material texture index is outside bindless table");
        slots = store.slotCapacity();
        materialCount = static_cast<u32>(store.materials().size());
        // Keep the diagnostic graph/resources stable. --debug-rt N controls
        // CPU checking cadence; GPU snapshots are produced in every frame.
        diagnostic = options.debugRt > 0;
        trace = debugView || options.rtProbe != RtProbe::None || diagnostic;
        selectedProbe = options.rtProbe != RtProbe::None ? probeType(options.rtProbe) : RT_PROBE_PRIMARY;
        stats.probe = trace ? probeName(selectedProbe) : "none";
        secondary = trace && selectedProbe != RT_PROBE_PRIMARY;
        sampled = diagnostic && !debugView && options.rtProbe == RtProbe::None && !options.visibility;
        const u64 pixels = u64(p.width) * p.height;
        const u64 allocationPixels = u64(std::max(p.width, p.allocationWidth)) * std::max(p.height, p.allocationHeight);
        if (allocationPixels > std::numeric_limits<u32>::max()) throw std::invalid_argument("RT image exceeds ray index range");
        rayCount = trace ? static_cast<u32>(sampled ? std::min<u64>(512, pixels) : pixels) : 0;
        const u32 allocationRays = sampled ? std::min<u32>(512, static_cast<u32>(allocationPixels))
                                           : static_cast<u32>(allocationPixels);
        reserveFrame(f, slots, rayCount, allocationRays, diagnostic, materialCount);
        prepareMaintenance();
        // The descriptor count follows logical slots (masked holes included),
        // while backing buffers/AS may retain a larger capacity after shrink.
        f.descriptor->setInstanceCount(slots);
        const GPURtParams params{slots, static_cast<u32>(meshTable.size()), materialCount,
            diagnostic ? static_cast<u32>(options.debugRtCorrupt) : RT_CORRUPT_NONE};
        std::memcpy(f.params->contents(), &params, sizeof(params));
        std::memcpy(f.meshTable->contents(), meshTable.data(), meshTable.size() * sizeof(GPURtMesh));
        GPURtProbeParams probe{};
        const auto inverse = glm::inverse(glm::make_mat4(p.constants.viewProjection));
        std::memcpy(probe.inverseViewProjection, glm::value_ptr(inverse), sizeof(probe.inverseViewProjection));
        std::memcpy(probe.cameraPosition, p.constants.cameraPosition, sizeof(probe.cameraPosition));
        // Fixed directional light makes the probe corpus stable across benches.
        const glm::vec3 light = glm::normalize(glm::vec3(-0.4f, 1.0f, 0.3f));
        probe.lightDirection[0] = light.x; probe.lightDirection[1] = light.y; probe.lightDirection[2] = light.z;
        probe.lightDirection[3] = selectedProbe == RT_PROBE_AO ? 0.5f : 1e30f;
        probe.width = p.width; probe.height = p.height; probe.rayCount = rayCount; probe.probeType = selectedProbe;
        probe.slotCount = slots; probe.meshCount = static_cast<u32>(meshTable.size());
        probe.frameIndex = static_cast<u32>(p.index);
        probe.flags = 1u | ((p.constants.debugMode % 3u) << 1u) | (sampled ? 8u : 0u);
        std::memcpy(f.probe->contents(), &probe, sizeof(probe));
        f.tables[ClearInstances]->setAddress(f.counters->gpuAddress(), 0);
        f.tables[ClearPrimary]->setAddress(f.primaryCounters->gpuAddress(), 0);
        f.tables[ClearProbe]->setAddress(f.probeCounters->gpuAddress(), 0);
        auto* table = f.tables[Instances];
        table->setAddress(renderer.buffers().instances()->gpuAddress(), 0);
        table->setAddress(renderer.buffers().materials()->gpuAddress(), 1);
        table->setAddress(f.meshTable->gpuAddress(), 2);
        table->setAddress(f.descriptors->gpuAddress(), 3);
        table->setAddress(f.params->gpuAddress(), 4);
        table->setAddress(f.counters->gpuAddress(), 5);
        updateIft(f);
        if (trace) {
            bindRayTable(f, GeneratePrimary, f.primaryRays, f.primaryHits, f.primaryCounters);
            bindRayTable(f, TracePrimary, f.primaryRays, f.primaryHits, secondary ? f.primaryCounters : f.probeCounters);
            bindRayTable(f, GenerateSecondary, f.rays, f.hits, f.probeCounters);
            bindRayTable(f, TraceFinal, f.rays, f.hits, f.probeCounters);
            bindRayTable(f, Debug, secondary ? f.rays : f.primaryRays, secondary ? f.hits : f.primaryHits, f.probeCounters);
        }
        f.action = ledger.tlasAction(p.slot, slots, p.index, options.rtTlasRebuildEvery);
        f.frame = p.index; f.width = p.width; f.height = p.height;
        f.instancesUsed = slots; f.materialsUsed = materialCount; f.raysUsed = rayCount;
        f.check = diagnostic; f.secondary = secondary; f.trace = trace;
        f.type = selectedProbe; f.recorded = false; f.accounted = false;
        f.geometrySnapshot = diagnostic ? cpuVertices : nullptr;
        f.geometryRevision = geometryVersion;
        stats.instances = store.instanceCount();
        stats.capacity = slots;
        // Only structural conditions enter the cached graph key. Camera, ray
        // parameters and frame counters are runtime data in per-slot buffers.
        u64 signature = 1469598103934665603ull;
        auto hash = [&](u64 v) { signature ^= v; signature *= 1099511628211ull; };
        hash(trace); hash(secondary); hash(selectedProbe); hash(diagnostic); hash(!maintenance.empty()); hash(!vertexUpdates.empty());
        hash(slots); hash(meshTable.size()); hash(f.rayCapacity); hash(materialCount);
        hash(maintenance.size()); hash(f.queried.size()); hash(vertexUpload ? vertexUpload.buffer->length() : 0);
        if (debugView) { hash(p.width); hash(p.height); }
        for (const auto& job : maintenance) { hash(job.work.mesh); hash(static_cast<u32>(job.work.kind)); }
        if (signature != lastGraphSignature) { ++graphVersion; lastGraphSignature = signature; }
    }

    void dispatch(MTL4::ComputeCommandEncoder* encoder, pipe::PipelineHandle handle, MTL4::ArgumentTable* table, u32 n) {
        auto* pipeline = pipelines.compute(handle);
        if (!pipeline || !n) throw std::logic_error("RT dispatch requires a pipeline and nonempty work");
        encoder->setComputePipelineState(pipeline);
        encoder->setArgumentTable(table);
        const u32 group = std::min<u32>(64, static_cast<u32>(pipeline->maxTotalThreadsPerThreadgroup()));
        encoder->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(group, 1, 1));
    }

    void addPasses(rg::RenderGraph& graph) {
        using namespace rg;
        if (!enabled) return;
        auto& f = frames[frame.slot];
        descriptorRef = graph.importBuffer("RT descriptors", {u64(f.capacity) * sizeof(GPURtInstanceDesc)}, ImportPerFrame);
        meshTableRef = graph.importBuffer("RT BLAS table", {meshTable.size() * sizeof(GPURtMesh)}, ImportPerFrame | ImportContentsDefined);
        vertexRef = graph.importBuffer("RT/raster geometry vertices", {renderer.vertexBuffer()->length()}, ImportContentsDefined);
        indexRef = graph.importBuffer("RT geometry indices", {indexBuffer->length()}, ImportContentsDefined);
        counterRef = graph.importBuffer("RT instance counters", {sizeof(GPURtCounters)}, ImportPerFrame | ImportOutput);
        tlasScratchRef = graph.importBuffer("RT TLAS scratch", {f.scratch->length()}, ImportPerFrame);
        tlas = graph.importAccelerationStructure("RT TLAS", f.sizes.accelerationStructureSize,
                                                ImportPerFrame | ImportContentsDefined | ImportOutput);
        blasRefs.clear();
        blasRefs.resize(blases.size());
        for (u32 mesh = 0; mesh < blases.size(); ++mesh) {
            if (!blases[mesh].descriptor) continue;
            blasRefs[mesh] = graph.importAccelerationStructure("RT BLAS " + std::to_string(mesh),
                                blases[mesh].sizes.accelerationStructureSize, ImportContentsDefined);
        }
        if (!vertexUpdates.empty()) {
            uploadRef = graph.importBuffer("RT vertex upload", {vertexUpload.buffer->length()}, ImportPerFrame | ImportContentsDefined);
            graph.addPass("RT vertex update", PassType::Blit,
                [&](PassBuilder& b) {
                    b.read(uploadRef, Usage::CopySrc, StageBlit);
                    vertexRef = b.write(vertexRef, Usage::CopyDst, StageBlit);
                    renderer.declareGeometryWrite(b, StageBlit);
                },
                [this](PassContext& ctx) {
                    auto* e = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                    for (const auto& update : vertexUpdates)
                        e->copyFromBuffer(vertexUpload.buffer, vertexUpload.offset + update.uploadOffset,
                                          renderer.vertexBuffer(), update.offset, update.bytes);
                });
        } else uploadRef = {};
        sourceRefs.clear();
        sourceRefs.resize(maintenance.size());
        if (!maintenance.empty()) {
            const bool needsScratch = std::any_of(maintenance.begin(), maintenance.end(),
                [](const auto& job) { return job.work.kind != RtWorkKind::Compact; });
            if (needsScratch)
                scratchRef = graph.importBuffer("RT BLAS maintenance scratch", {maintenanceScratchCapacity}, ImportContentsDefined);
            else scratchRef = {};
            if (!f.queried.empty())
                queryRef = graph.importBuffer("RT compact size queries", {f.querySizes->length()}, ImportPerFrame | ImportOutput);
            else queryRef = {};
            for (u32 i = 0; i < maintenance.size(); ++i) {
                const auto& job = maintenance[i];
                if (job.source && job.work.kind != RtWorkKind::Build)
                    sourceRefs[i] = graph.importAccelerationStructure("RT previous BLAS " + std::to_string(job.work.mesh),
                                            job.source->size(), ImportContentsDefined);
            }
            graph.addPass("RT BLAS maintenance", PassType::Compute,
                [&](PassBuilder& b) {
                    b.read(vertexRef, Usage::ShaderRead, StageAccelerationStructure);
                    b.read(indexRef, Usage::ShaderRead, StageAccelerationStructure);
                    if (needsScratch) scratchRef = b.write(scratchRef, Usage::ShaderWrite, StageAccelerationStructure);
                    if (!f.queried.empty()) queryRef = b.write(queryRef, Usage::ShaderWrite, StageAccelerationStructure);
                    for (u32 i = 0; i < maintenance.size(); ++i) {
                        const auto& job = maintenance[i];
                        if (sourceRefs[i].valid()) b.read(sourceRefs[i], Usage::ShaderRead, StageAccelerationStructure);
                        auto& ref = blasRefs[job.work.mesh];
                        if (job.work.kind == RtWorkKind::Refit) b.read(ref, Usage::ShaderRead, StageAccelerationStructure);
                        ref = b.write(ref, Usage::ShaderWrite, StageAccelerationStructure);
                    }
                },
                [this](PassContext& ctx) {
                    auto* e = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                    for (const auto& job : maintenance) {
                        if (job.work.kind == RtWorkKind::Compact) {
                            e->copyAndCompactAccelerationStructure(job.source, job.destination);
                        } else if (job.work.kind == RtWorkKind::Build) {
                            e->buildAccelerationStructure(job.destination, blases[job.work.mesh].descriptor,
                                                          range(maintenanceScratch, job.scratchOffset, job.scratchBytes));
                        } else {
                            e->refitAccelerationStructure(job.source, blases[job.work.mesh].descriptor, job.destination,
                                                          range(maintenanceScratch, job.scratchOffset, job.scratchBytes));
                        }
                    }
                    auto& current = frames[frame.slot];
                    if (!current.queried.empty()) {
                        // Internal build -> size-query ordering within one pass.
                        e->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageAccelerationStructure,
                                                     MTL4::VisibilityOptionDevice);
                        for (auto [mesh, version] : current.queried) {
                            (void)version;
                            e->writeCompactedAccelerationStructureSize(resources.at(ledger.mesh(mesh).resourceID),
                                range(current.querySizes, u64(mesh) * sizeof(u64), sizeof(u64)));
                        }
                    }
                });
        } else { scratchRef = {}; queryRef = {}; }
        graph.addPass("RT instances", PassType::Compute,
            [&](PassBuilder& b) {
                b.read(renderer.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(meshTableRef, Usage::ShaderRead, StageDispatch);
                descriptorRef = b.write(descriptorRef, Usage::ShaderWrite, StageDispatch);
                counterRef = b.write(counterRef, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("rt_clear_counters,rt_write_instances");
            },
            [this](PassContext& ctx) {
                auto* e = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                auto& current = frames[frame.slot];
                dispatch(e, clearPipeline, current.tables[ClearInstances], sizeof(GPURtCounters) / sizeof(u32));
                e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                dispatch(e, instancePipeline, current.tables[Instances], slots);
            });
        graph.addPass("RT TLAS", PassType::Compute,
            [&](PassBuilder& b) {
                b.read(descriptorRef, Usage::ShaderRead, StageAccelerationStructure);
                tlasScratchRef = b.write(tlasScratchRef, Usage::ShaderWrite, StageAccelerationStructure);
                for (auto ref : blasRefs) if (ref.valid()) b.read(ref, Usage::ShaderRead, StageAccelerationStructure);
                // Read-write permits both build and in-place refit with the
                // same compiled graph; physical memory is unique per slot.
                b.read(tlas, Usage::ShaderRead, StageAccelerationStructure);
                tlas = b.write(tlas, Usage::ShaderWrite, StageAccelerationStructure);
            },
            [this](PassContext& ctx) {
                auto* e = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                auto& current = frames[frame.slot];
                if (current.action == RtTlasAction::Build) {
                    e->buildAccelerationStructure(current.tlas, current.descriptor, range(current.scratch));
                    ++stats.tlasBuilds;
                } else if (current.action == RtTlasAction::Refit) {
                    e->refitAccelerationStructure(current.tlas, current.descriptor, current.tlas, range(current.scratch));
                    ++stats.tlasRefits;
                } else throw std::logic_error("RT TLAS pass has no work");
                ledger.commitSnapshot(frame.slot, slots, frame.index, current.action);
                current.recorded = true;
            });
        if (diagnostic) {
            snapshotRef = graph.importBuffer("RT diagnostic instance snapshot", {f.instanceReadback->length()}, ImportPerFrame | ImportOutput);
            materialSnapshotRef = materialCount ? graph.importBuffer("RT diagnostic material snapshot", {f.materialReadback->length()},
                                                                    ImportPerFrame | ImportOutput) : BufferRef{};
            graph.addPass("RT diagnostic snapshots", PassType::Blit,
                [&](PassBuilder& b) {
                    b.read(renderer.dataRef(), Usage::CopySrc, StageBlit);
                    snapshotRef = b.write(snapshotRef, Usage::CopyDst, StageBlit);
                    if (materialSnapshotRef.valid()) materialSnapshotRef = b.write(materialSnapshotRef, Usage::CopyDst, StageBlit);
                },
                [this](PassContext& ctx) {
                    auto* e = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                    auto& current = frames[frame.slot];
                    e->copyFromBuffer(renderer.buffers().instances(), 0, current.instanceReadback, 0,
                                      u64(current.instancesUsed) * sizeof(GPUInstance));
                    if (current.materialsUsed)
                        e->copyFromBuffer(renderer.buffers().materials(), 0, current.materialReadback, 0,
                                          u64(current.materialsUsed) * sizeof(GPUMaterial));
                });
        } else { snapshotRef = {}; materialSnapshotRef = {}; }
        primaryRaysRef = {}; primaryHitsRef = {}; raysRef = {}; hitsRef = {};
        primaryCounterRef = {}; probeCounterRef = {};
        if (!trace) return;
        primaryRaysRef = graph.importBuffer("RT primary rays", {u64(f.rayCapacity) * sizeof(GPURtRay)}, ImportPerFrame | ImportOutput);
        primaryHitsRef = graph.importBuffer("RT primary hits", {u64(f.rayCapacity) * sizeof(GPURtHit)}, ImportPerFrame | ImportOutput);
        primaryCounterRef = graph.importBuffer("RT primary setup counters", {sizeof(GPURtCounters)}, ImportPerFrame | ImportOutput);
        probeCounterRef = graph.importBuffer("RT probe counters", {sizeof(GPURtCounters)}, ImportPerFrame | ImportOutput);
        graph.addPass("RT primary rays", PassType::Compute,
            [&](PassBuilder& b) {
                primaryRaysRef = b.write(primaryRaysRef, Usage::ShaderWrite, StageDispatch);
                primaryCounterRef = b.write(primaryCounterRef, Usage::ShaderWrite, StageDispatch);
                probeCounterRef = b.write(probeCounterRef, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("rt_clear_counters,rt_generate_primary");
            },
            [this](PassContext& ctx) {
                auto* e = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                auto& current = frames[frame.slot];
                dispatch(e, clearPipeline, current.tables[ClearPrimary], sizeof(GPURtCounters) / sizeof(u32));
                dispatch(e, clearPipeline, current.tables[ClearProbe], sizeof(GPURtCounters) / sizeof(u32));
                dispatch(e, generatePrimary, current.tables[GeneratePrimary], rayCount);
            });
        graph.addPass(secondary ? "RT primary setup" : "RT probe primary", PassType::Compute,
            [&](PassBuilder& b) {
                b.read(tlas, Usage::ShaderRead, StageDispatch);
                for (auto ref : blasRefs) if (ref.valid()) b.read(ref, Usage::ShaderRead, StageDispatch);
                b.read(primaryRaysRef, Usage::ShaderRead, StageDispatch);
                b.read(renderer.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(meshTableRef, Usage::ShaderRead, StageDispatch);
                b.read(vertexRef, Usage::ShaderRead, StageDispatch);
                b.read(indexRef, Usage::ShaderRead, StageDispatch);
                auto& counters = secondary ? primaryCounterRef : probeCounterRef;
                b.read(counters, Usage::ShaderRead, StageDispatch);
                counters = b.write(counters, Usage::ShaderWrite, StageDispatch);
                primaryHitsRef = b.write(primaryHitsRef, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("rt_trace_rays");
            },
            [this](PassContext& ctx) {
                dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()), tracePipeline,
                         frames[frame.slot].tables[TracePrimary], rayCount);
            });
        if (!secondary) {
            raysRef = primaryRaysRef;
            hitsRef = primaryHitsRef;
            return;
        }
        raysRef = graph.importBuffer("RT secondary rays", {u64(f.rayCapacity) * sizeof(GPURtRay)}, ImportPerFrame | ImportOutput);
        hitsRef = graph.importBuffer("RT secondary hits", {u64(f.rayCapacity) * sizeof(GPURtHit)}, ImportPerFrame | ImportOutput);
        graph.addPass("RT secondary rays", PassType::Compute,
            [&](PassBuilder& b) {
                b.read(primaryRaysRef, Usage::ShaderRead, StageDispatch);
                b.read(primaryHitsRef, Usage::ShaderRead, StageDispatch);
                b.read(renderer.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(meshTableRef, Usage::ShaderRead, StageDispatch);
                b.read(vertexRef, Usage::ShaderRead, StageDispatch);
                b.read(indexRef, Usage::ShaderRead, StageDispatch);
                raysRef = b.write(raysRef, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("rt_generate_secondary");
            },
            [this](PassContext& ctx) {
                dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()), generateSecondary,
                         frames[frame.slot].tables[GenerateSecondary], rayCount);
            });
        graph.addPass(std::string("RT probe ") + probeName(selectedProbe), PassType::Compute,
            [&](PassBuilder& b) {
                b.read(tlas, Usage::ShaderRead, StageDispatch);
                for (auto ref : blasRefs) if (ref.valid()) b.read(ref, Usage::ShaderRead, StageDispatch);
                b.read(raysRef, Usage::ShaderRead, StageDispatch);
                b.read(renderer.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(meshTableRef, Usage::ShaderRead, StageDispatch);
                b.read(vertexRef, Usage::ShaderRead, StageDispatch);
                b.read(indexRef, Usage::ShaderRead, StageDispatch);
                b.read(probeCounterRef, Usage::ShaderRead, StageDispatch);
                probeCounterRef = b.write(probeCounterRef, Usage::ShaderWrite, StageDispatch);
                hitsRef = b.write(hitsRef, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("rt_trace_rays");
            },
            [this](PassContext& ctx) {
                dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()), tracePipeline,
                         frames[frame.slot].tables[TraceFinal], rayCount);
            });
    }

    rg::TextureRef addPresent(rg::RenderGraph& graph, rg::TextureRef target, rg::Format format) {
        using namespace rg;
        if (!enabled || !trace || !debugView) return target;
        presentPipeline = requestPresent(format);
        graph.addPass("RT debug image", PassType::Compute,
            [&](PassBuilder& b) {
                b.read(hitsRef, Usage::ShaderRead, StageDispatch);
                b.read(renderer.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(meshTableRef, Usage::ShaderRead, StageDispatch);
                b.read(vertexRef, Usage::ShaderRead, StageDispatch);
                b.read(indexRef, Usage::ShaderRead, StageDispatch);
                debugTexture = b.createTexture("RT debug linear color", {Format::RGBA16Float, frame.width, frame.height});
                debugTexture = b.write(debugTexture, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("rt_debug_view");
            },
            [this](PassContext& ctx) {
                auto* color = static_cast<MTL::Texture*>(ctx.texture(debugTexture));
                auto* table = frames[frame.slot].tables[Debug];
                table->setTexture(color->gpuResourceID(), 0);
                dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()), debugPipeline, table, rayCount);
            });
        graph.addPass("RT present", PassType::Raster,
            [&](PassBuilder& b) {
                b.read(debugTexture, Usage::ShaderRead, StageFragment);
                target = b.writeColor(target, 0, LoadIntent::Clear);
                b.setProfileShaders("rt_present_vs,rt_present_fs");
            },
            [this](PassContext& ctx) {
                auto* e = static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());
                auto* pipeline = pipelines.render(presentPipeline);
                if (!pipeline) throw std::runtime_error("RT present pipeline not ready");
                auto* table = frames[frame.slot].tables[Present];
                table->setTexture(static_cast<MTL::Texture*>(ctx.texture(debugTexture))->gpuResourceID(), 0);
                e->setRenderPipelineState(pipeline);
                e->setArgumentTable(table, MTL::RenderStageFragment);
                e->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
            });
        return target;
    }

    void bind(MetalGraphExecutor& executor) {
        if (!enabled) return;
        auto& f = frames[frame.slot];
        executor.bindBuffer(descriptorRef, f.descriptors);
        executor.bindBuffer(meshTableRef, f.meshTable);
        executor.bindBuffer(vertexRef, renderer.vertexBuffer());
        executor.bindBuffer(indexRef, indexBuffer);
        executor.bindBuffer(counterRef, f.counters);
        executor.bindAccelerationStructure(tlas, f.tlas);
        executor.bindBuffer(tlasScratchRef, f.scratch);
        for (u32 mesh = 0; mesh < blasRefs.size(); ++mesh)
            if (blasRefs[mesh].valid())
                executor.bindAccelerationStructure(blasRefs[mesh], resources.at(ledger.mesh(mesh).resourceID));
        for (u32 i = 0; i < sourceRefs.size(); ++i)
            if (sourceRefs[i].valid()) executor.bindAccelerationStructure(sourceRefs[i], maintenance[i].source);
        if (uploadRef.valid()) executor.bindBuffer(uploadRef, vertexUpload.buffer);
        if (scratchRef.valid()) executor.bindBuffer(scratchRef, maintenanceScratch);
        if (queryRef.valid()) executor.bindBuffer(queryRef, f.querySizes);
        if (snapshotRef.valid()) executor.bindBuffer(snapshotRef, f.instanceReadback);
        if (materialSnapshotRef.valid()) executor.bindBuffer(materialSnapshotRef, f.materialReadback);
        if (!trace) return;
        executor.bindBuffer(primaryRaysRef, f.primaryRays);
        executor.bindBuffer(primaryHitsRef, f.primaryHits);
        executor.bindBuffer(primaryCounterRef, f.primaryCounters);
        executor.bindBuffer(probeCounterRef, f.probeCounters);
        if (secondary) {
            executor.bindBuffer(raysRef, f.rays);
            executor.bindBuffer(hitsRef, f.hits);
        }
    }
};

AccelerationStructures::AccelerationStructures(MetalContext& c, PipelineCache& p, SceneRenderer& r, const LaunchOptions& o)
    : impl_(std::make_unique<Impl>(c, p, r, o)) {}
AccelerationStructures::~AccelerationStructures() = default;
void AccelerationStructures::loadScene(const GpuScene& scene, const SceneStore& store, MetalTextureManager& textures) {
    impl_->load(scene, store, textures);
}
void AccelerationStructures::clear() { impl_->clear(); }
void AccelerationStructures::prepareFrame(const SceneStore& store, const FrameParams& params) { impl_->prepare(store, params); }
void AccelerationStructures::addPassesToGraph(rg::RenderGraph& graph) { impl_->addPasses(graph); }
rg::TextureRef AccelerationStructures::addDebugPresent(rg::RenderGraph& graph, rg::TextureRef target, rg::Format format) {
    return impl_->addPresent(graph, target, format);
}
void AccelerationStructures::bindResources(MetalGraphExecutor& executor) { impl_->bind(executor); }
u64 AccelerationStructures::version() const { return impl_->graphVersion; }
bool AccelerationStructures::active() const { return impl_->enabled; }
bool AccelerationStructures::hasTrace() const { return impl_->enabled && impl_->trace; }
bool AccelerationStructures::maintenancePending() const { return !impl_->maintenance.empty() || impl_->ledger.hasWork(); }
rg::AccelerationStructureRef AccelerationStructures::tlasRef() const { return impl_->tlas; }
rg::BufferRef AccelerationStructures::geometryBufferRef() const { return impl_->vertexRef; }

AccelerationStructures::TraceResources AccelerationStructures::traceResources(u32 slot) const {
    const auto& f = impl_->frames.at(slot);
    if (!impl_->enabled || !f.tlas || !impl_->textures) return {};
    return {f.tlas, impl_->renderer.buffers().materials(), impl_->textures->tableBuffer(),
            impl_->renderer.vertexBuffer(), impl_->indexBuffer, impl_->renderer.buffers().instances(),
            f.meshTable, f.params};
}
void AccelerationStructures::declareTraceReads(rg::PassBuilder& b) const {
    using namespace rg;
    if (!impl_->enabled || !impl_->tlas.valid()) throw std::logic_error("RT consumer has no graph TLAS");
    b.read(impl_->tlas, Usage::ShaderRead, StageDispatch);
    for (auto ref : impl_->blasRefs) if (ref.valid()) b.read(ref, Usage::ShaderRead, StageDispatch);
    b.read(impl_->renderer.dataRef(), Usage::ShaderRead, StageDispatch);
    b.read(impl_->vertexRef, Usage::ShaderRead, StageDispatch);
    b.read(impl_->indexRef, Usage::ShaderRead, StageDispatch);
    b.read(impl_->meshTableRef, Usage::ShaderRead, StageDispatch);
}

AccelerationStructures::Readback AccelerationStructures::readback(u32 slot) const {
    const auto& f = impl_->frames.at(slot);
    if (!f.recorded || !f.check || !f.trace || impl_->context.frameEvent()->signaledValue() <= f.frame) return {};
    const auto* rays = static_cast<const GPURtRay*>((f.secondary ? f.rays : f.primaryRays)->contents());
    const auto* hits = static_cast<const GPURtHit*>((f.secondary ? f.hits : f.primaryHits)->contents());
    Readback out;
    out.valid = true;
    out.frame = f.frame;
    out.width = f.width; out.height = f.height; out.probeType = f.type;
    out.rays = {rays, f.raysUsed};
    out.hits = {hits, f.raysUsed};
    out.instances = {static_cast<const GPUInstance*>(f.instanceReadback->contents()), f.instancesUsed};
    if (f.materialsUsed) out.materials = {static_cast<const GPUMaterial*>(f.materialReadback->contents()), f.materialsUsed};
    if (f.geometrySnapshot) out.vertices = *f.geometrySnapshot;
    out.geometryRevision = f.geometryRevision;
    return out;
}
GPURtCounters AccelerationStructures::counters(u32 slot) const { return impl_->frameCounters(slot); }
RtReport AccelerationStructures::report() const {
    impl_->collectCompleted();
    auto out = impl_->stats;
    out.blasBytes = 0;
    for (const auto& [id, as] : impl_->resources) { (void)id; out.blasBytes += as->allocatedSize(); }
    out.compactedCount = 0;
    for (const auto& m : impl_->ledger.meshes()) if (m.state == RtBlasState::Compacted) ++out.compactedCount;
    out.tlasBytes = out.tlasScratchBytes = 0;
    for (const auto& f : impl_->frames) {
        if (f.tlas) out.tlasBytes += f.tlas->allocatedSize();
        if (f.scratch) out.tlasScratchBytes += f.scratch->length();
    }
    return out;
}
MTL::Buffer* AccelerationStructures::rayBuffer(u32 slot) const {
    const auto& f = impl_->frames.at(slot); return f.secondary ? f.rays : f.primaryRays;
}
MTL::Buffer* AccelerationStructures::hitBuffer(u32 slot) const {
    const auto& f = impl_->frames.at(slot); return f.secondary ? f.hits : f.primaryHits;
}
MTL::Buffer* AccelerationStructures::primaryRayBuffer(u32 slot) const { return impl_->frames.at(slot).primaryRays; }
MTL::Buffer* AccelerationStructures::primaryHitBuffer(u32 slot) const { return impl_->frames.at(slot).primaryHits; }
u32 AccelerationStructures::rayCount(u32 slot) const { return impl_->frames.at(slot).raysUsed; }
rg::BufferRef AccelerationStructures::rayRef() const { return impl_->raysRef; }
rg::BufferRef AccelerationStructures::hitRef() const { return impl_->hitsRef; }
rg::BufferRef AccelerationStructures::primaryRayRef() const { return impl_->primaryRaysRef; }
rg::BufferRef AccelerationStructures::primaryHitRef() const { return impl_->primaryHitsRef; }
std::span<const GPURtMesh> AccelerationStructures::meshes() const { return impl_->meshTable; }
const RtProxyGeometry& AccelerationStructures::geometry() const { return impl_->proxy; }
std::span<const GPUVertex> AccelerationStructures::vertices() const { return *impl_->cpuVertices; }
u64 AccelerationStructures::geometryRevision() const { return impl_->geometryVersion; }

void AccelerationStructures::requestRefit(u32 mesh, u64 revision) {
    const auto& m = impl_->ledger.mesh(mesh);
    if (!m.requested) throw std::invalid_argument("RT refit mesh is not built");
    impl_->ledger.request(mesh, m.topologyRevision, revision);
}
void AccelerationStructures::requestRebuild(u32 mesh) { impl_->ledger.requestRebuild(mesh); }
void AccelerationStructures::updateVertices(u32 mesh, std::span<const GPUVertex> vertices, u64 revision, bool forceRebuild) {
    auto& state = *impl_;
    const auto& blas = state.blases.at(mesh);
    if (!blas.descriptor || vertices.size() != blas.vertexCount)
        throw std::invalid_argument("RT vertex update must preserve the mesh vertex range");
    if (state.proxy.selections[mesh].level != RtProxyLevel::Full)
        throw std::invalid_argument("RT proxy deformation has no validated error; reload full geometry before updating");
    const auto& record = state.ledger.mesh(mesh);
    if (revision == record.vertexRevision || revision == record.requestedVertices)
        throw std::invalid_argument("RT vertex update needs a new content revision");
    for (const auto& v : vertices)
        if (!std::isfinite(v.px) || !std::isfinite(v.py) || !std::isfinite(v.pz))
            throw std::invalid_argument("RT vertex update contains nonfinite position");
    // Retain old CPU snapshots for diagnostics whose GPU frame is still in flight.
    if (state.cpuVertices.use_count() != 1) state.cpuVertices = std::make_shared<std::vector<GPUVertex>>(*state.cpuVertices);
    const auto offset = state.meshTable[mesh].vertexOffset;
    std::memmove(state.cpuVertices->data() + offset, vertices.data(), vertices.size_bytes());
    const Impl::VertexUpdate update{mesh, u64(offset) * sizeof(GPUVertex), vertices.size_bytes(), 0};
    auto it = std::find_if(state.pendingVertexUpdates.begin(), state.pendingVertexUpdates.end(),
                           [&](const auto& old) { return old.mesh == mesh; });
    if (it == state.pendingVertexUpdates.end()) state.pendingVertexUpdates.push_back(update);
    else *it = update;
    state.ledger.request(mesh, record.topologyRevision, revision);
    if (forceRebuild) state.ledger.requestRebuild(mesh);
    ++state.geometryVersion;
}

} // namespace phosphor
