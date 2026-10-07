// F9-K1: deterministic regression for the actual engine RT kernels. Run only
// through the soc harness; no GPU work may overlap another engine/benchmark.
#include "f9_common.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstring>
#include <limits>

namespace f9 {
namespace {
using namespace phosphor;
constexpr u32 kSize = 32, kRays = kSize * kSize, kSlots = 7;
static_assert(sizeof(GPURtInstanceDesc) == sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
static_assert(sizeof(MTL::ResourceID) == 8);

struct Kernels {
    explicit Kernels(::soc::Context& context) : ctx(context) {}
    ::soc::Context& ctx;
    MTL4::ArgumentTable* table = nullptr;
    MTL::ComputePipelineState *clear = nullptr, *descriptors = nullptr, *trace = nullptr, *primary = nullptr, *secondary = nullptr;
    MTL::IntersectionFunctionTable* ift = nullptr;
    MTL::Buffer *instances = nullptr, *materials = nullptr, *meshes = nullptr, *params = nullptr, *probe = nullptr;
    MTL::Buffer *rays = nullptr, *hits = nullptr, *counters = nullptr, *primaryRays = nullptr, *primaryHits = nullptr;
    MTL::Buffer* textureTable = nullptr;
    GpuGeometry geometry;
    Tlas tlas;
    GPURtParams p{kSlots, 1, 2, 0};
    GPURtProbeParams q{};

    void dispatch(MTL4::ComputeCommandEncoder* e, MTL::ComputePipelineState* ps, u32 n) {
        e->setComputePipelineState(ps);
        e->setArgumentTable(table);
        e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(64, 1, 1));
    }
    void upload() {
        std::memcpy(params->contents(), &p, sizeof p);
        std::memcpy(probe->contents(), &q, sizeof q);
    }
    void build() {
        upload();
        timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
            table->setAddress(counters->gpuAddress(), 0);
            dispatch(e, clear, 8);
            e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
            table->setAddress(instances->gpuAddress(), 0);
            table->setAddress(materials->gpuAddress(), 1);
            table->setAddress(meshes->gpuAddress(), 2);
            table->setAddress(tlas.instances->gpuAddress(), 3);
            table->setAddress(params->gpuAddress(), 4);
            table->setAddress(counters->gpuAddress(), 5);
            dispatch(e, descriptors, kSlots);
            e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageAccelerationStructure, MTL4::VisibilityOptionDevice);
            e->buildAccelerationStructure(tlas.as, tlas.desc, range(tlas.scratch));
            e->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        });
    }
    double runTrace() {
        upload();
        return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
            table->setResource(tlas.as->gpuResourceID(), 0);
            table->setAddress(rays->gpuAddress(), 1);
            table->setAddress(hits->gpuAddress(), 2);
            table->setAddress(probe->gpuAddress(), 3);
            table->setResource(ift->gpuResourceID(), 4);
            table->setAddress(instances->gpuAddress(), 5);
            table->setAddress(counters->gpuAddress(), 9);
            dispatch(e, trace, kRays);
        });
    }
};

void engineKernels(::soc::Context& ctx, ::soc::Report& rep) {
    Kernels k{ctx};
    SceneData scene;
    {
        QuietStderr quiet;
        addMesh(scene, {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, {0, 1, 2, 0, 2, 3},
                 {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    }
    k.geometry = uploadGeometry(ctx, scene);
    BlasOptions options;
    options.opaque = false; // same geometry shared by opaque and MASK instances
    auto blases = allocateBlases(ctx, scene, k.geometry, options);
    buildBlases(ctx, blases, scratchFor(ctx, blases));
    k.tlas = allocateTlas(ctx, kSlots, MTL::AccelerationStructureUsageNone, AsPlacement::Device);
    k.instances = ctx.buffer(kSlots * sizeof(GPUInstance));
    k.materials = ctx.buffer(2 * sizeof(GPUMaterial));
    k.meshes = ctx.buffer(sizeof(GPURtMesh));
    k.params = ctx.buffer(sizeof(GPURtParams));
    k.probe = ctx.buffer(sizeof(GPURtProbeParams));
    k.rays = ctx.buffer(kRays * sizeof(GPURtRay));
    k.hits = ctx.buffer(kRays * sizeof(GPURtHit));
    k.primaryRays = ctx.buffer(kRays * sizeof(GPURtRay));
    k.primaryHits = ctx.buffer(kRays * sizeof(GPURtHit));
    k.counters = ctx.buffer(sizeof(GPURtCounters));
    k.textureTable = ctx.buffer(sizeof(MTL::ResourceID));

    auto* instances = static_cast<GPUInstance*>(k.instances->contents());
    for (u32 i = 0; i < kSlots; ++i) {
        instances[i] = {};
        std::memcpy(instances[i].modelMatrix, glm::value_ptr(glm::mat4(1)), sizeof instances[i].modelMatrix);
        instances[i].generation = 100u + i;
        instances[i].flags = INSTANCE_FLAG_VALID | 3u;
    }
    instances[0].modelMatrix[14] = -1.0f; // opaque backing plane
    instances[1].materialIndex = 1;       // checker MASK front plane
    instances[2].modelMatrix[0] = -1.0f;
    instances[2].modelMatrix[12] = 3.0f;
    instances[2].flags |= INSTANCE_FLAG_MIRRORED;
    instances[3].flags = 0;              // dead slot with intentionally unusable fields
    instances[3].meshIndex = ~0u;
    std::memset(instances[3].modelMatrix, 0, sizeof instances[3].modelMatrix);
    instances[4].meshIndex = ~0u;        // invalid live mesh
    instances[5].modelMatrix[12] = -3.0f;
    instances[5].flags = INSTANCE_FLAG_VALID | 2u; // shadow-only instance
    instances[6].modelMatrix[0] = std::numeric_limits<float>::quiet_NaN();
    auto* materials = static_cast<GPUMaterial*>(k.materials->contents());
    for (u32 i = 0; i < 2; ++i) {
        materials[i] = {};
        materials[i].baseColor[3] = 1;
        materials[i].baseColorTex = INVALID_TEXTURE_INDEX;
    }
    materials[1].alphaCutoff = 0.5f;
    materials[1].baseColorTex = 0;
    GPURtMesh mesh{};
    const u64 resource = blases[0].as->gpuResourceID()._impl;
    mesh.blasLo = u32(resource);
    mesh.blasHi = u32(resource >> 32);
    mesh.indexCount = 6;
    mesh.flags = RT_MESH_ALPHA;
    std::memcpy(k.meshes->contents(), &mesh, sizeof mesh);

    auto* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kSize, kSize, true);
    td->setStorageMode(MTL::StorageModeShared);
    td->setUsage(MTL::TextureUsageShaderRead);
    auto* texture = ctx.texture(td);
    std::vector<::soc::u8> pixels(kRays * 4, 255);
    auto opaque = [](u32 x, u32 y) { return ((x / 4u) ^ (y / 4u)) & 1u; };
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) pixels[(y * kSize + x) * 4u + 3u] = opaque(x, y) ? 255 : 0;
    texture->replaceRegion(MTL::Region::Make2D(0, 0, kSize, kSize), 0, pixels.data(), kSize * 4);
    // Deliberately distinct mips prove that the primary cone is consumed and
    // shadow rays stay at LOD0 even when they carry a nonzero cone width.
    for (u32 level = 1, side = kSize / 2; side > 0; ++level, side /= 2) {
        const std::vector<::soc::u8> mip(side * side * 4, level == 1 ? 0 : 255);
        texture->replaceRegion(MTL::Region::Make2D(0, 0, side, side), level, mip.data(), side * 4);
    }
    const auto textureID = texture->gpuResourceID();
    std::memcpy(k.textureTable->contents(), &textureID, sizeof textureID);
    auto* ad = MTL4::ArgumentTableDescriptor::alloc()->init();
    ad->setMaxBufferBindCount(12);
    ad->setMaxTextureBindCount(1);
    NS::Error* error = nullptr;
    k.table = ctx.device()->newArgumentTable(ad, &error);
    ad->release();
    if (!k.table) throw ::soc::BenchError("F9-K1 argument table unavailable");
    ctx.keep(k.table);
    auto* lib = f9Library(ctx, "s6_engine.metal");
    k.clear = ctx.compute(lib, "rt_clear_counters");
    k.descriptors = ctx.compute(lib, "rt_write_instances");
    k.primary = ctx.compute(lib, "rt_generate_primary");
    k.secondary = ctx.compute(lib, "rt_generate_secondary");
    k.trace = linkedPipeline(ctx, lib, "rt_trace_rays", {"rt_alpha_generic"});
    auto* id = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
    id->setFunctionCount(1);
    k.ift = k.trace->newIntersectionFunctionTable(id);
    id->release();
    if (!k.ift) throw ::soc::BenchError("F9-K1 intersection function table unavailable");
    ctx.adopt(k.ift);
    auto* handle = k.trace->functionHandle(NS::String::string("rt_alpha_generic", NS::UTF8StringEncoding));
    if (!handle) throw ::soc::BenchError("F9-K1 alpha function handle unavailable");
    k.ift->setFunction(handle, 0);
    k.ift->setBuffer(k.materials, 0, 0);
    k.ift->setBuffer(k.textureTable, 0, 1);
    k.ift->setBuffer(k.geometry.vertices, 0, 2);
    k.ift->setBuffer(k.geometry.indices, 0, 3);
    k.ift->setBuffer(k.instances, 0, 4);
    k.ift->setBuffer(k.meshes, 0, 5);
    k.ift->setBuffer(k.params, 0, 6);
    ctx.commitResidency();
    k.q.width = kSize; k.q.height = kSize; k.q.rayCount = kRays;
    k.q.slotCount = kSlots; k.q.meshCount = 1; k.q.flags = 1;
    k.build();

    u32 wrongDescriptors = 0;
    const auto* desc = static_cast<const GPURtInstanceDesc*>(k.tlas.instances->contents());
    for (u32 i = 0; i < kSlots; ++i) {
        const bool valid = i < 3 || i == 5;
        wrongDescriptors += desc[i].userID != i || desc[i].blasLo != mesh.blasLo || desc[i].blasHi != mesh.blasHi;
        wrongDescriptors += desc[i].mask != (valid ? (i == 5 ? RT_MASK_SHADOW : RT_MASK_ALL) : 0u);
        wrongDescriptors += desc[i].intersectionFunctionTableOffset != 0u;
        if (!valid) {
            for (u32 f = 0; f < 12; ++f) wrongDescriptors += desc[i].transform[f] != ((f == 0 || f == 4 || f == 8) ? 1.0f : 0.0f);
        } else {
            const u32 expectedOptions = (i == 2 ? 2u : 0u) | (i == 1 ? 0u : 4u);
            wrongDescriptors += desc[i].options != expectedOptions;
        }
    }
    const auto counts = *static_cast<const GPURtCounters*>(k.counters->contents());
    wrongDescriptors += counts.activeInstances != 4 || counts.maskedInstances != 3 || counts.invalidMesh != 1;

    auto* rays = static_cast<GPURtRay*>(k.rays->contents());
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) {
            GPURtRay& r = rays[y * kSize + x];
            r = {};
            r.ox = -1 + 2 * (float(x) + 0.5f) / kSize;
            r.oy = -1 + 2 * (float(y) + 0.5f) / kSize;
            r.oz = 2;
            r.dz = -1; r.tmax = 10;
            r.mask = RT_MASK_PRIMARY; r.type = RT_PROBE_PRIMARY;
        }
    const double primaryMs = k.runTrace();
    const auto* hits = static_cast<const GPURtHit*>(k.hits->contents());
    u32 primaryWrong = 0;
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) {
            const GPURtHit h = hits[y * kSize + x];
            const u32 slot = opaque(x, y) ? 1 : 0;
            primaryWrong += h.hit != 1 || h.slot != slot || h.generation != 100u + slot || std::abs(h.t - (slot ? 2.0f : 3.0f)) > 1e-5f;
        }
    const GPURtCounters primaryCounts = *static_cast<const GPURtCounters*>(k.counters->contents());
    primaryWrong += primaryCounts.alphaTests == 0 || primaryCounts.opaqueAlphaTests != 0;
    std::memcpy(k.primaryRays->contents(), rays, kRays * sizeof(GPURtRay));
    std::memcpy(k.primaryHits->contents(), hits, kRays * sizeof(GPURtHit));
    for (u32 i = 0; i < kRays; ++i) rays[i].coneWidth = 0.125f; // t=2, 32px across 2m: LOD2
    k.runTrace();
    u32 coneWrong = 0;
    for (u32 i = 0; i < kRays; ++i) coneWrong += hits[i].hit != 1 || hits[i].slot != 1;
    for (u32 i = 0; i < kRays; ++i) { rays[i].mask = RT_MASK_SHADOW; rays[i].type = RT_PROBE_SHADOW; rays[i].tmax = 2.5f; }
    const double shadowMs = k.runTrace();
    u32 shadowWrong = 0;
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) shadowWrong += hits[y * kSize + x].hit != opaque(x, y);

    // Descriptor corruptions must disagree with the known plane visibility;
    // malformed inactive slots must never become traversable.
    u32 caught = 0;
    for (u32 corruption : {RT_CORRUPT_TRANSFORM, RT_CORRUPT_MASK, RT_CORRUPT_BLAS}) {
        k.p.corruption = corruption;
        k.build();
        k.runTrace();
        u32 mismatches = 0;
        for (u32 y = 0; y < kSize; ++y)
            for (u32 x = 0; x < kSize; ++x) mismatches += hits[y * kSize + x].hit != opaque(x, y);
        caught += mismatches > 0;
        rep.value("negative." + std::to_string(corruption) + ".mismatches", "rays", mismatches, {}, true);
    }
    k.p.corruption = RT_CORRUPT_NONE;
    k.build();

    // Secondary origin correctness: outgoing +Z shadows revisit the checker
    // at exactly the primary UV; holes stay open and no plane self-intersects.
    k.q.probeType = RT_PROBE_SHADOW;
    k.q.lightDirection[2] = 1;
    k.q.lightDirection[3] = 10;
    k.upload();
    timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        k.table->setAddress(k.rays->gpuAddress(), 1);
        k.table->setAddress(k.probe->gpuAddress(), 3);
        k.table->setAddress(k.instances->gpuAddress(), 5);
        k.table->setAddress(k.meshes->gpuAddress(), 6);
        k.table->setAddress(k.geometry.vertices->gpuAddress(), 7);
        k.table->setAddress(k.geometry.indices->gpuAddress(), 8);
        k.table->setAddress(k.primaryHits->gpuAddress(), 10);
        k.table->setAddress(k.primaryRays->gpuAddress(), 11);
        k.dispatch(e, k.secondary, kRays);
    });
    k.runTrace();
    u32 offsetWrong = 0;
    for (u32 i = 0; i < kRays; ++i) offsetWrong += hits[i].hit != 0 || rays[i].dz != 1 || rays[i].tmax != 10;

    // Primary generator must use the finite reverse-Z near plane. Center rays
    // are compared against the same inverse projection in CPU arithmetic.
    const glm::mat4 vp = glm::perspectiveRH_ZO(glm::radians(60.0f), 1.0f, 100.0f, 0.1f) *
                         glm::lookAtRH(glm::vec3(0, 0, 2), glm::vec3(0), glm::vec3(0, 1, 0));
    const glm::mat4 inv = glm::inverse(vp);
    std::memcpy(k.q.inverseViewProjection, glm::value_ptr(inv), sizeof k.q.inverseViewProjection);
    k.q.cameraPosition[2] = 2;
    k.upload();
    timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        k.table->setAddress(k.rays->gpuAddress(), 1);
        k.table->setAddress(k.probe->gpuAddress(), 3);
        k.dispatch(e, k.primary, kRays);
    });
    u32 generatedWrong = 0;
    for (u32 y = 0; y < kSize; ++y)
        for (u32 x = 0; x < kSize; ++x) {
            const glm::vec2 pixel(x + 0.5f, y + 0.5f);
            const glm::vec2 ndc = glm::vec2(2, -2) * pixel / float(kSize) + glm::vec2(-1, 1);
            const glm::vec4 near = inv * glm::vec4(ndc, 1, 1);
            const glm::vec3 d = glm::normalize(glm::vec3(near) / near.w - glm::vec3(0, 0, 2));
            const GPURtRay r = rays[y * kSize + x];
            generatedWrong += glm::length(d - glm::vec3(r.dx, r.dy, r.dz)) > 1e-5f || !(r.coneWidth > 0 && r.coneWidth < 0.1f);
        }
    const bool pass = wrongDescriptors == 0 && primaryWrong == 0 && coneWrong == 0 && shadowWrong == 0 && offsetWrong == 0 && generatedWrong == 0 && caught == 3;
    rep.value("descriptors.wrong", "fields", wrongDescriptors, {}, false);
    rep.value("primary.wrong", "rays", primaryWrong, {{"rays", kRays}}, false);
    rep.value("primary_cone.wrong", "rays", coneWrong, {{"rays", kRays}}, false);
    rep.value("shadow.wrong", "rays", shadowWrong, {{"rays", kRays}}, false);
    rep.value("offset.self_hits_or_invalid", "rays", offsetWrong, {}, false);
    rep.value("primary_generated.wrong", "rays", generatedWrong, {}, false);
    rep.value("primary.ms", "ms", primaryMs, {}, false);
    rep.value("shadow.ms", "ms", shadowMs, {}, false);
    rep.value("alpha.tests", "calls", primaryCounts.alphaTests, {}, false);
    rep.value("alpha.opaque_tests", "calls", primaryCounts.opaqueAlphaTests, {}, false);
    rep.note("F9-K1 is a correctness test, not a throughput benchmark. Actual engine rt_scene.metal; primary LOD0 analytic checker, primary generator cone, shadow LOD0, opaque IFT bypass, descriptor safety and world W&B offset. Device family is an effective capability selection only.");
    if (!pass) rep.status(::soc::Status::Failed, "engine RT kernel regression failed");
    rep.negative(pass, "all positive checks passed and all three descriptor corruptions changed the known checker visibility");
}
} // namespace
SOC_BENCH("F9-K1", "engine_rt_kernels", "Engine RT descriptors, alpha traversal, camera/secondary rays and negative controls", engineKernels);
} // namespace f9
