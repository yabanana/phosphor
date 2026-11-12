// F14-K2: the engine's brute-force DI shading kernel (light_cluster_shade_rt)
// in isolation, run repeatedly on identical inputs.
//
// The F14.4 day/night pair diverges under MTL_SHADER_VALIDATION only through
// this kernel's RT visibility trace (RESOURCE_USAGE instrumentation), losing
// the low components of the result for whole SIMD groups with no validation
// fault (docs/implementation/F14.4-VALIDATION-NONDETERMINISM-2026-11-12.md).
// F14-K1 (a minimal accumulator across traces) does not reproduce it.  This
// spike feeds the real kernel synthetic but engine-shaped inputs: receivers on
// a floor, the twelve triangle lights of an emissive cube with the moving-light
// colour, one occluding quad in the TLAS, the engine dispatch shape (1D, eight
// threads per threadgroup).  Each frame index is dispatched kRepeats times and
// compared bit for bit; without shader instrumentation every repeat must match.
#include "f9_common.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace f9 {
namespace {
using namespace phosphor;
constexpr u32 kWidth = 640, kHeight = 360, kThreads = kWidth * kHeight, kFrames = 8, kRepeats = 12;

void put3(float* dst, glm::vec3 v) { dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; }

void engineDi(::soc::Context& ctx, ::soc::Report& rep) {
    SceneData scene;
    {
        QuietStderr quiet;
        addMesh(scene, {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, {0, 1, 2, 0, 2, 3},
                {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    }
    GpuGeometry geometry = uploadGeometry(ctx, scene);
    BlasOptions options;
    options.opaque = true;
    auto blases = allocateBlases(ctx, scene, geometry, options);
    buildBlases(ctx, blases, scratchFor(ctx, blases));
    Tlas tlas = allocateTlas(ctx, 1, MTL::AccelerationStructureUsageNone, AsPlacement::Device);

    // Occluder: the quad turned horizontal at y = 1, covering x in [-1.6, 0.4].
    GPUInstance instance{};
    glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(-0.6f, 1.0f, 0.0f)) *
                      glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1, 0, 0)) * glm::scale(glm::mat4(1), glm::vec3(1.0f, 2.0f, 1.0f));
    std::memcpy(instance.modelMatrix, glm::value_ptr(model), sizeof instance.modelMatrix);
    instance.generation = 1;
    instance.flags = INSTANCE_FLAG_VALID | 3u;
    GPUMaterial material{};
    material.baseColor[3] = 1;
    material.baseColorTex = INVALID_TEXTURE_INDEX;
    material.emissiveTex = INVALID_TEXTURE_INDEX;
    material.emissive[0] = 8.0f; material.emissive[1] = 0.4f; material.emissive[2] = 0.2f;
    GPURtMesh mesh{};
    const u64 resource = blases[0].as->gpuResourceID()._impl;
    mesh.blasLo = u32(resource);
    mesh.blasHi = u32(resource >> 32);
    mesh.indexCount = 6;
    const GPURtParams rp{1, 1, 1, 0};

    // The emissive cube: centre (0, 2, 0), half size 0.25, twelve CCW triangles.
    const glm::vec3 c(0, 2, 0);
    const float h = 0.25f;
    std::array<glm::vec3, 8> v{};
    for (u32 i = 0; i < 8; ++i) v[i] = c + glm::vec3((i & 1) ? h : -h, (i & 2) ? h : -h, (i & 4) ? h : -h);
    const u32 tris[12][3] = {{0, 2, 1}, {1, 2, 3}, {4, 5, 6}, {5, 7, 6}, {0, 1, 4}, {1, 5, 4},
                             {2, 6, 3}, {3, 6, 7}, {0, 4, 2}, {2, 4, 6}, {1, 3, 5}, {3, 7, 5}};
    std::vector<GPUSampledLight> lights(12);
    std::vector<GPUEmissiveSurface> emitters(12);
    for (u32 i = 0; i < 12; ++i) {
        const glm::vec3 a = v[tris[i][0]], b = v[tris[i][1]], d = v[tris[i][2]];
        GPUSampledLight& l = lights[i];
        l = {};
        l.id = i; l.generation = 1; l.type = DI_LIGHT_TRIANGLE; l.flags = DI_LIGHT_TEXTURED_EMISSION;
        put3(l.position, a); put3(l.axisU, b - a); put3(l.axisV, d - a);
        put3(l.emission, glm::vec3(8.0f, 0.4f, 0.2f));
        l.range = 1e6f;
        GPUEmissiveSurface& e = emitters[i];
        e = {};
        put3(e.p0, a); put3(e.p1, b); put3(e.p2, d);
        e.valid = 1;
    }
    std::vector<GPUDISurface> surfaces(kThreads);
    for (u32 y = 0; y < kHeight; ++y)
        for (u32 x = 0; x < kWidth; ++x) {
            GPUDISurface& s = surfaces[y * kWidth + x];
            s = {};
            put3(s.position, glm::vec3(-3.0f + 6.0f * (float(x) + 0.5f) / kWidth, 0.0f, -2.0f + 4.0f * (float(y) + 0.5f) / kHeight));
            s.depth = 4.0f;
            put3(s.geometricNormal, glm::vec3(0, 1, 0));
            put3(s.shadingNormal, glm::vec3(0, 1, 0));
            s.roughness = 0.5f;
            put3(s.albedo, glm::vec3(0.75f, 0.1f, 0.06f));
            put3(s.viewDirection, glm::normalize(glm::vec3(0.0f, 0.6f, 0.8f)));
            s.valid = 1;
        }

    auto bufferOf = [&](const void* data, size_t bytes) {
        auto* b = ctx.buffer(std::max<size_t>(bytes, 16));
        std::memcpy(b->contents(), data, bytes);
        return b;
    };
    auto* instances = bufferOf(&instance, sizeof instance);
    auto* materials = bufferOf(&material, sizeof material);
    auto* meshes = bufferOf(&mesh, sizeof mesh);
    auto* rtParams = bufferOf(&rp, sizeof rp);
    auto* counters = ctx.buffer(sizeof(GPURtCounters));
    auto* textureTable = ctx.buffer(64);
    std::memset(textureTable->contents(), 0, 64);
    auto* lightBuffer = bufferOf(lights.data(), lights.size() * sizeof(GPUSampledLight));
    auto* emitterBuffer = bufferOf(emitters.data(), emitters.size() * sizeof(GPUEmissiveSurface));
    auto* surfaceBuffer = bufferOf(surfaces.data(), surfaces.size() * sizeof(GPUDISurface));
    GPULightClusterParams clusters{};
    auto* clusterBuffer = bufferOf(&clusters, sizeof clusters);
    auto* cells = ctx.buffer(64);
    auto* indices = ctx.buffer(64);
    auto* ranks = ctx.buffer(64);
    auto* signalErrors = ctx.buffer(64);
    std::memset(cells->contents(), 0, 64);
    std::memset(indices->contents(), 0, 64);
    std::memset(ranks->contents(), 0, 64);
    auto* paramsBuffer = ctx.buffer(sizeof(GPUDIParams));

    auto* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, kWidth, kHeight, false);
    td->setStorageMode(MTL::StorageModeShared);
    td->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    auto* output = ctx.texture(td);

    auto* ad = MTL4::ArgumentTableDescriptor::alloc()->init();
    ad->setMaxBufferBindCount(16);
    ad->setMaxTextureBindCount(1);
    NS::Error* error = nullptr;
    auto* table = ctx.device()->newArgumentTable(ad, &error);
    ad->release();
    if (!table) throw ::soc::BenchError("F14-K2 argument table unavailable");
    ctx.keep(table);
    auto* lib = f9Library(ctx, "s8_engine_di.metal", !std::getenv("F14_NO_FAST_MATH"));
    auto* clear = ctx.compute(lib, "rt_clear_counters");
    auto* descriptors = ctx.compute(lib, "rt_write_instances");
    const std::string kernelName = std::getenv("F14_KERNEL") ? std::getenv("F14_KERNEL") : "light_cluster_shade_rt";
    const u32 cfg = std::getenv("F14_CFG") ? u32(std::strtoul(std::getenv("F14_CFG"), nullptr, 0)) : 0u;
    MTL::FunctionConstantValues* constants = nullptr;
    if (kernelName == "k_cfg") {
        constants = MTL::FunctionConstantValues::alloc()->init();
        constants->setConstantValue(&cfg, MTL::DataTypeUInt, NS::UInteger(0));
    }
    auto* shade = linkedPipeline(ctx, lib, kernelName, {"rt_alpha_generic"}, constants);
    if (constants) constants->release();
    auto* outBuffer = ctx.buffer(kThreads * 16);
    auto* id = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
    id->setFunctionCount(1);
    auto* ift = shade->newIntersectionFunctionTable(id);
    id->release();
    if (!ift) throw ::soc::BenchError("F14-K2 intersection function table unavailable");
    ctx.adopt(ift);
    auto* handle = shade->functionHandle(NS::String::string("rt_alpha_generic", NS::UTF8StringEncoding));
    if (!handle) throw ::soc::BenchError("F14-K2 alpha function handle unavailable");
    ift->setFunction(handle, 0);
    ift->setBuffer(materials, 0, 0);
    ift->setBuffer(textureTable, 0, 1);
    ift->setBuffer(geometry.vertices, 0, 2);
    ift->setBuffer(geometry.indices, 0, 3);
    ift->setBuffer(instances, 0, 4);
    ift->setBuffer(meshes, 0, 5);
    ift->setBuffer(rtParams, 0, 6);
    ctx.commitResidency();

    auto dispatch1D = [&](MTL4::ComputeCommandEncoder* e, MTL::ComputePipelineState* ps, u32 n, u32 group) {
        e->setComputePipelineState(ps);
        e->setArgumentTable(table);
        e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(std::min(group, n), 1, 1));
    };
    timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        table->setAddress(counters->gpuAddress(), 0);
        dispatch1D(e, clear, 8, 64);
        e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        table->setAddress(instances->gpuAddress(), 0);
        table->setAddress(materials->gpuAddress(), 1);
        table->setAddress(meshes->gpuAddress(), 2);
        table->setAddress(tlas.instances->gpuAddress(), 3);
        table->setAddress(rtParams->gpuAddress(), 4);
        table->setAddress(counters->gpuAddress(), 5);
        dispatch1D(e, descriptors, 1, 64);
        e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageAccelerationStructure, MTL4::VisibilityOptionDevice);
        e->buildAccelerationStructure(tlas.as, tlas.desc, range(tlas.scratch));
    });

    const u32 group = std::getenv("F14_GROUP") ? u32(std::atoi(std::getenv("F14_GROUP"))) : 8u; // engine: 8
    const bool instrumented = std::getenv("MTL_SHADER_VALIDATION") && std::string(std::getenv("MTL_SHADER_VALIDATION")) != "0";
    rep.note(instrumented ? "MTL_SHADER_VALIDATION on" : "no shader validation");
    std::vector<float> first(kThreads * 4), current(kThreads * 4);
    u32 worstThreads = 0, divergentDispatches = 0, lowZero = 0, nonzero = 0;
    for (u32 frame = 0; frame < kFrames; ++frame) {
        GPUDIParams p{};
        p.width = kWidth; p.height = kHeight; p.lightCount = 12; p.frameIndex = frame;
        p.flags = DI_ENABLE_VISIBILITY; p.slotCount = 1; p.pad1 = 1;
        p.emitterSpecularLow = 0.2f; p.emitterSpecularHigh = 0.4f;
        std::memcpy(paramsBuffer->contents(), &p, sizeof p);
        for (u32 r = 0; r < kRepeats; ++r) {
            timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
                table->setAddress(paramsBuffer->gpuAddress(), 0);
                table->setAddress(surfaceBuffer->gpuAddress(), 1);
                table->setAddress(lightBuffer->gpuAddress(), 2);
                table->setAddress(clusterBuffer->gpuAddress(), 3);
                table->setAddress(cells->gpuAddress(), 4);
                table->setAddress(indices->gpuAddress(), 5);
                table->setAddress(ranks->gpuAddress(), 6);
                table->setAddress(instances->gpuAddress(), 7);
                table->setResource(tlas.as->gpuResourceID(), 8);
                table->setResource(ift->gpuResourceID(), 9);
                table->setAddress(emitterBuffer->gpuAddress(), 12);
                table->setAddress(materials->gpuAddress(), 13);
                table->setAddress(textureTable->gpuAddress(), 14);
                table->setAddress(signalErrors->gpuAddress(), 15);
                table->setAddress(outBuffer->gpuAddress(), 10);
                table->setTexture(output->gpuResourceID(), 0);
                dispatch1D(e, shade, kThreads, group);
            });
            if (kernelName == "k_cfg" && !(cfg & 16u)) std::memcpy(current.data(), outBuffer->contents(), kThreads * 16);
            else output->getBytes(current.data(), kWidth * 16, MTL::Region::Make2D(0, 0, kWidth, kHeight), 0);
            if (r == 0) {
                first = current;
                if (frame == 0)
                    for (u32 t = 0; t < kThreads; ++t) nonzero += current[t * 4] > 0;
                continue;
            }
            u32 differing = 0, zeroed = 0;
            for (u32 t = 0; t < kThreads; ++t) {
                const float* a = &first[t * 4];
                const float* b = &current[t * 4];
                if (std::memcmp(a, b, 12) == 0) continue;
                ++differing;
                zeroed += (a[0] == 0.0f) != (b[0] == 0.0f) && a[2] == b[2];
                if (std::getenv("F14_DUMP") && differing <= 3 && frame == 0)
                    std::printf("  frame %u repeat %u thread %u (x %u y %u lane %u): first (%g %g %g) now (%g %g %g)\n", frame, r, t,
                                t % kWidth, t / kWidth, t % 32u, a[0], a[1], a[2], b[0], b[1], b[2]);
            }
            worstThreads = std::max(worstThreads, differing);
            divergentDispatches += differing > 0;
            lowZero += zeroed;
        }
    }
    std::printf("F14-K2 group %u %s cfg %u: lit threads %u of %u; divergent repeats %u of %u, worst %u threads, low-component-zero threads %u\n",
                group, kernelName.c_str(), cfg, nonzero, kThreads, divergentDispatches, kFrames * (kRepeats - 1), worstThreads, lowZero);
    rep.value("lit_threads", "threads", nonzero, {}, true);
    rep.value("divergent_repeats", "dispatches", divergentDispatches, {{"repeats", kFrames * (kRepeats - 1)}}, false);
    rep.value("worst_divergent_threads", "threads", worstThreads, {}, false);
    rep.value("low_component_zero_threads", "threads", lowZero, {}, false);
}

} // namespace

SOC_BENCH("F14-K2", "engine_di_shade", "Engine brute-force DI shading kernel repeated on identical inputs", engineDi);

} // namespace f9
