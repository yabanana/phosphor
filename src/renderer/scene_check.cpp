#include "renderer/scene_check.h"

#include "renderer/cull_reference.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "renderer/transform_reference.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace phosphor {

namespace {

void fail(SceneCheckResult& r, const std::string& what) {
    r.pass = false;
    if (r.first.empty()) r.first = what;
}

} // namespace

SceneCheckResult compareScene(const SceneStore& store, const GpuScene& scene, const ECS& ecs, const GPUCullParams& cull,
                              const float* motionSinCos, const SceneReadback& gpu, bool gpuDriven) {
    SceneCheckResult r;
    const u32 slots = store.slotCapacity();
    r.slots = slots;
    const auto mirror = store.instances();
    if (gpu.instances.size() < slots || mirror.size() < slots) {
        ++r.instanceErrors;
        fail(r, "instance read-back smaller than the slot capacity");
        r.ecs = store.verifyAgainstEcs(ecs, scene);
        return r;
    }

    // Instances: the GPU-computed matrices against the CPU reference, every
    // other byte against the mirror.
    std::vector<float> worlds;
    referenceWorlds(store, motionSinCos, worlds);
    for (u32 s = 0; s < slots; ++s) {
        const GPUInstance& g = gpu.instances[s];
        const GPUInstance& m = mirror[s];
        if (std::memcmp(&g.meshIndex, &m.meshIndex, sizeof(GPUInstance) - sizeof(g.modelMatrix)) != 0) {
            if (r.instanceErrors++ == 0) fail(r, "instance fields differ at slot " + std::to_string(s));
        }
        // Slack slots carry no matrix the GPU or the CPU keeps up to date.
        if ((m.flags & INSTANCE_FLAG_VALID) == 0) continue;
        if (std::memcmp(g.modelMatrix, &worlds[size_t(s) * 16], sizeof(g.modelMatrix)) != 0) {
            if (r.matrixErrors++ == 0) fail(r, "model matrix differs from the reference at slot " + std::to_string(s));
        }
    }
    // Materials.
    const auto materials = store.materials();
    if (gpu.materials.size() < materials.size() ||
        std::memcmp(gpu.materials.data(), materials.data(), materials.size_bytes()) != 0) {
        for (size_t i = 0; i < materials.size() && i < gpu.materials.size(); ++i) {
            if (std::memcmp(&gpu.materials[i], &materials[i], sizeof(GPUMaterial)) != 0) ++r.materialErrors;
        }
        if (gpu.materials.size() < materials.size()) ++r.materialErrors;
        fail(r, "materials differ from the mirror (" + std::to_string(r.materialErrors) + ")");
    }

    if (gpuDriven && (gpu.prefix.size() < size_t(slots) + 1 || gpu.visible.size() < slots)) {
        ++r.listErrors;
        fail(r, "visible list / prefix read-back too small");
    } else if (gpuDriven) {
        // Visibility of every slot from the GPU prefix, against the CPU
        // reference evaluated on the GPU's own instances (a matrix error is
        // reported above, not twice).
        GPUCullParams params = cull;
        params.slotCount = slots;
        std::vector<u8> expected;
        std::vector<float> margins;
        cullReference(gpu.instances, scene.meshInfos(), params, expected, &margins);
        std::vector<u8> gpuResult(slots, CULL_RESULT_INVALID);
        u32 gpuVisible = 0;
        for (u32 s = 0; s < slots; ++s) {
            const u32 delta = gpu.prefix[s + 1] - gpu.prefix[s];
            if (delta > 1) {
                if (r.listErrors++ == 0) fail(r, "prefix not monotonic by 0/1 at slot " + std::to_string(s));
                continue;
            }
            if (delta == 1) {
                if (gpu.visible[gpu.prefix[s]] != s && r.listErrors++ == 0) {
                    fail(r, "visible list entry " + std::to_string(gpu.prefix[s]) + " is not slot " + std::to_string(s));
                }
                ++gpuVisible;
            }
            const bool gpuSees = delta == 1;
            const bool cpuSees = expected[s] == 0;
            if (expected[s] == CULL_RESULT_INVALID) {
                if (gpuSees && r.visibleErrors++ == 0) fail(r, "invalid slot " + std::to_string(s) + " is visible");
                continue;
            }
            if (gpuSees != cpuSees) {
                if (std::fabs(margins[s]) < CULL_BAND) {
                    ++r.bandDifferences;
                } else if (r.visibleErrors++ == 0) {
                    fail(r, "visibility of slot " + std::to_string(s) + " differs from the reference (margin " +
                                std::to_string(margins[s]) + ")");
                }
            }
        }
        if (gpu.prefix[0] != 0 && r.listErrors++ == 0) fail(r, "prefix[0] != 0");

        // Draw arguments of every command from the GPU's prefix.
        std::vector<u32> args;
        drawArgsReference(store.gpuBuckets(), store.commandBuckets(), gpu.prefix, args);
        u32 nonEmpty = 0;
        for (size_t i = 0; i < args.size(); ++i) {
            if (i >= gpu.drawArgs.size() || gpu.drawArgs[i] != args[i]) {
                if (r.argErrors++ == 0) fail(r, "draw argument word " + std::to_string(i) + " differs from the reference");
            }
            if (i % 2 == 0 && args[i] > 0) ++nonEmpty;
        }
        // Counters.
        const GPUSceneCounters& c = gpu.counters;
        if (c.visible != gpuVisible) {
            ++r.counterErrors;
            fail(r, "visible counter " + std::to_string(c.visible) + " != list " + std::to_string(gpuVisible));
        }
        if (c.tested != store.instanceCount()) {
            ++r.counterErrors;
            fail(r, "tested counter " + std::to_string(c.tested) + " != live instances " +
                        std::to_string(store.instanceCount()));
        }
        const u32 expectedDraws = gpu.drawGateOpen ? nonEmpty : 0u;
        if (c.drawCommands != expectedDraws) {
            ++r.counterErrors;
            fail(r, "draw command counter " + std::to_string(c.drawCommands) + " != " + std::to_string(expectedDraws) +
                        (gpu.drawGateOpen ? "" : " (draw gate closed)"));
        }
        if (c.visible + c.culledFrustum + c.culledDistance + c.culledSize != c.tested) {
            ++r.counterErrors;
            fail(r, "visible + culled != tested");
        }
    }
    if (gpu.counters.queueOverflow != 0) {
        ++r.counterErrors;
        fail(r, "GPU queue overflow " + std::to_string(gpu.counters.queueOverflow));
    }
    r.ecs = store.verifyAgainstEcs(ecs, scene);
    if (!r.ecs.empty()) fail(r, "mirror != ECS: " + r.ecs);
    return r;
}

std::string formatSceneCheck(const SceneCheckResult& r) {
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "slots %u | instances %s (%u) | matrices %s (%u) | materials %s (%u) | visible %s (%u, band %u) | "
                  "list %s | args %s (%u) | counters %s | ecs %s | %s",
                  r.slots, r.instanceErrors ? "WRONG" : "ok", r.instanceErrors, r.matrixErrors ? "WRONG" : "ok",
                  r.matrixErrors, r.materialErrors ? "WRONG" : "ok", r.materialErrors, r.visibleErrors ? "WRONG" : "ok",
                  r.visibleErrors, r.bandDifferences, r.listErrors ? "WRONG" : "ok", r.argErrors ? "WRONG" : "ok",
                  r.argErrors, r.counterErrors ? "WRONG" : "ok", r.ecs.empty() ? "ok" : "WRONG",
                  r.pass ? "PASS" : "FAIL");
    std::string line = buf;
    if (!r.pass) line += " | first: " + r.first;
    return line;
}

} // namespace phosphor
