#include "testbench/testbench.h"
#include "testbench/lighting_validation.h"
#include "testbench/torus_demo.h"
#include "testbench/pbr_grid.h"
#include "testbench/stress_test.h"
#include "testbench/scene_viewer.h"
#include "testbench/many_lights.h"
#include "testbench/cornell_box.h"
#include "testbench/culling_viz.h"
#include "testbench/million_instances.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// Name table
// ---------------------------------------------------------------------------

static constexpr const char* kBenchNames[] = {
    "Torus Demo",
    "PBR Material Grid",
    "Stress Test (100K)",
    "Scene Viewer (glTF)",
    "Many Lights (1024)",
    "Cornell Box (GI)",
    "Culling Visualization",
    "1M Instances (dynamic)",
};
static_assert(std::size(kBenchNames) == static_cast<size_t>(TestBenchType::COUNT));

const char* testBenchName(TestBenchType type) {
    int i = static_cast<int>(type);
    if (i < 0 || i >= static_cast<int>(TestBenchType::COUNT)) return "Unknown";
    return kBenchNames[i];
}

const char* testBenchName(int index) {
    return testBenchName(static_cast<TestBenchType>(index));
}

int testBenchCount() {
    return static_cast<int>(TestBenchType::COUNT);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<TestBench> createTestBench(TestBenchType type) {
    return createTestBench(type, TestBenchParams{});
}

std::unique_ptr<TestBench> createTestBench(TestBenchType type, const TestBenchParams& params) {
    switch (type) {
        case TestBenchType::TorusDemo:   return std::make_unique<TorusDemo>();
        case TestBenchType::PBRGrid:     return std::make_unique<PBRGrid>();
        case TestBenchType::StressTest:  return std::make_unique<StressTest>();
        case TestBenchType::SceneViewer:
            return std::make_unique<SceneViewer>(params.scenePath);
        case TestBenchType::ManyLights:  return std::make_unique<ManyLights>(params);
        case TestBenchType::CornellBox:
            if(!params.lightingScenario.empty())return std::make_unique<LightingValidation>(params.lightingScenario);
            return std::make_unique<CornellBox>();
        case TestBenchType::CullingViz:  return std::make_unique<CullingViz>(params.cullingScript);
        case TestBenchType::MillionInstances: return std::make_unique<MillionInstances>(params);
        default:                         return std::make_unique<TorusDemo>();
    }
}

} // namespace phosphor
