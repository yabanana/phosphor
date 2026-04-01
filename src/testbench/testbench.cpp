#include "testbench/testbench.h"
#include "testbench/torus_demo.h"
#include "testbench/pbr_grid.h"
#include "testbench/stress_test.h"
#include "testbench/scene_viewer.h"
#include "testbench/many_lights.h"
#include "testbench/cornell_box.h"
#include "testbench/culling_viz.h"

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
    switch (type) {
        case TestBenchType::TorusDemo:   return std::make_unique<TorusDemo>();
        case TestBenchType::PBRGrid:     return std::make_unique<PBRGrid>();
        case TestBenchType::StressTest:  return std::make_unique<StressTest>();
        case TestBenchType::SceneViewer: return std::make_unique<SceneViewer>();
        case TestBenchType::ManyLights:  return std::make_unique<ManyLights>();
        case TestBenchType::CornellBox:  return std::make_unique<CornellBox>();
        case TestBenchType::CullingViz:  return std::make_unique<CullingViz>();
        default:                         return std::make_unique<TorusDemo>();
    }
}

} // namespace phosphor
