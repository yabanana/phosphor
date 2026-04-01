#pragma once

#include "core/types.h"
#include <glm/glm.hpp>
#include <memory>

namespace phosphor {

class ECS;
class GpuScene;
class TextureManager;

// ---------------------------------------------------------------------------
// TestBenchType -- enumerates the built-in demo scenes.
// ---------------------------------------------------------------------------

enum class TestBenchType : int {
    TorusDemo   = 0,
    PBRGrid     = 1,
    StressTest  = 2,
    SceneViewer = 3,
    ManyLights  = 4,
    CornellBox  = 5,
    CullingViz  = 6,
    COUNT
};

/// Human-readable name for a test bench type.
const char* testBenchName(TestBenchType type);

// Overloads used by UIPanels (linked against, declared extern in ui_panels.cpp)
const char* testBenchName(int index);
int testBenchCount();

// ---------------------------------------------------------------------------
// CameraSetup -- initial camera placement returned by each test bench.
// ---------------------------------------------------------------------------

struct CameraSetup {
    glm::vec3 position{0.0f, 2.0f, 5.0f};
    glm::vec3 target{0.0f, 0.0f, 0.0f};
    float     distance = 5.0f;
    bool      orbit    = false; // true = orbit camera, false = FPS camera
};

// ---------------------------------------------------------------------------
// TestBench -- abstract base for demo scenes.
// ---------------------------------------------------------------------------

class TestBench {
public:
    virtual ~TestBench() = default;

    /// Populate the ECS and upload geometry/materials to the GPU scene.
    virtual void setup(ECS& ecs, GpuScene& gpuScene, TextureManager& textures) = 0;

    /// Per-frame update (animations, rotations, etc.).
    virtual void update(float dt, ECS& ecs) = 0;

    /// Remove all entities created by this bench and release GPU resources.
    virtual void teardown(ECS& ecs, GpuScene& gpuScene) = 0;

    /// Display name shown in the UI.
    [[nodiscard]] virtual const char* getName() const = 0;

    /// Suggested initial camera placement.
    [[nodiscard]] virtual CameraSetup getDefaultCamera() const = 0;
};

/// Factory: create a concrete TestBench by type enum.
std::unique_ptr<TestBench> createTestBench(TestBenchType type);

} // namespace phosphor
