#pragma once

#include "testbench/testbench.h"
#include "scene/components.h"
#include "testbench/corpus_rig.h"
#include <vector>
#include <string>

namespace phosphor {

// ---------------------------------------------------------------------------
// SceneViewer -- loads a glTF/GLB scene from disk, with a fallback to
// procedural geometry (cubes + spheres) if no asset file is found.
// ---------------------------------------------------------------------------

class SceneViewer final : public TestBench {
public:
  explicit SceneViewer(std::string scenePath = {}, std::string materialOverridesPath = {}, CorpusRigRequest rig = {})
      : scenePath_(std::move(scenePath)), materialOverridesPath_(std::move(materialOverridesPath)), rig_(rig) {}
  void setup(ECS &ecs, GpuScene &gpuScene, TextureManager &textures) override;
  void update(float dt, ECS &ecs) override;
  void teardown(ECS &ecs, GpuScene &gpuScene) override;

  [[nodiscard]] const char *assetSource() const override { return loadedGltf_ ? loadedPath_.c_str() : "procedural"; }
  [[nodiscard]] const char *getName() const override { return "Scene Viewer (glTF)"; }
  [[nodiscard]] CameraSetup getDefaultCamera() const override;

private:
    void setupProceduralFallback(ECS& ecs, GpuScene& gpuScene, TextureManager& textures);

    std::vector<EntityID> entities_;
    bool loadedGltf_ = false;
    std::string scenePath_;
    std::string materialOverridesPath_; // F15.1 --material-overrides (empty = none)
    std::string loadedPath_;
    CorpusRigRequest rig_;      // F15.3 ticket 22 (--corpus-rig)
    CorpusRigView    rigView_;  // framing computed at setup from the loaded entities' world bounds
};

} // namespace phosphor
