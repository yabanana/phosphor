#include "renderer/material_class_export.h"

#include "renderer/gpu_scene.h"
#include "renderer/material_extension_pack.h"
#include "scene/ecs.h"
#include "scene/texture_manager.h"
#include "testbench/material_validation.h"
#include "testbench/transmission_validation.h"
#include "testbench/transmissive_shadow_validation.h"

#include <memory>
#include <stdexcept>

namespace phosphor::matclass {
namespace {

// A TextureManager that keeps nothing: the export reads materials, never pixels.
class DiscardTextureManager final : public TextureManager {
public:
    [[nodiscard]] u32 textureCount() const override { return count_; }

protected:
    u32 createTexture(const u8*, u32, u32, bool) override { return count_++; }

private:
    u32 count_ = 0;
};

} // namespace

SceneEntry entryFromStore(std::string name, std::string family, const SceneStore& store) {
    std::vector<u32> masks;
    masks.reserve(store.materials().size());
    for (const GPUMaterial& m : store.materials()) masks.push_back(materialFeatureBits(m));
    return makeSceneEntry(std::move(name), std::move(family), masks, store.volumetricSubsurface());
}

std::vector<std::string> fixtureFamilies() { return {"material", "transmission", "shadow"}; }

std::vector<std::string> fixtureScenarios(std::string_view family) {
    std::vector<std::string> out;
    const auto copy = [&](std::span<const std::string_view> names) {
        for (std::string_view n : names) out.emplace_back(n);
    };
    if (family == "material") copy(MaterialValidation::scenarios());
    else if (family == "transmission") copy(TransmissionValidation::scenarios());
    else if (family == "shadow") copy(TransmissiveShadowValidation::scenarios());
    else throw std::invalid_argument("unknown fixture family: " + std::string(family));
    return out;
}

SceneEntry exportFixtureScene(std::string_view family, std::string_view scenario) {
    ECS ecs;
    GpuScene scene;
    DiscardTextureManager textures;
    std::unique_ptr<TestBench> bench;
    if (family == "material") bench = std::make_unique<MaterialValidation>(std::string(scenario));
    else if (family == "transmission") bench = std::make_unique<TransmissionValidation>(std::string(scenario));
    else if (family == "shadow") bench = std::make_unique<TransmissiveShadowValidation>(std::string(scenario));
    else throw std::invalid_argument("unknown fixture family: " + std::string(family));
    bench->setup(ecs, scene, textures);
    SceneStore store;
    store.sync(ecs, scene);
    SceneEntry entry = entryFromStore(std::string(family) + ":" + std::string(scenario), std::string(family), store);
    bench->teardown(ecs, scene);
    return entry;
}

Manifest exportFixtureFamily(std::string_view family) {
    Manifest m;
    m.source = "tools/class_manifest (F15 fixtures, family " + std::string(family) + ")";
    for (const std::string& s : fixtureScenarios(family)) m.scenes.push_back(exportFixtureScene(family, s));
    return m;
}

} // namespace phosphor::matclass
