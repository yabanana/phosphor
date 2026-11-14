#include "testbench/terrain_fixture.h"
#include "core/log.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "renderer/scene_extract.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/terrain_textures.h"
#include "scene/texture_manager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {
namespace {
constexpr float pi = 3.14159265358979323846f;

// The scenarios the class-manifest export covers: every runner key's scene shape (stack with and without the coat finish, the
// single-material twin, the flat transition and furnace fixtures).
constexpr std::array<std::string_view, 7> kScenarios{
    "terrain",
    "terrain/coat=off",
    "terrain/coat=zero",
    "terrain/single=0/coat=off",
    "terrain/ramp=on",
    "terrain/ramp=on/single=0",
    "terrain/furnace=on",
};

bool parseFloat(std::string_view text, float& out) {
    if (text.empty()) return false;
    const std::string copy(text);
    char* end = nullptr;
    const float v = std::strtof(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size() || !std::isfinite(v)) return false;
    out = v;
    return true;
}

glm::quat orientLight(glm::vec3 towardLight) {
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return glm::quat(1, 0, 0, 0);
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}

u32 upload(TextureManager& textures, const terrain::Image& image, bool sRGB) {
    return textures.loadTextureFromMemory(image.rgba.data(), image.width, image.height, 4, sRGB);
}

struct LayerTextures {
    u32 color = INVALID_TEXTURE_INDEX, normal = INVALID_TEXTURE_INDEX, roughness = INVALID_TEXTURE_INDEX,
        occlusion = INVALID_TEXTURE_INDEX, height = INVALID_TEXTURE_INDEX;
};

LayerTextures layerTextures(TextureManager& textures, u32 layer, const std::string& dir) {
    LayerTextures t;
    if (dir.empty()) {
        const terrain::LayerImages images = terrain::proceduralLayer(layer);
        t.color = upload(textures, images.color, true);
        t.normal = upload(textures, images.normal, false);
        t.roughness = upload(textures, images.roughness, false);
        t.occlusion = upload(textures, images.occlusion, false);
        t.height = upload(textures, images.height, false);
        return t;
    }
    // The owner-approved CC0 sets (JPG only).  A missing map is an error: a half-replaced stack would be a different material.
    for (const char* map : {"Color", "NormalGL", "Roughness", "AmbientOcclusion", "Displacement"}) {
        const std::string path = terrain::cc0Path(dir, layer, map);
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("terrain texture missing: " + path);
    }
    t.color = textures.loadTexture(terrain::cc0Path(dir, layer, "Color"), true);
    t.normal = textures.loadTexture(terrain::cc0Path(dir, layer, "NormalGL"), false);
    t.roughness = textures.loadTexture(terrain::cc0Path(dir, layer, "Roughness"), false);
    t.occlusion = textures.loadTexture(terrain::cc0Path(dir, layer, "AmbientOcclusion"), false);
    t.height = textures.loadTexture(terrain::cc0Path(dir, layer, "Displacement"), false);
    return t;
}

MaterialComponent layerComponent(u32 layer, const LayerTextures& t) {
    const terrain::LayerSpec& spec = terrain::layerSpec(layer);
    MaterialComponent m;
    m.baseColorFactor = glm::vec4(1.0f);
    m.metallicFactor = 0.0f;
    m.roughnessFactor = spec.roughness;
    m.normalScale = spec.normalScale;
    m.occlusionStrength = 1.0f;
    m.baseColorTexIndex = t.color;
    m.normalTexIndex = t.normal;
    m.metallicRoughnessTexIndex = t.roughness; // grey: G = roughness, B ignored (metallic factor 0)
    m.occlusionTexIndex = t.occlusion;
    return m;
}

MaterialComponent plainComponent(float albedo, float metallic, float roughness) {
    MaterialComponent m;
    m.baseColorFactor = glm::vec4(albedo, albedo, albedo, 1.0f);
    m.metallicFactor = metallic;
    m.roughnessFactor = roughness;
    return m;
}
} // namespace

bool TerrainFixture::parse(std::string_view scenario, TerrainOptions& out, std::string& error) {
    out = TerrainOptions{};
    std::vector<std::string_view> parts;
    for (size_t start = 0; start <= scenario.size();) {
        const size_t slash = scenario.find('/', start);
        parts.push_back(scenario.substr(start, slash == std::string_view::npos ? std::string_view::npos : slash - start));
        if (slash == std::string_view::npos) break;
        start = slash + 1;
    }
    if (parts.empty() || parts[0] != "terrain") {
        error = "unknown terrain scenario '" + std::string(scenario) + "'";
        return false;
    }
    for (size_t i = 1; i < parts.size(); ++i) {
        const size_t eq = parts[i].find('=');
        if (eq == std::string_view::npos) {
            error = "option '" + std::string(parts[i]) + "' is not key=value";
            return false;
        }
        const std::string_view key = parts[i].substr(0, eq), value = parts[i].substr(eq + 1);
        auto bad = [&] {
            error = "option '" + std::string(parts[i]) + "' is not valid for terrain";
            return false;
        };
        if (key == "weights") {
            std::array<float, 4> w{};
            size_t start = 0;
            for (size_t k = 0; k < 4; ++k) {
                const size_t comma = value.find(',', start);
                const std::string_view piece = value.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
                if (!parseFloat(piece, w[k]) || w[k] < 0.0f || w[k] > 1.0f) return bad();
                if (comma == std::string_view::npos) {
                    if (k != 3) return bad();
                    start = value.size();
                } else {
                    start = comma + 1;
                }
            }
            for (size_t k = 0; k < 4; ++k) out.weights[k] = static_cast<u8>(std::lround(w[k] * 255.0f));
            out.constantMask = true;
        } else if (key == "layers") {
            float v = 0.0f;
            if (!parseFloat(value, v) || v < 1.0f || v > 4.0f || v != std::floor(v)) return bad();
            out.layers = static_cast<u32>(v);
        } else if (key == "single") {
            float v = 0.0f;
            if (!parseFloat(value, v) || v < 0.0f || v > 3.0f || v != std::floor(v)) return bad();
            out.single = static_cast<int>(v);
        } else if (key == "height") {
            if (value != "on" && value != "off") return bad();
            out.height = value == "on";
        } else if (key == "coat") {
            if (value == "on") out.coat = TerrainOptions::Coat::On;
            else if (value == "off") out.coat = TerrainOptions::Coat::Off;
            else if (value == "zero") out.coat = TerrainOptions::Coat::Zero;
            else return bad();
        } else if (key == "ramp") {
            if (value == "on") out.ramp = TerrainOptions::Ramp::Forward;
            else if (value == "reverse") out.ramp = TerrainOptions::Ramp::Reverse;
            else return bad();
        } else if (key == "furnace") {
            if (value != "on") return bad();
            out.furnace = true;
        } else {
            return bad();
        }
    }
    if (out.ramp != TerrainOptions::Ramp::None && out.furnace) {
        error = "ramp and furnace are exclusive";
        return false;
    }
    if (out.ramp != TerrainOptions::Ramp::None && out.single > 1) {
        error = "ramp has two layers: single must be 0 or 1";
        return false;
    }
    if (out.single >= 0 && out.furnace) {
        error = "furnace has no single twin";
        return false;
    }
    if (out.single >= 0 && out.constantMask) {
        error = "single and weights are exclusive";
        return false;
    }
    return true;
}

bool TerrainFixture::validScenario(std::string_view scenario) {
    TerrainOptions options;
    std::string error;
    return parse(scenario, options, error);
}
std::span<const std::string_view> TerrainFixture::scenarios() { return kScenarios; }

float TerrainFixture::environmentRadiance(std::string_view scenario) {
    TerrainOptions options;
    std::string error;
    if (!parse(scenario, options, error)) return 0.0f;
    if (options.furnace) return 1.0f;
    if (options.ramp != TerrainOptions::Ramp::None) return 0.0f;
    return PatchEnvironment;
}

CameraSetup TerrainFixture::camera(const TerrainOptions& o) {
    if (o.furnace) return {{0.0f, 3.0f, 6.0f}, {0.0f, 0.0f, 0.0f}, 6.7f, false};
    if (o.ramp != TerrainOptions::Ramp::None) return {{0.0f, 6.5f, 6.5f}, {0.0f, 0.0f, 0.0f}, 9.2f, false};
    return {{0.0f, 5.0f, 8.0f}, {0.0f, 0.1f, 0.5f}, 9.4f, false};
}

glm::vec3 TerrainFixture::towardSun(const TerrainOptions& o) {
    if (o.furnace) return glm::vec3(0.0f);
    if (o.ramp != TerrainOptions::Ramp::None) return glm::normalize(glm::vec3(0.3f, 0.9f, 0.3f));
    return glm::normalize(glm::vec3(0.55f, 0.65f, 0.45f));
}

TerrainFixture::TerrainFixture(std::string scenario, std::string texturesDir)
    : scenario_(std::move(scenario)), texturesDir_(std::move(texturesDir)) {
    std::string error;
    if (!parse(scenario_, options_, error)) throw std::invalid_argument("Unknown terrain scenario: " + scenario_ + ": " + error);
    name_ = "Terrain fixture: " + scenario_;
}

CameraSetup TerrainFixture::getDefaultCamera() const { return camera(options_); }

void TerrainFixture::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!entities_.empty()) throw std::logic_error("Terrain fixture already set up");
    textures.createDefaultTextures();
    const TerrainOptions& o = options_;
    const bool flat = o.ramp != TerrainOptions::Ramp::None || o.furnace;

    // --- the layer materials (ordinary library materials) and the stack's own material ---------------------------------
    std::vector<MaterialComponent> layers; // the library entries
    std::vector<u32> layerUvScale;         // per layer: the tiling (1 for the plain fixtures)
    std::vector<LayerTextures> layerTex;
    if (o.furnace) {
        for (u32 i = 0; i < 4; ++i) layers.push_back(plainComponent(1.0f, FurnaceMetallic[i], FurnaceRoughness[i]));
    } else if (o.ramp != TerrainOptions::Ramp::None) {
        layers.push_back(plainComponent(RampDark, 0.0f, RampRoughness));
        layers.push_back(plainComponent(RampLight, 0.0f, RampRoughness));
    } else {
        for (u32 i = 0; i < terrain::kLayerCount; ++i) {
            layerTex.push_back(layerTextures(textures, i, texturesDir_));
            layers.push_back(layerComponent(i, layerTex.back()));
        }
    }

    MeshData mesh;
    if (flat) mesh = ProceduralMeshes::generatePlane(terrain::kPatchSize, terrain::kPatchSize, 1, 1);
    else mesh = terrain::makePatch(PatchGrid, o.single >= 0 ? terrain::layerSpec(static_cast<u32>(o.single)).uvScale : 1.0f);
    const u32 handle = scene.uploadMesh(mesh.positions, mesh.normals, mesh.tangents, mesh.uvs, mesh.indices);

    MaterialComponent owner;
    if (o.single >= 0) {
        owner = layers.at(static_cast<size_t>(o.single)); // the twin: the layer's own material, no stack
    } else {
        // The stack's own record (secondary hits and probes read it: an average appearance, declared) and its description.
        owner = plainComponent(0.3f, 0.0f, 0.8f);
        std::vector<u32> indices;
        for (const MaterialComponent& l : layers) indices.push_back(scene.addMaterial(toGPUMaterial(l)));
        auto description = std::make_shared<MaterialDescription>();
        description->layers = LayerStack{};
        LayerStack& stack = *description->layers;
        terrain::Image maskImage;
        if (o.furnace) {
            maskImage.width = FlatMaskWidth;
            maskImage.height = FlatMaskHeight;
            maskImage.rgba.assign(static_cast<size_t>(FlatMaskWidth) * FlatMaskHeight * 4, 255);
            for (u32 y = 0; y < FlatMaskHeight; ++y)
                for (u32 x = 0; x < FlatMaskWidth; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / FlatMaskWidth, v = (static_cast<float>(y) + 0.5f) / FlatMaskHeight;
                    u8* p = &maskImage.rgba[(static_cast<size_t>(y) * FlatMaskWidth + x) * 4];
                    p[0] = static_cast<u8>(std::lround(255.0f * (1.0f - u) * (1.0f - v)));
                    p[1] = static_cast<u8>(std::lround(255.0f * u * (1.0f - v)));
                    p[2] = static_cast<u8>(std::lround(255.0f * (1.0f - u) * v));
                    p[3] = static_cast<u8>(std::lround(255.0f * u * v));
                }
        } else if (o.ramp != TerrainOptions::Ramp::None) {
            maskImage.width = FlatMaskWidth;
            maskImage.height = 4;
            maskImage.rgba.assign(static_cast<size_t>(FlatMaskWidth) * 4 * 4, 0);
            for (u32 y = 0; y < 4; ++y)
                for (u32 x = 0; x < FlatMaskWidth; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / FlatMaskWidth;
                    const float t = o.ramp == TerrainOptions::Ramp::Forward ? u : 1.0f - u; // weight of layer 1
                    u8* p = &maskImage.rgba[(static_cast<size_t>(y) * FlatMaskWidth + x) * 4];
                    p[0] = static_cast<u8>(std::lround(255.0f * (1.0f - t)));
                    p[1] = static_cast<u8>(std::lround(255.0f * t));
                    p[3] = 255;
                }
        } else if (o.constantMask) {
            maskImage = terrain::constantImage(o.weights);
        } else {
            maskImage = terrain::proceduralMask();
        }
        stack.mask.index = upload(textures, maskImage, false);
        const u32 count = flat ? static_cast<u32>(layers.size()) : o.layers;
        for (u32 i = 0; i < count; ++i) {
            MaterialLayer l;
            l.materialIndex = indices[i];
            l.materialName = flat ? "plain" + std::to_string(i) : terrain::layerSpec(i).name;
            l.maskChannel = static_cast<TextureChannel>(i); // R, G, B, A
            if (!flat) {
                l.uvScale = terrain::layerSpec(i).uvScale;
                if (o.height) {
                    l.heightContrast = terrain::layerSpec(i).heightContrast;
                    l.heightTexture.index = layerTex[i].height;
                }
            }
            stack.layers.push_back(l);
        }
        if (!flat && o.coat != TerrainOptions::Coat::Off) {
            description->clearcoat.emplace();
            description->clearcoat->factor = 1.0f;
            description->clearcoat->roughness = 0.12f;
            description->clearcoat->factorTexture.index =
                upload(textures, o.coat == TerrainOptions::Coat::Zero ? terrain::constantImage({0, 0, 0, 255}) : terrain::proceduralCavity(), false);
        }
        std::string problems;
        for (const std::string& p : validate(*description, static_cast<u32>(scene.materials().size()))) problems += p + "; ";
        if (!problems.empty()) throw std::runtime_error("terrain material description invalid: " + problems);
        owner.extension = std::move(description);
    }

    const EntityID e = ecs.createEntity();
    entities_.push_back(e);
    TransformComponent transform;
    transform.updateMatrix();
    ecs.addComponent(e, std::move(transform));
    MeshInstanceComponent instance;
    instance.meshHandle = handle;
    instance.materialIndex = 0;
    instance.setVisible(true);
    instance.setCastsShadows(true);
    instance.setStatic(true);
    ecs.addComponent(e, std::move(instance));
    ecs.addComponent(e, std::move(owner));

    const glm::vec3 sun = towardSun(o);
    if (glm::dot(sun, sun) > 0.0f) {
        const EntityID light = ecs.createEntity();
        entities_.push_back(light);
        TransformComponent lt;
        lt.rotation = orientLight(sun);
        lt.updateMatrix();
        ecs.addComponent(light, std::move(lt));
        LightComponent component;
        component.type = LightType::Directional;
        component.color = glm::vec3(1.0f);
        component.intensity = SunIrradiance;
        ecs.addComponent(light, std::move(component));
    }
    LOG_INFO("TerrainFixture: %s (%s textures, %zu entities)", scenario_.c_str(), texturesDir_.empty() ? "procedural" : "CC0", entities_.size());
}

void TerrainFixture::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : entities_) ecs.destroyEntity(e);
    entities_.clear();
}
} // namespace phosphor
