#include "testbench/chess_fixture.h"
#include "core/log.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "renderer/material_description_io.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/gltf_loader.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {
namespace {
constexpr std::array<std::string_view, 3> kRigs{"chess-backlight", "chess-sidelight", "chess-shadow"};
constexpr float pi = 3.14159265358979323846f;

bool startsWith(const std::string& s, std::string_view prefix) { return s.compare(0, prefix.size(), prefix) == 0; }

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
    // Same convention as the other fixtures: local -Z through the entity matrix is the direction the light travels.
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return glm::quat(1, 0, 0, 0);
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}

bool hiddenPiece(const std::string& material) {
    for (const char* prefix : {"King_", "Queen_", "Castle_", "Knight_", "Bishop_"})
        if (startsWith(material, prefix)) return true;
    return false;
}
} // namespace

bool ChessFixture::parse(std::string_view scenario, ChessOptions& out, std::string& error) {
    out = ChessOptions{};
    std::vector<std::string_view> parts;
    for (size_t start = 0; start <= scenario.size();) {
        const size_t slash = scenario.find('/', start);
        parts.push_back(scenario.substr(start, slash == std::string_view::npos ? std::string_view::npos : slash - start));
        if (slash == std::string_view::npos) break;
        start = slash + 1;
    }
    if (parts.empty() || std::find(kRigs.begin(), kRigs.end(), parts[0]) == kRigs.end()) {
        error = "unknown chess rig '" + std::string(parts.empty() ? scenario : parts[0]) + "'";
        return false;
    }
    if (parts[0] == "chess-backlight") out.rig = ChessOptions::Rig::Backlight;
    else if (parts[0] == "chess-sidelight") out.rig = ChessOptions::Rig::Sidelight;
    else out.rig = ChessOptions::Rig::Shadow;
    out.pawnsOnly = out.rig != ChessOptions::Rig::Sidelight;
    const bool back = out.rig == ChessOptions::Rig::Backlight, side = out.rig == ChessOptions::Rig::Sidelight,
               shadow = out.rig == ChessOptions::Rig::Shadow;
    for (size_t i = 1; i < parts.size(); ++i) {
        const size_t eq = parts[i].find('=');
        if (eq == std::string_view::npos) {
            error = "option '" + std::string(parts[i]) + "' is not key=value";
            return false;
        }
        const std::string_view key = parts[i].substr(0, eq), value = parts[i].substr(eq + 1);
        auto bad = [&] {
            error = "option '" + std::string(parts[i]) + "' is not valid for " + std::string(parts[0]);
            return false;
        };
        if (key == "mask") {
            if (value != "1" && value != "0") return bad();
            out.mask = value == "1";
        } else if (key == "emit") {
            if (value != "1" && value != "0") return bad();
            out.emit = value == "1";
        } else if (key == "transmission" && (back || shadow)) {
            if (!parseFloat(value, out.transmission) || out.transmission < 0.0f || out.transmission > 1.0f) return bad();
        } else if (key == "tint" && (back || shadow)) {
            if (value != "asset" && value != "white") return bad();
            out.whiteTint = value == "white";
        } else if (key == "panel" && back) {
            if (value != "on" && value != "off") return bad();
            out.panel = value == "on";
        } else if (key == "weight" && side) {
            if (!parseFloat(value, out.weight) || out.weight < 0.0f || out.weight > 1.0f) return bad();
        } else if (key == "scale" && side) {
            if (!parseFloat(value, out.scale) || !(out.scale > 0.0f) || out.scale > 100.0f) return bad();
        } else if (key == "sss" && side) {
            if (value != "on" && value != "off") return bad();
            out.sss = value == "on";
        } else if (key == "view" && side) {
            if (value != "black" && value != "white") return bad();
            out.viewWhite = value == "white";
        } else if (key == "tops" && shadow) {
            if (value == "on") out.tops = ChessOptions::Tops::AsAuthored;
            else if (value == "opaque") out.tops = ChessOptions::Tops::Opaque;
            else if (value == "hidden") out.tops = ChessOptions::Tops::Hidden;
            else return bad();
        } else {
            return bad();
        }
    }
    if (out.emit && !out.mask) {
        error = "option 'emit=1' needs mask=1 for " + std::string(parts[0]);
        return false;
    }
    return true;
}

bool ChessFixture::validScenario(std::string_view scenario) {
    ChessOptions options;
    std::string error;
    return parse(scenario, options, error);
}
std::span<const std::string_view> ChessFixture::rigs() { return kRigs; }

float ChessFixture::environmentRadiance(std::string_view scenario) {
    ChessOptions options;
    std::string error;
    if (!parse(scenario, options, error) || options.mask) return 0.0f;
    return options.rig == ChessOptions::Rig::Backlight ? BacklightEnvironment : 0.0f;
}

std::vector<std::string> ChessFixture::targetMaterials(const ChessOptions& o) {
    if (o.rig == ChessOptions::Rig::Sidelight)
        return o.viewWhite ? std::vector<std::string>{"King_White", "Queen_White"} : std::vector<std::string>{"King_Black", "Queen_Black"};
    if (o.rig == ChessOptions::Rig::Shadow) return {"Pawn_Top_Black", "Pawn_Top_White"};  // the receiver region excludes both rows' own pixels
    return {"Pawn_Top_Black"};
}

CameraSetup ChessFixture::camera(const ChessOptions& o) {
    switch (o.rig) {
    case ChessOptions::Rig::Backlight:
        return {{0.0f, 0.16f, -0.40f}, {0.0f, 0.07f, 0.156f}, 0.56f, false};
    case ChessOptions::Rig::Sidelight: {
        const float z = o.viewWhite ? -1.0f : 1.0f;
        return {{0.0f, 0.10f, 0.53f * z}, {0.0f, 0.095f, 0.22f * z}, 0.31f, false};
    }
    case ChessOptions::Rig::Shadow:
        return {{0.0f, 0.35f, -0.20f}, {0.0f, 0.0185f, 0.10f}, 0.45f, false};
    }
    return {};
}

glm::vec3 ChessFixture::towardSun(const ChessOptions& o) {
    switch (o.rig) {
    case ChessOptions::Rig::Backlight: return glm::vec3(0.0f);
    case ChessOptions::Rig::Sidelight:
        return glm::normalize(o.viewWhite ? glm::vec3(-0.9f, 0.25f, -0.35f) : glm::vec3(0.9f, 0.25f, 0.35f));
    case ChessOptions::Rig::Shadow: // 60 degrees from the vertical, behind the black pawn row: the shadows fall towards the camera
        return {0.0f, 0.5f, 0.8660254f};
    }
    return glm::vec3(0.0f);
}

ChessFixture::ChessFixture(std::string scenario, std::string scenePath, std::string overridesPath)
    : scenario_(std::move(scenario)), scenePath_(std::move(scenePath)), overridesPath_(std::move(overridesPath)) {
    std::string error;
    if (!parse(scenario_, options_, error)) throw std::invalid_argument("Unknown chess scenario: " + scenario_ + ": " + error);
    name_ = "Chess fixture: " + scenario_;
}

CameraSetup ChessFixture::getDefaultCamera() const { return camera(options_); }

void ChessFixture::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!entities_.empty()) throw std::logic_error("Chess fixture already set up");
    if (scenePath_.empty() || scenePath_ == "procedural")
        throw std::runtime_error("the chess fixture needs --scene <ABeautifulGame.gltf> (the pinned Khronos asset)");
    textures.createDefaultTextures();
    {
        GltfLoader loader(scene, textures, ecs);
        if (!loader.loadFromFile(scenePath_)) throw std::runtime_error("Required scene failed to load: " + scenePath_ + ": " + loader.lastError());
        entities_.assign(loader.createdEntities().begin(), loader.createdEntities().end());
    }
    const std::filesystem::path assetDir = std::filesystem::path(scenePath_).parent_path();

    const std::vector<std::string> names = scene.materialNames();
    auto edit = [&](const std::string& name, auto&& change) {
        for (u32 i = 0; i < names.size(); ++i) {
            if (names[i] != name) continue;
            MaterialDescription d = scene.materialDescriptions()[i];
            change(d);
            if (!scene.setMaterialDescription(i, d)) throw std::runtime_error("cannot edit material " + name);
        }
    };

    const std::vector<std::string> targets = targetMaterials(options_);
    if (options_.mask) {
        // The mask capture: every material untextured and without extensions, black except the targets (white diffuse, specular off)
        // under one sun pi W/m2 aimed along the view direction: a target pixel reads cos(N, V) (> 0.2 over all but a rim of the
        // silhouette), every other pixel is black.  (Pure emissive targets were the first design: the kings' and queens' emissive
        // triangle count hung the GPU on frame 0 until the F15 light budget (shaders/light_cluster.metal); the mask keeps the lit white, `emit=1` selects
        // the emissive design as the regression scene.)
        for (u32 i = 0; i < names.size(); ++i) {
            MaterialDescription d;
            d.base.metallic = 0.0f;
            d.base.roughness = 1.0f;
            const bool target = std::find(targets.begin(), targets.end(), names[i]) != targets.end();
            d.base.baseColor = target ? std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f} : std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f};
            d.specular.emplace();
            d.specular->specularFactor = 0.0f;
            // emit=1: the targets are PURE emissive white (the F15 GPU-safety regression scene: every triangle of a king or queen is a light).
            if (target && options_.emit) d.base.emissive = {1.0f, 1.0f, 1.0f};
            if (!scene.setMaterialDescription(i, d)) throw std::runtime_error("cannot edit material " + names[i]);
        }
    } else {
        if (options_.rig == ChessOptions::Rig::Sidelight && options_.sss) {
            if (overridesPath_.empty()) throw std::runtime_error("chess-sidelight needs --material-overrides <chess-subsurface.overrides.json>");
            std::ifstream file(overridesPath_, std::ios::binary);
            if (!file) throw std::runtime_error("cannot read material overrides " + overridesPath_);
            std::stringstream json;
            json << file.rdbuf();
            // Texture paths of the chess override file are relative to the glTF's directory (the pinned asset's textures).
            const TexturePathResolver resolve = [&](const std::string& path, bool sRGB) {
                return textures.loadTexture((assetDir / path).string(), sRGB);
            };
            std::string error;
            if (!applyMaterialOverridesToScene(json.str(), scene, resolve, error))
                throw std::runtime_error("material overrides " + overridesPath_ + ": " + error);
            for (const char* name : {"King_Black", "King_White", "Queen_Black", "Queen_White"})
                edit(name, [&](MaterialDescription& d) {
                    if (!d.subsurface) throw std::runtime_error(std::string(name) + " has no subsurface block after the overrides");
                    if (options_.weight >= 0.0f) d.subsurface->weight = options_.weight;
                    d.subsurface->radius *= options_.scale;
                });
        }
        if (options_.rig != ChessOptions::Rig::Sidelight) {
            for (const char* name : {"Pawn_Top_Black", "Pawn_Top_White"})
                edit(name, [&](MaterialDescription& d) {
                    if (!d.transmission) throw std::runtime_error(std::string(name) + " has no transmission block (is this ABeautifulGame?)");
                    if (options_.transmission >= 0.0f) d.transmission->factor = options_.transmission;
                    if (options_.tops == ChessOptions::Tops::Opaque) d.transmission->factor = 0.0f;
                    if (options_.whiteTint) d.base.baseColor = {1.0f, 1.0f, 1.0f, 1.0f};
                });
        }
    }

    // Declared instance visibility: the rigs hide the pieces that would occlude (pawnsOnly) and, for the shadow control, the pawn tops.
    for (EntityID e : entities_) {
        MeshInstanceComponent& instance = ecs.getComponent<MeshInstanceComponent>(e);
        if (instance.materialIndex >= names.size()) continue;
        const std::string& material = names[instance.materialIndex];
        const bool hide = (options_.pawnsOnly && hiddenPiece(material)) ||
                          (options_.tops == ChessOptions::Tops::Hidden && startsWith(material, "Pawn_Top"));
        if (hide) {
            instance.setVisible(false);
            instance.setCastsShadows(false); // the snapshot reader requires equal visible and shadow roles
        }
    }

    if (options_.rig == ChessOptions::Rig::Backlight && options_.panel && !options_.mask) {
        const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1); // unit square facing +Y
        const u32 mesh = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
        auto quad = [&](glm::vec3 centre, float width, float height, float emission) {
            const EntityID e = ecs.createEntity();
            entities_.push_back(e);
            TransformComponent transform;
            transform.position = centre;
            transform.rotation = glm::angleAxis(-pi * 0.5f, glm::vec3(1, 0, 0)); // faces -Z, towards the camera
            transform.scale = {width, 1.0f, height};
            transform.updateMatrix();
            ecs.addComponent(e, std::move(transform));
            MeshInstanceComponent instance;
            instance.meshHandle = mesh;
            instance.materialIndex = 0;
            instance.setVisible(true);
            instance.setCastsShadows(true);
            instance.setStatic(true);
            ecs.addComponent(e, std::move(instance));
            MaterialComponent material;
            material.baseColorFactor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // purely emissive: nothing diffuse to light
            material.metallicFactor = 0.0f;
            material.roughnessFactor = 1.0f;
            material.emissiveFactor = glm::vec3(emission);
            ecs.addComponent(e, std::move(material));
        };
        // A dark backing plate and the bright checkerboard cells 0.5 mm in front of it.
        quad({0.0f, PanelHeight * 0.5f, PanelZ}, 2.0f * PanelHalfWidth, PanelHeight, 0.0f);
        const int columns = static_cast<int>(std::lround(2.0f * PanelHalfWidth / CellSize)), rows = static_cast<int>(std::lround(PanelHeight / CellSize));
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                if ((row + column) % 2 == 0)
                    quad({-PanelHalfWidth + CellSize * (float(column) + 0.5f), CellSize * (float(row) + 0.5f), PanelZ - 0.0005f}, CellSize,
                         CellSize, PanelEmission);
    }

    const CameraSetup view = camera(options_);
    const glm::vec3 sun = options_.mask ? glm::normalize(view.position - view.target) : towardSun(options_);
    if (glm::dot(sun, sun) > 0.0f) {
        const EntityID light = ecs.createEntity();
        entities_.push_back(light);
        TransformComponent transform;
        transform.rotation = orientLight(sun);
        transform.updateMatrix();
        ecs.addComponent(light, std::move(transform));
        LightComponent component;
        component.type = LightType::Directional;
        component.color = glm::vec3(1.0f);
        component.intensity = options_.mask ? pi : SunIrradiance;
        ecs.addComponent(light, std::move(component));
    }
    LOG_INFO("ChessFixture: %s on '%s' (%zu entities)", scenario_.c_str(), scenePath_.c_str(), entities_.size());
}

void ChessFixture::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : entities_) ecs.destroyEntity(e);
    entities_.clear();
}
} // namespace phosphor
