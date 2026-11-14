#pragma once

// F15.4 (ticket 23): the data of the procedural terrain patch fixture (testbench/terrain_fixture.h), portable and deterministic.
//
//  * the heightfield of the patch (`heightAt`, metres) and the RGBA weight mask / cavity map derived from it (noise + height
//    and slope rules; the seed is `kSeed`, no RNG state, every texel is a pure function of its coordinates),
//  * the PROCEDURAL STAND-IN textures of the four layers (tileable value-noise images: colour, tangent-space normal from the
//    height, roughness in the G channel like glTF metallicRoughness, occlusion, height),
//  * the declared constants of the four layers shared with the CC0 sets: asset name, uv tiling (powers of two, so that the
//    equivalence fixture can scale mesh uvs exactly), factors and the height-contrast amplitude.
//
// The CC0 ambientCG sets (Grass004, Rock030, Ground037, Snow006, 1K JPG; tools/fetch_terrain_materials.py) replace the stand-ins
// through `--terrain-textures DIR` without any code change: the file names follow the ambientCG convention
// `<DIR>/<Asset>/<Asset>_1K-JPG_{Color,NormalGL,Roughness,AmbientOcclusion,Displacement}.jpg` (`cc0Path`).

#include "core/types.h"
#include "scene/procedural.h"

#include <array>
#include <string>
#include <vector>

namespace phosphor::terrain {

constexpr u32 kLayerCount = 4;
constexpr u32 kSeed = 23;
constexpr float kPatchSize = 8.0f;       // metres, square patch centred on the origin
constexpr float kHeightAmplitude = 1.1f; // metres, peak to valley of the heightfield
constexpr u32 kMaskSize = 256;           // RGBA8 weight mask and cavity map resolution
constexpr u32 kStandInSize = 256;        // stand-in layer textures

struct Image {
    u32 width = 0, height = 0;
    std::vector<u8> rgba; // tightly packed RGBA8
};

struct LayerSpec {
    const char* name;      // layer material name
    const char* asset;     // ambientCG asset directory
    float uvScale;         // texture tiling over the patch (a power of two)
    float roughness;       // factor on the roughness texture (G)
    float heightContrast;  // metres: amplitude of the height-contrast blend of this layer
    float normalScale;
};

[[nodiscard]] const LayerSpec& layerSpec(u32 layer);

/// Path of one ambientCG map of one layer under `dir`; map is "Color", "NormalGL", "Roughness", "AmbientOcclusion" or "Displacement".
[[nodiscard]] std::string cc0Path(const std::string& dir, u32 layer, const char* map);

/// Height of the patch in metres at uv in [0,1]^2 (u along +x, v along +z).
[[nodiscard]] float heightAt(float u, float v);

/// RGBA8 weight mask: R grass, G rock, B ground, A snow (rules: slope, height, noise).  Texel (x, y) is uv ((x + .5)/size, (y + .5)/size).
[[nodiscard]] Image proceduralMask(u32 size = kMaskSize);

/// Cavity coat map (R = coat weight in [0, 1], G = B = R, A = 255): high where the surface lies below its neighbourhood.
[[nodiscard]] Image proceduralCavity(u32 size = kMaskSize);

/// A constant RGBA8 image (the mask of the endpoint tests).
[[nodiscard]] Image constantImage(std::array<u8, 4> rgba, u32 size = 4);

/// Procedural stand-in textures of one layer.
struct LayerImages {
    Image color;     // sRGB
    Image normal;    // linear, tangent space, OpenGL (+Y up) like glTF and ambientCG NormalGL
    Image roughness; // linear, grey (the G channel is the glTF roughness)
    Image occlusion; // linear, grey
    Image height;    // linear, grey (R)
};
[[nodiscard]] LayerImages proceduralLayer(u32 layer, u32 size = kStandInSize);

/// Terrain patch mesh (grid x grid quads) displaced by heightAt; tangents along +u, handedness +1; uvs span [0,1]^2 times `uvScale`
/// (a power of two keeps the scaled uv exactly equal to uv * uvScale in float).
[[nodiscard]] MeshData makePatch(u32 grid, float uvScale = 1.0f);

} // namespace phosphor::terrain
