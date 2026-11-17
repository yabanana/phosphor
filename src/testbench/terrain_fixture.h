#pragma once

#include "testbench/testbench.h"
#include <array>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace phosphor {

// F15.4 (ticket 23) terrain patch fixture: the layered-material consumer (decision 12).  Bench 2 with
// `--material-scenario terrain[/key=value...]` (optionally `--terrain-textures DIR`, the unpacked CC0 sets of
// tools/fetch_terrain_materials.py; without it the deterministic procedural stand-ins of scene/terrain_textures.h).
//
// `terrain` is an 8 m heightfield patch (scene/terrain_textures.h: heightAt, 128 x 128 quads) whose one material carries a layer
// stack: four layers (grass, rock, ground, snow: ordinary materials of the library with their own textures, tiled by a power-of-two
// uvScale), the generated RGBA weight mask (slope / height / noise rules, seed declared in terrain_textures.h), the height
// contrast of each layer (metres) and, on the stack's own description, a clearcoat "cavity coat" finish (R of the cavity map =
// coat weight: a FEATURE layer, never part of the coverage weights).  One directional sun (3 W/m2) and a constant environment
// (0.25); the camera is fixed (no scripted motion).
//
// Options:
//   weights=a,b,c,d   constant mask (each in [0,1], quantised to 8 bits) instead of the generated one (endpoint tests: 1,0,0,0)
//   layers=K          the stack has only the first K layers (1..4, default 4)
//   single=N          NO stack: the material is layer N alone and the mesh uvs carry its uvScale (the single-material twin of
//                     weights=e_N: same textures, same uv * 2^k, same camera and lights)
//   ior=F             coat=spec only: the twin block's ior in [1, 3] (default 1.5; 1.4 is the wrong-F0 control of the coat-zero gate)
//   height=on|off     height-contrast blend (default on)
//   coat=union       (F15.4, the exact twin of zero) NO coat on the stack, but an UNUSED library material carries a clearcoat block: the scene's feature
//                     union is that of zero, so both run the same pipeline, and a coat that resolves to weight 0 must be this pixel exactly
//   coat=on|off|zero|spec  cavity coat finish (default on; zero = the clearcoat block present with a black cavity map: coat weight 0; spec = NO
//                     coat, a specular block at ior 1.5 and factor 1 - 2^-24 instead: the extended-path twin of zero, F0 0.04 to 6e-8)
//   ramp=on|reverse   FLAT 8 m quad, two plain layers (albedo 0.05 and 0.9, roughness 0.8, dielectric), mask R = 1 - u, G = u (reverse
//                     swaps them); sun only.  With single=0|1 the plain layer alone (the endpoints of the transition)
//   furnace=on        FLAT quad seen at a grazing view, four albedo-1 layers (metallic 0/1/0/1, roughness 0.1/0.4/0.7/1) blended by
//                     a bilinear mask (w0 = (1-u)(1-v), w1 = u(1-v), w2 = (1-u)v, w3 = uv), environment 1, no sun
//   block=coat|sheen|aniso|irid|spec|emissive|thin|all   (F15.4) FLAT quad, two layers of one dielectric (albedo 0.5, roughness 0.6;
//                     layer 0 tiling 1, layer 1 tiling 4) where layer 1 carries one feature block with textures (a checker factor):
//                     the layers' features blend with the weights.  With ramp=on|reverse the ramp mask, with weights=a,b a constant mask
//                     (weights=0,1: endpoint 1), with single=0|1 the layer alone as an ordinary material (the twin); blockless=on removes
//                     the block from layer 1 (control).  Environment 0.5 and the sun; `all` = every block on layer 1
//   layerblocks=on    (F15.4, patch only) the four layers of the textured patch carry feature blocks: 0 clearcoat, 1 sheen, 2 anisotropy + specular/ior,
//                     3 iridescence + thin subsurface: the resolve cost of the blended features at 1..4 active layers (tools/f15_layers_timing.py)
//   hit=on            (F15.4, secondary hits) FLAT quad with two COLOURED layers (green 0.1/0.7/0.15, red 0.85/0.12/0.08, dielectric), a mirror
//                     sphere above it and a white wall behind it: the reflection of the quad in the sphere (RT hits, probe capture) and the bounce on
//                     the wall (GI hits) show the quad's secondary-hit material.  weights=a,b (constant mask), single=0|1 (the layer alone) or owner=on
//                     (the stack's own record alone: the pre-blend secondary-hit appearance, the negative control)
// ramp and furnace ignore height/coat/layers (declared: plain layers, no coat).  Mirrored by tools/f15_layers.py
// (tests/test_terrain_fixture.cpp compares the constants).
//
// SOURCE ONLY / NOT VERIFIED on a device until tools/f15_check.py has captured the keys.
struct TerrainOptions {
    enum class Coat { Off, On, Zero, Spec, Union };
    // F15.4 feature blocks of layer 1 (flat quad): the layers are ordinary material descriptions, so the block blends with the layer.
    enum class Block { None, Coat, Sheen, Aniso, Irid, Spec, Emissive, Thin, All };
    enum class Ramp { None, Forward, Reverse };
    u32 layers = 4;
    bool height = true;
    float specIor = 1.5f; // coat=spec: the ior of the twin block
    Coat coat = Coat::On;
    bool constantMask = false;
    std::array<u8, 4> weights{0, 0, 0, 0};
    int single = -1;
    Ramp ramp = Ramp::None;
    bool furnace = false;
    bool layerBlocks = false; // F15.4 timing/diagnosis: the PATCH's four layers carry blocks (0 coat, 1 sheen, 2 anisotropy + specular, 3 iridescence + thin subsurface)
    bool hit = false;       // F15.4: a mirror sphere and a wall around the flat quad (secondary hits of reflections, probes and GI see the quad)
    bool ownerOnly = false; // hit=on with owner=on: NO stack, the entity is the stack's own record alone (what secondary hits saw before the blend)
    Block block = Block::None;
    bool blockless = false; // block=... with blockless=on: layer 1 WITHOUT the block (the negative control of the equality gates)
};

class TerrainFixture final : public TestBench {
public:
    // `texturesDir` empty = the procedural stand-ins; otherwise the unpacked CC0 sets (setup throws when a file is missing).
    explicit TerrainFixture(std::string scenario, std::string texturesDir = {});
    [[nodiscard]] static bool validScenario(std::string_view scenario);
    [[nodiscard]] static bool parse(std::string_view scenario, TerrainOptions& out, std::string& error);
    // The scenarios the class-manifest export covers (each a valid scenario string).
    [[nodiscard]] static std::span<const std::string_view> scenarios();
    // Radiance of the scenario's declared constant environment (0 for an unknown scenario).
    [[nodiscard]] static float environmentRadiance(std::string_view scenario);

    void setup(ECS&, GpuScene&, TextureManager&) override;
    void update(float, ECS&) override {}
    void teardown(ECS&, GpuScene&) override;
    [[nodiscard]] const char* getName() const override { return name_.c_str(); }
    [[nodiscard]] const char* assetSource() const override { return texturesDir_.empty() ? "procedural" : texturesDir_.c_str(); }
    [[nodiscard]] CameraSetup getDefaultCamera() const override;

    [[nodiscard]] const TerrainOptions& options() const { return options_; }
    // Declared constants (mirrored in tools/f15_layers.py).
    [[nodiscard]] static CameraSetup camera(const TerrainOptions& options);
    [[nodiscard]] static glm::vec3 towardSun(const TerrainOptions& options); // zero for the furnace
    static constexpr float PatchEnvironment = 0.25f;
    static constexpr float SunIrradiance = 3.0f;
    static constexpr u32 PatchGrid = 128;
    static constexpr u32 FlatMaskWidth = 256, FlatMaskHeight = 256;
    static constexpr float RampDark = 0.05f, RampLight = 0.9f, RampRoughness = 0.8f;
    static constexpr float BlockAlbedo = 0.5f, BlockRoughness = 0.6f, BlockEnvironment = 0.5f; // block=... scenes
    static constexpr std::array<float, 2> BlockTiling{1.0f, 4.0f};
    static constexpr std::array<float, 4> FurnaceMetallic{0.0f, 1.0f, 0.0f, 1.0f};
    static constexpr std::array<float, 4> FurnaceRoughness{0.1f, 0.4f, 0.7f, 1.0f};

private:
    std::string scenario_, name_, texturesDir_;
    TerrainOptions options_;
    std::vector<EntityID> entities_;
};

} // namespace phosphor
