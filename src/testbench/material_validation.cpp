#include "testbench/material_validation.h"
#include "testbench/transmission_validation.h"
#include "testbench/chess_fixture.h"
#include "testbench/terrain_fixture.h"
#include "renderer/gpu_scene.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <utility>
#include <glm/gtc/quaternion.hpp>

namespace phosphor {
namespace {
// The names are mirrored by tools/f15_check.py SCENARIOS (tests/test_material_validation.cpp compares both).
constexpr std::array<std::string_view, 67> kScenarios{"base-grid-sun",  "base-grid-env",       "base-grid-env-resolve",
                                                      "furnace-base",   "furnace-resolve",     "roughness-extremes",
                                                      "roughness-extremes-sun", "specular-grid-sun",
                                                      "specular-grid-env-resolve", "specular-grid-point", "specular-grid-env", "furnace-specular",
                                                      "clearcoat-grid-sun", "clearcoat-grid-env-resolve", "clearcoat-grid-env",
                                                      "clearcoat-grid-point", "furnace-clearcoat",
                                                      "specular-grid-edit", "specular-grid-rough",
                                                      "sheen-grid-sun", "sheen-grid-env-resolve", "sheen-grid-env",
                                                      "sheen-grid-point", "furnace-sheen",
                                                      "aniso-grid-sun", "aniso-grid-env-resolve", "aniso-grid-env",
                                                      "aniso-grid-point", "furnace-aniso", "aniso-mirrored-sun",
                                                      "iridescence-grid-sun", "iridescence-grid-env-resolve", "iridescence-grid-env",
                                                      "iridescence-grid-point", "furnace-iridescence",
                                                      "subsurface-thin-sun", "subsurface-thin-env-resolve", "subsurface-thin-env",
                                                      "subsurface-thin-point", "furnace-thin-subsurface",
                                                      "subsurface-volumetric-sun", "subsurface-volumetric-radius",
                                                      "subsurface-volumetric-subpixel", "subsurface-volumetric-subpixel-off",
                                                      "subsurface-volumetric-backlit",
                                                      "combo-coat-sheen-sun", "combo-coat-sheen-env", "combo-aniso-coat-sun",
                                                      "combo-aniso-coat-env", "combo-aniso-sheen-sun", "combo-aniso-sheen-env",
                                                      "combo-specular-sheen-sun", "combo-specular-sheen-env", "combo-irid-coat-sun",
                                                      "combo-irid-coat-env", "combo-irid-sheen-sun", "combo-irid-sheen-env",
                                                      "combo-irid-aniso-sun", "combo-irid-aniso-env", "combo-irid-specular-sun",
                                                      "combo-irid-specular-env", "combo-thin-coat-sun", "combo-thin-coat-env",
                                                      "combo-all-lobes-sun", "combo-all-lobes-env", "furnace-combinations",
                                                      "subsurface-thin-gi-back"};
constexpr float pi = 3.14159265358979323846f;

glm::quat orientLight(glm::vec3 towardLight) {
    // Same convention as the lighting fixtures: local -Z through the entity
    // matrix is the direction the light travels; the renderer negates it.
    const glm::vec3 direction = -glm::normalize(towardLight), forward(0, 0, -1);
    const float cosine = std::clamp(glm::dot(forward, direction), -1.0f, 1.0f);
    if (cosine > 0.999999f) return glm::quat(1, 0, 0, 0);
    if (cosine < -0.999999f) return glm::angleAxis(pi, glm::vec3(0, 1, 0));
    return glm::angleAxis(std::acos(cosine), glm::normalize(glm::cross(forward, direction)));
}

// F15.3 (ticket 19): the critical lobe combinations (tools/f15_reference.py composes them from the layering text of
// research/khr-materials.md section 12).  Every combination is a grid of texture-free spheres whose cells carry the lobes in play and
// their neutral neighbours (a cell with one lobe neutral reduces to the other, so the same grid also pins the single lobes).
using Cell = MaterialScenarioSpec::ComboCell;
// The scenario name as the string_view of its literal in kScenarios (the specs hold views; a name that is not declared there is a bug).
std::string_view declaredName(const std::string& wanted) {
    for (const std::string_view name : kScenarios)
        if (name == wanted) return name;
    throw std::logic_error("combination scenario not declared in kScenarios: " + wanted);
}
Cell withSpecular(Cell c, float ior, float factor, glm::vec3 color) {
    c.hasSpecular = true; c.ior = ior; c.specularFactor = factor; c.specularColor = color;
    return c;
}
Cell withCoat(Cell c, float factor, float roughness) { c.coatFactor = factor; c.coatRoughness = roughness; return c; }
Cell withSheen(Cell c, glm::vec3 color, float roughness) { c.sheenColor = color; c.sheenRoughness = roughness; return c; }
Cell withAniso(Cell c, float strength, float rotation) { c.anisoStrength = strength; c.anisoRotation = rotation; return c; }
Cell withIrid(Cell c, float factor, float ior, float thickness) {
    c.iridFactor = factor; c.iridIor = ior; c.iridThickness = thickness;
    return c;
}
Cell withThin(Cell c, float weight, glm::vec3 color) { c.thinWeight = weight; c.thinColor = color; return c; }

struct ComboLook {
    std::vector<float> roughness; // per column
    std::vector<float> metallic;  // per row
    glm::vec3          base;
    std::vector<Cell>  cells;     // rows x columns, row-major
    bool               combed = false;
    glm::vec3          towardSun{-0.8f, 0.35f, -0.3f};
};

// One combination: `name`-sun (sun only: the direct gate) and `name`-env (environment 0.5 only: the exact quadrature through the
// transport, and, captured with --reflections off, the declared resolve model).  The grids share the base-grid rig.
void addCombination(std::vector<MaterialScenarioSpec>& out, const std::string& name, const ComboLook& look) {
    MaterialScenarioSpec sun;
    sun.name           = declaredName("combo-" + name + "-sun");
    sun.roughness      = look.roughness;
    sun.metallic       = look.metallic;
    sun.baseColor      = look.base;
    sun.sun            = true;
    sun.sunIrradiance  = 3.0f;
    sun.towardSun      = glm::normalize(look.towardSun);
    sun.combo          = look.cells;
    sun.combedCarriers = look.combed;
    out.push_back(sun);
    MaterialScenarioSpec env = sun;
    env.name          = declaredName("combo-" + name + "-env");
    env.sun           = false;
    env.sunIrradiance = 0.0f;
    env.environment   = 0.5f;
    out.push_back(env);
}

void addCombinations(std::vector<MaterialScenarioSpec>& out) {
    constexpr float quarter = 0.78539816f, half = 1.57079633f;
    const glm::vec3 white(1.0f), orange(1.0f, 0.5f, 0.25f), blue(0.25f, 0.5f, 1.0f), warm(1.0f, 0.55f, 0.25f), teal(0.1f, 0.7f, 0.6f);
    const glm::vec3 copper(0.8f, 0.35f, 0.15f), darkCopper(0.3f, 0.15f, 0.08f), dark(0.1f);
    const Cell none{};
    auto grid = [](size_t rows, size_t cols, auto&& make) {
        std::vector<Cell> cells;
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) cells.push_back(make(r, c));
        return cells;
    };
    // Clearcoat + sheen (no Khronos sample has it): rows = sheen (white 0.4, orange 0.75) over a dielectric, blue 0.5 over a metal;
    // columns = the coat (none, 0.5 / 0.3, 1 / 0.3, 1 / 0.6).  Dark rough base: the sheen is the visible feature, the coat over it.
    {
        const std::vector<std::pair<glm::vec3, float>> sheens{{white, 0.4f}, {orange, 0.75f}, {blue, 0.5f}};
        const std::vector<std::pair<float, float>>     coats{{0.0f, 0.0f}, {0.5f, 0.3f}, {1.0f, 0.3f}, {1.0f, 0.6f}};
        addCombination(out, "coat-sheen",
                       {std::vector<float>(4, 0.7f), {0.0f, 0.0f, 1.0f}, dark, grid(3, 4, [&](size_t r, size_t c) {
                            Cell cell = withSheen(none, sheens[r].first, sheens[r].second);
                            return coats[c].first > 0.0f ? withCoat(cell, coats[c].first, coats[c].second) : cell;
                        })});
    }
    // Anisotropy + clearcoat: rows = dielectric, metal (copper); columns = (strength 0.5 at pi/4, coat 1 / 0.3), (1 at 0, coat 1 / 0.3),
    // (1 at pi/2, coat 0.5 / 0.35), (no anisotropy, coat 1 / 0.3).
    addCombination(out, "aniso-coat",
                   {std::vector<float>(4, 0.3f), {0.0f, 1.0f}, copper, grid(2, 4, [&](size_t, size_t c) {
                        switch (c) {
                            case 0: return withCoat(withAniso(none, 0.5f, quarter), 1.0f, 0.3f);
                            case 1: return withCoat(withAniso(none, 1.0f, 0.0f), 1.0f, 0.3f);
                            case 2: return withCoat(withAniso(none, 1.0f, half), 0.5f, 0.35f);
                            default: return withCoat(none, 1.0f, 0.3f);
                        }
                    }), true});
    // Anisotropy + sheen (carried from ticket 11): rows = dielectric, metal; columns = the anisotropic lobe under a sheen.
    addCombination(out, "aniso-sheen",
                   {std::vector<float>(4, 0.3f), {0.0f, 1.0f}, darkCopper, grid(2, 4, [&](size_t, size_t c) {
                        switch (c) {
                            case 0: return withSheen(withAniso(none, 0.5f, quarter), white, 0.4f);
                            case 1: return withSheen(withAniso(none, 1.0f, 0.0f), orange, 0.4f);
                            case 2: return withSheen(withAniso(none, 1.0f, half), white, 0.75f);
                            default: return withSheen(none, white, 0.4f);
                        }
                    }), true});
    // Specular + sheen: rows = sheen (white 0.4, orange 0.75); columns = (sheen alone), (ior 2.4), (ior 1.33), (ior 1.5, weight 0.5, colour 2 1 .5).
    {
        const std::vector<std::pair<glm::vec3, float>> sheens{{white, 0.4f}, {orange, 0.75f}};
        addCombination(out, "specular-sheen",
                       {std::vector<float>(4, 0.35f), {0.0f, 0.0f}, glm::vec3(0.2f), grid(2, 4, [&](size_t r, size_t c) {
                            Cell cell = withSheen(none, sheens[r].first, sheens[r].second);
                            switch (c) {
                                case 0: return cell; // the sheen alone (the neutral extension)
                                case 1: return withSpecular(cell, 2.4f, 1.0f, white);
                                case 2: return withSpecular(cell, 1.33f, 1.0f, white);
                                default: return withSpecular(cell, 1.5f, 0.5f, glm::vec3(2.0f, 1.0f, 0.5f));
                            }
                        })});
    }
    // Iridescence + clearcoat (the interface between coat and film is ignored by the specification): rows = dielectric, metal;
    // columns = (film 400 nm, coat 1 / 0.3), (800 nm, coat 1 / 0.3), (film ior 1.8, 400 nm, coat 0.5 / 0.35), (film only).
    addCombination(out, "irid-coat",
                   {std::vector<float>(4, 0.3f), {0.0f, 1.0f}, copper, grid(2, 4, [&](size_t, size_t c) {
                        switch (c) {
                            case 0: return withCoat(withIrid(none, 1.0f, 1.3f, 400.0f), 1.0f, 0.3f);
                            case 1: return withCoat(withIrid(none, 1.0f, 1.3f, 800.0f), 1.0f, 0.3f);
                            case 2: return withCoat(withIrid(none, 1.0f, 1.8f, 400.0f), 0.5f, 0.35f);
                            default: return withIrid(none, 1.0f, 1.3f, 400.0f);
                        }
                    })});
    // Iridescence + sheen: rows = dielectric, metal; columns = film under a sheen, the last one the film alone.
    addCombination(out, "irid-sheen",
                   {std::vector<float>(4, 0.3f), {0.0f, 1.0f}, darkCopper, grid(2, 4, [&](size_t, size_t c) {
                        switch (c) {
                            case 0: return withSheen(withIrid(none, 1.0f, 1.3f, 400.0f), white, 0.4f);
                            case 1: return withSheen(withIrid(none, 1.0f, 1.3f, 800.0f), orange, 0.4f);
                            case 2: return withSheen(withIrid(none, 1.0f, 1.8f, 400.0f), white, 0.75f);
                            default: return withIrid(none, 1.0f, 1.3f, 600.0f); // the film alone
                        }
                    })});
    // Iridescence + anisotropy (carried from tickets 11/12): the exact environment integrates the film over V.H histograms of the
    // anisotropic half-vector quadrature (tools/f15_reference.py aniso_vh_histograms).
    addCombination(out, "irid-aniso",
                   {std::vector<float>(4, 0.3f), {0.0f, 1.0f}, copper, grid(2, 4, [&](size_t, size_t c) {
                        switch (c) {
                            case 0: return withAniso(withIrid(none, 1.0f, 1.3f, 400.0f), 0.5f, quarter);
                            case 1: return withAniso(withIrid(none, 1.0f, 1.3f, 800.0f), 1.0f, 0.0f);
                            case 2: return withAniso(withIrid(none, 1.0f, 1.3f, 400.0f), 1.0f, half);
                            default: return withIrid(none, 1.0f, 1.3f, 600.0f);
                        }
                    }), true});
    // Iridescence + ior/specular: rows = (400 nm, film ior 1.3), (800 nm, 1.3), (400 nm, film ior 1.8) over dielectrics, then a metal
    // (400 nm: the extension has no effect on a metal); columns = (film alone), (ior 1.8, weight 0.7, colour 1 .8 .6), (ior 2.4), (ior 1.5,
    // weight 0.5, colour 2 1 .5).  Strength 1 everywhere (for a fractional strength the specification's two forms of the diffuse weight
    // differ, see f15_reference.py irid_lobe_weights).
    {
        const std::vector<std::tuple<float, float, float>> films{{1.3f, 400.0f, 0.0f}, {1.3f, 800.0f, 0.0f}, {1.8f, 400.0f, 0.0f}, {1.3f, 400.0f, 1.0f}};
        addCombination(out, "irid-specular",
                       {std::vector<float>(4, 0.3f), {0.0f, 0.0f, 0.0f, 1.0f}, copper, grid(4, 4, [&](size_t r, size_t c) {
                            Cell cell = withIrid(none, 1.0f, std::get<0>(films[r]), std::get<1>(films[r]));
                            switch (c) {
                                case 0: return cell; // the film alone (the neutral extension)
                                case 1: return withSpecular(cell, 1.8f, 0.7f, glm::vec3(1.0f, 0.8f, 0.6f));
                                case 2: return withSpecular(cell, 2.4f, 1.0f, white);
                                default: return withSpecular(cell, 1.5f, 0.5f, glm::vec3(2.0f, 1.0f, 0.5f));
                            }
                        })});
    }
    // Thin subsurface + clearcoat: rows = thin weight 0.5 (warm), 1 (teal); columns = coat (none), (1 / 0.3), (0.5 / 0.35), (1 / 0.6);
    // sun behind the spheres (the thin-sun rig): the transmitted lobe under the coat.
    addCombination(out, "thin-coat",
                   {std::vector<float>(4, 0.5f), {0.0f, 0.0f}, glm::vec3(0.3f, 0.5f, 0.8f), grid(2, 4, [&](size_t r, size_t c) {
                        Cell cell = withThin(none, r == 0 ? 0.5f : 1.0f, r == 0 ? warm : teal);
                        switch (c) {
                            case 1: return withCoat(cell, 1.0f, 0.3f);
                            case 2: return withCoat(cell, 0.5f, 0.35f);
                            case 3: return withCoat(cell, 1.0f, 0.6f);
                            default: return cell;
                        }
                    }), false, glm::normalize(glm::vec3(-0.55f, 0.4f, -0.73f))});
    // Everything but the film at once (ior/specular + coat + sheen + anisotropy): rows = dielectric, metal.
    addCombination(out, "all-lobes",
                   {std::vector<float>(4, 0.3f), {0.0f, 1.0f}, copper, grid(2, 4, [&](size_t, size_t c) {
                        switch (c) {
                            case 0: return withAniso(withSheen(withCoat(withSpecular(none, 1.8f, 0.7f, glm::vec3(1.0f, 0.8f, 0.6f)), 0.5f, 0.35f), white, 0.4f), 0.5f, quarter);
                            case 1: return withAniso(withSheen(withCoat(withSpecular(none, 1.8f, 0.7f, glm::vec3(1.0f, 0.8f, 0.6f)), 0.5f, 0.35f), white, 0.4f), 1.0f, 0.0f);
                            case 2: return withAniso(withSheen(withCoat(withSpecular(none, 2.4f, 1.0f, white), 1.0f, 0.3f), orange, 0.75f), 1.0f, half);
                            default: return withSheen(withCoat(withSpecular(none, 1.5f, 0.5f, glm::vec3(2.0f, 1.0f, 0.5f)), 1.0f, 0.6f), blue, 0.4f);
                        }
                    }), true});
    // Furnace over every implemented combination: white base, environment 1, no sun.  Rows 0-1 dielectric, rows 2-3 metal; the ten
    // combinations run over (row % 2, column); the column roughness varies the base lobe.
    {
        const std::vector<Cell> combos{
            withSheen(withCoat(none, 1.0f, 0.3f), white, 0.4f),                                                           // coat + sheen
            withCoat(withAniso(none, 1.0f, quarter), 1.0f, 0.3f),                                                         // aniso + coat
            withSheen(withAniso(none, 1.0f, quarter), white, 0.4f),                                                       // aniso + sheen
            withSheen(withSpecular(none, 1.8f, 0.7f, glm::vec3(1.0f, 0.8f, 0.6f)), white, 0.4f),                          // specular + sheen
            withCoat(withIrid(none, 1.0f, 1.3f, 400.0f), 1.0f, 0.3f),                                                     // irid + coat
            withSheen(withIrid(none, 1.0f, 1.3f, 400.0f), white, 0.4f),                                                   // irid + sheen
            withAniso(withIrid(none, 1.0f, 1.3f, 400.0f), 1.0f, quarter),                                                 // irid + aniso
            withSpecular(withIrid(none, 1.0f, 1.3f, 400.0f), 1.8f, 0.7f, glm::vec3(1.0f, 0.8f, 0.6f)),                    // irid + specular
            withThin(withCoat(none, 1.0f, 0.3f), 1.0f, white),                                                            // thin + coat
            withAniso(withSheen(withCoat(withIrid(withSpecular(none, 1.8f, 0.7f, glm::vec3(1.0f, 0.8f, 0.6f)), 1.0f, 1.3f, 400.0f), 1.0f, 0.3f), white, 0.4f), 1.0f, quarter)};
        MaterialScenarioSpec furnace;
        furnace.name           = declaredName("furnace-combinations");
        furnace.roughness      = {0.2f, 0.35f, 0.5f, 0.7f, 0.9f};
        furnace.metallic       = {0.0f, 0.0f, 1.0f, 1.0f};
        furnace.baseColor      = glm::vec3(1.0f);
        furnace.environment    = 1.0f;
        furnace.combedCarriers = true;
        furnace.combo          = grid(4, 5, [&](size_t r, size_t c) { return combos[(r % 2) * 5 + c]; });
        out.push_back(furnace);
    }
}

const std::vector<MaterialScenarioSpec>& specs() {
    // roughness 0.1 and 0.2 have alpha < 0.05: the sun disk uses the declared
    // spiral quadrature there (the reference's quadrature mask).
    static const std::vector<float> roughness{0.1f, 0.2f, 0.35f, 0.5f, 0.7f, 0.85f, 1.0f};
    static const std::vector<float> metallic{0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    static const glm::vec3 copper(0.8f, 0.35f, 0.15f);
    // Side-lit and slightly from behind (V.L about -0.33): the half vector sits
    // far from the view direction, so Schlick's (1 - V.H)^5 is non-negligible
    // and a missing or wrong Fresnel term is visible in the direct grid.
    static const glm::vec3 towardSun = glm::normalize(glm::vec3(-0.8f, 0.35f, -0.3f));
    // F15.1 (ticket 05): KHR_materials_ior x KHR_materials_specular on texture-free spheres.  Columns
    // are IORs (0 = the permanent Fresnel = 1 mode, 1.0 f0 = 0, 1.33 water, 1.5 the default, 2.4
    // diamond), rows are (specularFactor, specularColor): the neutral pair, a weight, a colour above 1
    // (f0 clamped before the weight), both; the last row is metallic 1, on which both extensions must
    // have no effect.  One roughness (0.35, alpha 0.12: wide lobe, no sun quadrature) keeps the grid small.
    static const std::vector<float> specIor{0.0f, 1.0f, 1.33f, 1.5f, 2.4f};
    static const std::vector<float> specRoughness(5, 0.35f);
    static const std::vector<float> specMetallic{0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    static const std::vector<float> specFactor{1.0f, 0.5f, 1.0f, 0.5f, 0.5f};
    static const std::vector<glm::vec3> specColor{glm::vec3(1.0f), glm::vec3(1.0f), glm::vec3(2.0f),
                                                  glm::vec3(2.0f, 1.0f, 0.5f), glm::vec3(2.0f)};
    // F15.1 (ticket 07): the same grid lit by one point light, side-lit from behind like the sun (V.L about
    // -0.5 at the grid centre, so Fresnel discriminates f90 and the specular weight).  Spheres may shadow
    // their neighbours: DI traces that visibility (the reference casts the same shadow rays), the legacy
    // resolve's local lights are unshadowed (reference without visibility).  9 m from the grid centre,
    // 250 W/sr gives about the sun's 3 W/m^2; range 100 m keeps the engine's window within 5e-4 of 1
    // (declared exactly by the reference).
    static const glm::vec3 gridCentre(0.0f, 2.0f, 0.0f);
    static const MaterialPointLight backLight{gridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                              250.0f, 100.0f};
    // F15.3 (ticket 09): KHR_materials_clearcoat on texture-free spheres.  Rows are the coat factor (0 = no coat, the
    // neutral material; 0.25, 0.5, 1), then coat 1 over a base of ior 2.4 (a different base F0 than the coat's fixed
    // 0.04: the Sample Renderer deviation "coat F0 from the material ior" is only visible there), then a metal under a
    // full coat.  Columns are the coat roughness: 0 and 0.05 (alpha 0.002 and 0.0025) and 0.15 (alpha 0.0225) exercise the
    // sun disk's narrow-lobe quadrature, 0.4 and 1.0 are wide.  The base is dielectric copper at roughness 0.6: a rough
    // base under a smooth coat, so the coat reflection is the sharp feature the denoiser must keep.
    static const std::vector<float> coatRoughness{0.0f, 0.05f, 0.15f, 0.4f, 1.0f};
    static const std::vector<float> coatFactor{0.0f, 0.25f, 0.5f, 1.0f, 1.0f, 1.0f};
    static const std::vector<float> coatBaseIor{0.0f, 0.0f, 0.0f, 0.0f, 2.4f, 0.0f}; // 0 = no ior block in that row
    static const std::vector<float> coatBaseRoughness(5, 0.6f);
    static const std::vector<float> coatMetallic{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    static const glm::vec3 coatGridCentre(0.0f, 2.5f, 0.0f);
    static const MaterialPointLight coatBackLight{coatGridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                                  250.0f, 100.0f};
    // F15.3 (ticket 10): KHR_materials_sheen on texture-free spheres.  Rows are the sheen colour (0 = no sheen, the neutral
    // material; white; an orange and a blue to see the colour of the fuzz and the max3 albedo scaling), columns are the
    // sheen roughness 0.1 ... 1 (alpha_g = r^2 from 0.01 to 1).  The base is a dark, rough dielectric (grey 0.1,
    // roughness 0.7) so the sheen is the visible feature; the grid is lit from the side and slightly from behind (the
    // base-grid rig), where the Charlie lobe (which peaks at grazing half vectors) is strongest.
    static const std::vector<glm::vec3> sheenColors{glm::vec3(0.0f), glm::vec3(1.0f), glm::vec3(1.0f, 0.5f, 0.25f),
                                                    glm::vec3(0.25f, 0.5f, 1.0f)};
    static const std::vector<float> sheenRoughnesses{0.1f, 0.3f, 0.5f, 0.75f, 1.0f};
    static const std::vector<float> sheenBaseRoughness(5, 0.7f);
    static const std::vector<float> sheenMetallic(4, 0.0f);
    static const glm::vec3 sheenBase(0.1f);
    static const glm::vec3 sheenGridCentre(0.0f, 1.5f, 0.0f);
    static const MaterialPointLight sheenBackLight{sheenGridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                                   250.0f, 100.0f};
    // F15.3 (ticket 11): KHR_materials_anisotropy on texture-free COMBED spheres (a uniform tangent field facing the
    // camera, handedness -1).  Rows are the strength {0 (no extension), 0.5, 1} over a dielectric (copper base) and then over
    // a metal; columns are the rotation {0, pi/4, pi/2, -pi/4}: 0 and pi/2 stretch the highlight along and across the comb,
    // +-pi/4 are the two diagonals (the pair that tells the rotation direction).  Base roughness 0.3 (alpha 0.09: a wide
    // lobe for the sun disk rule, no quadrature mask), base copper.
    static const std::vector<float> anisoStrength{0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 1.0f};
    static const std::vector<float> anisoMetallic{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    static const std::vector<float> anisoRotation{0.0f, 0.78539816f, 1.57079633f, -0.78539816f};
    static const std::vector<float> anisoRoughness(4, 0.3f);
    static const glm::vec3 anisoGridCentre(0.0f, 2.5f, 0.0f);
    static const MaterialPointLight anisoBackLight{anisoGridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                                   250.0f, 100.0f};
    // The mirrored carrier fixture: two rows (dielectric, metal) at strength 1, every column at rotation pi/4; columns
    // alternate plain and MIRRORED instances (scale x = -1) over meshes of handedness -1 then +1, so the rotation direction,
    // the tangent handedness w and the world-matrix determinant each flip the lobe on their own.
    static const std::vector<float> mirrorStrength{1.0f, 1.0f};
    static const std::vector<float> mirrorMetallic{0.0f, 1.0f};
    static const std::vector<float> mirrorRotation(4, 0.78539816f);
    static const std::vector<bool>  mirrorColumns{false, true, false, true};
    static const std::vector<float> mirrorHandedness{-1.0f, -1.0f, 1.0f, 1.0f};
    // F15.3 (ticket 12): KHR_materials_iridescence on texture-free spheres.  Rows are (factor, film ior) over a copper
    // dielectric (rows 0-2) and then a copper metal (rows 3-5): (0.5, 1.3), (1, 1.3), (1, 1.8); columns are the thickness
    // {0, 200, 400, 600, 800, 1200} nm (the MAXIMUM, the specification's thickness without a thickness texture; the minimum keeps
    // its default 100 nm so that a control reading the minimum differs in every column).  Thickness 0 disables the film (the
    // neutral column).  Base roughness 0.3 (alpha 0.09: a wide lobe for the sun disk rule, no quadrature mask).  Same rig as the
    // other grids (sun 3 W/m2, 250 W/sr point light 9 m from the grid centre, V.L about -0.5).
    static const std::vector<float> iridFactor{0.5f, 1.0f, 1.0f, 0.5f, 1.0f, 1.0f};
    static const std::vector<float> iridIor{1.3f, 1.3f, 1.8f, 1.3f, 1.3f, 1.8f};
    static const std::vector<float> iridThickness{0.0f, 200.0f, 400.0f, 600.0f, 800.0f, 1200.0f};
    static const std::vector<float> iridMetallic{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    static const std::vector<float> iridRoughness(6, 0.3f);
    static const glm::vec3 iridGridCentre(0.0f, 2.5f, 0.0f);
    static const MaterialPointLight iridBackLight{iridGridCentre + 9.0f * glm::normalize(glm::vec3(-0.65f, 0.55f, -0.52f)),
                                                  250.0f, 100.0f};
    // F15.3 (ticket 16): the thin subsurface (diffuse transmission) lobe.  Rows are the weight {0 (no extension, the neutral
    // material), 0.25, 0.5, 1}, columns the transmission colour {white, warm, teal}, over a blue-grey base (so the reflected and the
    // transmitted colour are told apart) of roughness 0.5.  The sun sits BEHIND the spheres (V.L about -0.73 at the grid centre):
    // the visible face is lit through the transmitted lobe, a crescent on the upper left from the front (the reflected lobe,
    // reduced by 1 - w).  Spheres are used for the sun and environment scenarios (N.V from 0 to 1 for the environment tables); the
    // point-light scenario uses single-sided quads (no self-occlusion of the DI shadow rays) and has no weight-0 row (a row that
    // nothing reaches would have no lit pixels; the sun scenario carries the neutral row).
    static const std::vector<float> thinRoughness(3, 0.5f);
    static const std::vector<float> thinMetallic4(4, 0.0f);
    static const std::vector<float> thinMetallic3(3, 0.0f);
    static const std::vector<float> thinWeights4{0.0f, 0.25f, 0.5f, 1.0f};
    static const std::vector<float> thinWeights3{0.25f, 0.5f, 1.0f};
    static const std::vector<glm::vec3> thinColors{glm::vec3(1.0f), glm::vec3(1.0f, 0.55f, 0.25f), glm::vec3(0.1f, 0.7f, 0.6f)};
    static const std::vector<glm::vec3> thinWhite(3, glm::vec3(1.0f));
    static const std::vector<float> thinFurnaceRoughness{0.2f, 0.5f, 0.9f};
    static const glm::vec3 thinBase(0.3f, 0.5f, 0.8f);
    static const glm::vec3 thinSun = glm::normalize(glm::vec3(-0.55f, 0.4f, -0.73f));
    static const glm::vec3 thinPointCentre(0.0f, 1.0f, 0.0f);
    static const MaterialPointLight thinBackLight{thinPointCentre + 9.0f * glm::normalize(glm::vec3(-0.55f, 0.4f, -0.73f)), 250.0f,
                                                  100.0f};
    // F15.3 (ticket 17): the volumetric subsurface mode (screen-space diffusion).  Texture-free spheres of radius 0.4 m on a
    // light grey base (0.8) of roughness 0.35 (alpha_b 0.1225: the largest roughness at which the Mitsuba rough dielectric lets out more than the brightest fixture colour 0.85, A_max 0.887; at 0.7 it returns only 0.507 and the reference is declared invalid), lit
    // by a sun from the SIDE (toward the sun (-0.9, 0.25, 0.35): the terminator crosses the visible disc to the right of its
    // centre).  Rows are the material: skin-like (multiple-scattering colour (0.85, 0.55, 0.45), radiusScale (1, 0.5, 0.25),
    // isotropic) and jade-like (0.30, 0.70, 0.45), (0.5, 1, 0.7), g = 0.3; columns the (weight, radius): {0 (neutral), 0.5 @ 0.08 m,
    // 1 @ 0.04 m, 1 @ 0.08 m, 1 @ 0.16 m}, i.e. weight {0, 0.5, 1} at the nominal radius 0.08 m and the radius sweep x0.5 / x1 / x2
    // at weight 1.  The radius sweep scene shares columns 2-4 with it (identical pieces: the reference render is cached across both)
    // and adds x0.25 and x3.  The mean free path of the dominant (green) channel at the nominal radius is 0.04 m (4.5 px at the
    // 5.5 m camera).
    static const std::vector<float> volRoughness(5, 0.35f);
    static const std::vector<float> volMetallic2(2, 0.0f);
    static const std::vector<float> volWeights{0.0f, 0.5f, 1.0f, 1.0f, 1.0f};
    static const std::vector<float> volRadii{0.08f, 0.08f, 0.04f, 0.08f, 0.16f};
    static const std::vector<float> volSweepRadii{0.02f, 0.24f, 0.04f, 0.08f, 0.16f};
    static const std::vector<float> volSweepWeights(5, 1.0f);
    static const std::vector<glm::vec3> volColors{glm::vec3(0.85f, 0.55f, 0.45f), glm::vec3(0.30f, 0.70f, 0.45f)};
    static const std::vector<glm::vec3> volScales{glm::vec3(1.0f, 0.5f, 0.25f), glm::vec3(0.5f, 1.0f, 0.7f)};
    static const std::vector<float> volAnisotropy{0.0f, 0.3f};
    static const glm::vec3 volBase(0.8f);
    static const glm::vec3 volSun = glm::normalize(glm::vec3(-0.9f, 0.25f, 0.35f));
    // Sub-pixel fixture: weight {1, 1, 0.5, 0.5} x radius {0.0005, 0.002, 0.0005, 0.002} m (the mean free path is at most 0.002 m =
    // 0.23 px at the 5.5 m camera), subsurface colour = the base colour (so the "diffusion skipped" image IS the base model), g = 0.
    // `-off` is the identical scene without any block: the runner compares the two engine captures.
    static const std::vector<float> subRoughness(4, 0.35f);
    static const std::vector<float> subMetallic1(1, 0.0f);
    static const std::vector<float> subWeights{1.0f, 1.0f, 0.5f, 0.5f};
    static const std::vector<float> subRadii{0.0005f, 0.002f, 0.0005f, 0.002f};
    static const std::vector<glm::vec3> subColors{volBase};
    static const std::vector<glm::vec3> subScales{glm::vec3(1.0f, 0.5f, 0.25f)};
    static const std::vector<float> subAnisotropy{0.0f};
    // Back-lit thick fixture: the sun BEHIND the spheres (V.L about -0.96), weight 1, radius {0.08, 0.16} m, both materials.  Measured
    // and reported, not gated: screen-space diffusion cannot carry light through a thick volume (a declared limit).
    static const std::vector<float> backRoughness(2, 0.35f);
    static const std::vector<float> backWeights(2, 1.0f);
    static const std::vector<float> backRadii{0.08f, 0.16f};
    static const glm::vec3 backSun = glm::normalize(glm::vec3(0.15f, 0.25f, -0.96f));
    static const std::vector<MaterialScenarioSpec> baseSpecs{
        {"base-grid-sun", roughness, metallic, copper, true, 3.0f, towardSun, 0.0f},
        {"base-grid-env", roughness, metallic, copper, false, 0.0f, towardSun, 0.5f},
        // The same scene as base-grid-env without the specular transport: the resolve
        // evaluates the declared environment itself (run with --reflections off
        // --lighting-denoise custom); gated against the declared-model reference.
        {"base-grid-env-resolve", roughness, metallic, copper, false, 0.0f, towardSun, 0.5f},
        {"furnace-base", roughness, metallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f},
        // Same scene without the specular transport: the resolve evaluates the
        // declared environment deterministically (run with --reflections off).
        {"furnace-resolve", roughness, metallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f},
        // Roughness 0 is the declared clamp (0.04); 1 is the upper bound.
        {"roughness-extremes", {0.0f, 1.0f}, {0.0f, 0.5f, 1.0f}, copper, true, 3.0f, towardSun, 0.3f},
        // The same spheres lit by the sun only (environment 0): the direct gate applies.
        {"roughness-extremes-sun", {0.0f, 1.0f}, {0.0f, 0.5f, 1.0f}, copper, true, 3.0f, towardSun, 0.0f},
        // Sun only: the direct gate applies (the extension lobes of the resolve and forward paths).
        {"specular-grid-sun", specRoughness, specMetallic, copper, true, 3.0f, towardSun, 0.0f, specIor, specFactor,
         specColor},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the
        // declared split-sum model f0 * A + f90 * B per lobe.
        {"specular-grid-env-resolve", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.5f, specIor,
         specFactor, specColor},
        // One point light only (no sun, environment 0): the resolve's local lights with --lighting legacy,
        // ReSTIR or clustered DI otherwise (the DI grid of ticket 07).
        {"specular-grid-point", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.0f, specIor,
         specFactor, specColor, {backLight}},
        // F15.1 (ticket 08): the same grid and environment as specular-grid-env-resolve with the specular transport on
        // (SSR by default; the runner also captures it with RT reflections and with the probe path).
        {"specular-grid-env", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.5f, specIor, specFactor,
         specColor},
        // Furnace over EXTENDED materials: the ior x specular grid at base colour 1 under a uniform environment of
        // radiance 1 (the runner also captures it without the transport, deterministic resolve terms).
        {"furnace-specular", specRoughness, specMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, specIor,
         specFactor, specColor},
        // F15.3 (ticket 09) clearcoat grids.  Sun only: the direct gate (coat lobe in the sun-disk quadrature).
        {"clearcoat-grid-sun", coatBaseRoughness, coatMetallic, copper, true, 3.0f, towardSun, 0.0f, {}, {}, {}, {},
         coatFactor, coatRoughness, coatBaseIor},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the declared coat term.
        {"clearcoat-grid-env-resolve", coatBaseRoughness, coatMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {},
         {}, coatFactor, coatRoughness, coatBaseIor},
        // The same scene with the specular transport on: the coat term in the environment, the coat lobe in the transport.
        {"clearcoat-grid-env", coatBaseRoughness, coatMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {},
         coatFactor, coatRoughness, coatBaseIor},
        // One point light only (no sun, environment 0), side-lit from behind like specular-grid-point.
        {"clearcoat-grid-point", coatBaseRoughness, coatMetallic, copper, false, 0.0f, towardSun, 0.0f, {}, {}, {},
         {coatBackLight}, coatFactor, coatRoughness, coatBaseIor},
        // Furnace over coated materials: albedo 1 under a uniform environment of radiance 1 (the runner also captures it
        // without the transport).
        {"furnace-clearcoat", coatBaseRoughness, coatMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, {}, {}, {},
         {}, coatFactor, coatRoughness, coatBaseIor},
        // Clip fixture of ticket 08 (not gated against the reference): the grid and environment of specular-grid-env
        // with the strong runtime edit of --material-toggle-every (tools/f15_clip_check.py: recovery after an edit).
        {"specular-grid-edit", specRoughness, specMetallic, copper, false, 0.0f, towardSun, 0.5f, specIor, specFactor,
         specColor, {}, {}, {}, {}, true},
        // Adaptive-shading fixture of ticket 08 (tools/f15_clip_check.py): rough (>= 0.75) dielectric spheres, the
        // reuse domain of visibility_adaptive, with every extension of the grid, lit by the sun (plain resolve).
        {"specular-grid-rough", std::vector<float>(5, 0.85f), std::vector<float>(5, 0.0f), copper, true, 3.0f, towardSun,
         0.0f, specIor, specFactor, specColor},
        // F15.3 (ticket 10) sheen grids.  Sun only: the direct gate (Charlie lobe and albedo scaling, point-evaluated disk).
        {"sheen-grid-sun", sheenBaseRoughness, sheenMetallic, sheenBase, true, 3.0f, towardSun, 0.0f, {}, {}, {}, {}, {}, {}, {},
         false, sheenColors, sheenRoughnesses},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the declared sheen term.
        {"sheen-grid-env-resolve", sheenBaseRoughness, sheenMetallic, sheenBase, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {},
         {}, {}, {}, false, sheenColors, sheenRoughnesses},
        // The same scene with the specular transport on: sheen is not sampled, the environment term comes from the composition.
        {"sheen-grid-env", sheenBaseRoughness, sheenMetallic, sheenBase, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {}, {}, {},
         {}, false, sheenColors, sheenRoughnesses},
        // One point light only (no sun, environment 0), side-lit from behind like the other grids.
        {"sheen-grid-point", sheenBaseRoughness, sheenMetallic, sheenBase, false, 0.0f, towardSun, 0.0f, {}, {}, {},
         {sheenBackLight}, {}, {}, {}, false, sheenColors, sheenRoughnesses},
        // Furnace over sheen materials: base albedo 1 under a uniform environment of radiance 1 (the runner also captures it
        // without the transport).
        {"furnace-sheen", sheenBaseRoughness, sheenMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, {}, {}, {}, {}, {},
         {}, {}, false, sheenColors, sheenRoughnesses},
        // F15.3 (ticket 11) anisotropy grids.  Sun only: the direct gate (the anisotropic lobe at the disk centre).
        {"aniso-grid-sun", anisoRoughness, anisoMetallic, copper, true, 3.0f, towardSun, 0.0f, {}, {}, {}, {}, {}, {}, {}, false, {},
         {}, anisoStrength, anisoRotation},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the declared anisotropy term.
        {"aniso-grid-env-resolve", anisoRoughness, anisoMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {}, {}, {}, {},
         false, {}, {}, anisoStrength, anisoRotation},
        // The same scene with the specular transport on: the anisotropic lobe in the transport (VNDF sampling) and the environment.
        {"aniso-grid-env", anisoRoughness, anisoMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {}, {}, {}, {}, false, {},
         {}, anisoStrength, anisoRotation},
        // One point light only (no sun, environment 0), side-lit from behind like the other grids.
        {"aniso-grid-point", anisoRoughness, anisoMetallic, copper, false, 0.0f, towardSun, 0.0f, {}, {}, {}, {anisoBackLight}, {},
         {}, {}, false, {}, {}, anisoStrength, anisoRotation},
        // Furnace over anisotropic materials: base albedo 1 under a uniform environment of radiance 1 (the runner also
        // captures it without the transport).
        {"furnace-aniso", anisoRoughness, anisoMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, {}, {}, {}, {}, {}, {}, {},
         false, {}, {}, anisoStrength, anisoRotation},
        // Mirrored carriers under the sun: rotation direction, handedness w and mirrored instances (see above).
        {"aniso-mirrored-sun", std::vector<float>(4, 0.3f), mirrorMetallic, copper, true, 3.0f, towardSun, 0.0f, {}, {}, {}, {}, {},
         {}, {}, false, {}, {}, mirrorStrength, mirrorRotation, mirrorColumns, mirrorHandedness},
        // F15.3 (ticket 12) iridescence grids.  Sun only: the direct gate (the film at V.H of the disk centre).
        {"iridescence-grid-sun", iridRoughness, iridMetallic, copper, true, 3.0f, towardSun, 0.0f, {}, {}, {}, {}, {}, {}, {}, false,
         {}, {}, {}, {}, {}, {}, iridFactor, iridIor, iridThickness},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the declared film at N.V.
        {"iridescence-grid-env-resolve", iridRoughness, iridMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {}, {}, {}, {},
         false, {}, {}, {}, {}, {}, {}, iridFactor, iridIor, iridThickness},
        // The same scene with the specular transport on: the weight uses the film at the sampled V.H.
        {"iridescence-grid-env", iridRoughness, iridMetallic, copper, false, 0.0f, towardSun, 0.5f, {}, {}, {}, {}, {}, {}, {}, false,
         {}, {}, {}, {}, {}, {}, iridFactor, iridIor, iridThickness},
        // One point light only (no sun, environment 0), side-lit from behind like the other grids.
        {"iridescence-grid-point", iridRoughness, iridMetallic, copper, false, 0.0f, towardSun, 0.0f, {}, {}, {}, {iridBackLight}, {},
         {}, {}, false, {}, {}, {}, {}, {}, {}, iridFactor, iridIor, iridThickness},
        // Furnace over iridescent materials: base albedo 1 under a uniform environment of radiance 1 (also without the transport).
        {"furnace-iridescence", iridRoughness, iridMetallic, glm::vec3(1.0f), false, 0.0f, towardSun, 1.0f, {}, {}, {}, {}, {}, {}, {},
         false, {}, {}, {}, {}, {}, {}, iridFactor, iridIor, iridThickness},
        // F15.3 (ticket 16) thin subsurface.  Sun behind the spheres only: the direct gate (reflected (1 - w) and transmitted lobe
        // at the disk centre).
        {.name = "subsurface-thin-sun", .roughness = thinRoughness, .metallic = thinMetallic4, .baseColor = thinBase, .sun = true,
         .sunIrradiance = 3.0f, .towardSun = thinSun, .environment = 0.0f, .subsurfaceWeight = thinWeights4,
         .subsurfaceColor = thinColors},
        // Environment only, deterministic resolve (--reflections off --lighting-denoise custom): the declared transmitted weight.
        {.name = "subsurface-thin-env-resolve", .roughness = thinRoughness, .metallic = thinMetallic4, .baseColor = thinBase,
         .towardSun = thinSun, .environment = 0.5f, .subsurfaceWeight = thinWeights4, .subsurfaceColor = thinColors},
        // The same scene with the specular transport on: the back-side environment of the composition (exact quadrature gate).
        {.name = "subsurface-thin-env", .roughness = thinRoughness, .metallic = thinMetallic4, .baseColor = thinBase,
         .towardSun = thinSun, .environment = 0.5f, .subsurfaceWeight = thinWeights4, .subsurfaceColor = thinColors},
        // One point light BEHIND single-sided quads only (no sun, environment 0): the resolve's local lights with --lighting legacy,
        // ReSTIR or clustered DI otherwise.  DI receivers skip lights with N.L <= 0 today (diBRDFWeighted).
        {.name = "subsurface-thin-point", .roughness = thinRoughness, .metallic = thinMetallic3, .baseColor = thinBase,
         .towardSun = thinSun, .environment = 0.0f, .pointLights = {thinBackLight}, .subsurfaceWeight = thinWeights3,
         .subsurfaceColor = thinColors, .quadCarriers = true},
        // Furnace over thin materials: white base, white transmission colour, albedo 1 under a uniform environment of radiance 1
        // (reflected + transmitted <= 1: energy moved, not added); columns vary the base roughness (the runner also captures it
        // without the transport).
        {.name = "furnace-thin-subsurface", .roughness = thinFurnaceRoughness, .metallic = thinMetallic4,
         .baseColor = glm::vec3(1.0f), .towardSun = thinSun, .environment = 1.0f, .subsurfaceWeight = thinWeights4,
         .subsurfaceColor = thinWhite},
        // F15.1 (ticket 19): GI at -N of the thin lobe (tools/f15_thin_gi_back_check.py, DDGI + DI): weight-1 quads of three transmission
        // colours in front of a sunlit diffuse wall; the sun comes from the camera side at 60 degrees from the wall normal's complement.
        {.name = "subsurface-thin-gi-back", .roughness = thinRoughness, .metallic = {0.0f}, .baseColor = glm::vec3(0.0f),
         .sun = true, .sunIrradiance = 3.0f, .towardSun = glm::normalize(glm::vec3(0.8f, 0.3f, 0.52f)), .environment = 0.0f,
         .subsurfaceWeight = {1.0f}, .subsurfaceColor = thinColors, .quadCarriers = true, .giBackWall = true, .giWallAlbedo = 0.8f},
        // F15.3 (ticket 17) volumetric subsurface.  Sun only (environment 0): the gate against the Mitsuba volumetric random walk.
        {.name = "subsurface-volumetric-sun", .roughness = volRoughness, .metallic = volMetallic2, .baseColor = volBase, .sun = true,
         .sunIrradiance = 3.0f, .towardSun = volSun, .environment = 0.0f, .volumetricWeight = volWeights, .volumetricRadius = volRadii,
         .volumetricColor = volColors, .volumetricRadiusScale = volScales, .volumetricAnisotropy = volAnisotropy},
        // The radius sweep (weight 1, radius x0.25 / x0.5 / x1 / x2 / x3 of the nominal 0.08 m; columns 2-4 equal the scene above).
        {.name = "subsurface-volumetric-radius", .roughness = volRoughness, .metallic = volMetallic2, .baseColor = volBase, .sun = true,
         .sunIrradiance = 3.0f, .towardSun = volSun, .environment = 0.0f, .volumetricWeight = volSweepWeights,
         .volumetricRadius = volSweepRadii, .volumetricColor = volColors, .volumetricRadiusScale = volScales,
         .volumetricAnisotropy = volAnisotropy},
        // Sub-pixel radius: must equal the scene without any block (the next one).
        {.name = "subsurface-volumetric-subpixel", .roughness = subRoughness, .metallic = subMetallic1, .baseColor = volBase,
         .sun = true, .sunIrradiance = 3.0f, .towardSun = volSun, .environment = 0.0f, .volumetricWeight = subWeights,
         .volumetricRadius = subRadii, .volumetricColor = subColors, .volumetricRadiusScale = subScales,
         .volumetricAnisotropy = subAnisotropy},
        {.name = "subsurface-volumetric-subpixel-off", .roughness = subRoughness, .metallic = subMetallic1, .baseColor = volBase,
         .sun = true, .sunIrradiance = 3.0f, .towardSun = volSun, .environment = 0.0f},
        // Back-lit thick spheres: measured and reported, not gated.
        {.name = "subsurface-volumetric-backlit", .roughness = backRoughness, .metallic = volMetallic2, .baseColor = volBase,
         .sun = true, .sunIrradiance = 3.0f, .towardSun = backSun, .environment = 0.0f, .volumetricWeight = backWeights,
         .volumetricRadius = backRadii, .volumetricColor = volColors, .volumetricRadiusScale = volScales,
         .volumetricAnisotropy = volAnisotropy, .cameraDistance = 4.0f},
    };
    static const std::vector<MaterialScenarioSpec> all = [&] {
        std::vector<MaterialScenarioSpec> v = baseSpecs;
        addCombinations(v);
        return v;
    }();
    return all;
}
}

MaterialValidation::MaterialValidation(std::string scenario, u32 toggleEvery)
    : scenario_(std::move(scenario)), toggleEvery_(toggleEvery) {
    if (!validScenario(scenario_)) throw std::invalid_argument("Unknown material validation scenario: " + scenario_);
    if (toggleEvery_ > 0 && !spec(scenario_).hasSpecularGrid())
        throw std::invalid_argument("material toggling needs a specular grid scenario");
    name_ = "Material validation: " + scenario_;
}
bool MaterialValidation::validScenario(std::string_view scenario) {
    return std::find(kScenarios.begin(), kScenarios.end(), scenario) != kScenarios.end();
}
std::span<const std::string_view> MaterialValidation::scenarios() { return kScenarios; }
const MaterialScenarioSpec& MaterialValidation::spec(std::string_view scenario) {
    for (const auto& s : specs())
        if (s.name == scenario) return s;
    throw std::invalid_argument("Unknown material validation scenario: " + std::string(scenario));
}
float MaterialValidation::environmentRadiance(std::string_view scenario) {
    // F15.3 (ticket 13): the transmission scenarios share the --material-scenario option; their environment is declared there.
    if (TransmissionValidation::validScenario(scenario)) return TransmissionValidation::environmentRadiance(scenario);
    if (ChessFixture::validScenario(scenario)) return ChessFixture::environmentRadiance(scenario);
    if (TerrainFixture::validScenario(scenario)) return TerrainFixture::environmentRadiance(scenario);
    return validScenario(scenario) ? spec(scenario).environment : 0.0f;
}
glm::vec3 MaterialValidation::sphereCenter(const MaterialScenarioSpec& s, u32 column, u32 row) {
    const float x0 = -0.5f * Spacing * float(s.roughness.size() - 1);
    return {x0 + Spacing * float(column), Spacing * float(row), 0.0f};
}

void MaterialValidation::setup(ECS& ecs, GpuScene& scene, TextureManager& textures) {
    if (!spheres_.empty()) throw std::logic_error("Material fixture already set up");
    textures.createDefaultTextures(); // idempotent; materials reference none of them
    const auto& s = spec(scenario_);
    // The anisotropy grids use combed spheres (one mesh per tangent handedness: -1 the engine's UV sphere convention).
    auto upload = [&](float handedness) {
        // F15.3 (ticket 16): the thin point-light fixture uses single-sided quads (generatePlane lies in XZ with normal +Y; the
        // instance is rotated by +90 degrees about X so the normal faces the camera).
        const auto sphere = s.quadCarriers ? ProceduralMeshes::generatePlane(2.0f * SphereRadius, 2.0f * SphereRadius, 8, 8)
            : (s.hasAnisotropyGrid() || s.combedCarriers)
            ? ProceduralMeshes::generateCombedSphere(SphereRadius, SphereSlices, SphereStacks, glm::vec3(1.0f, 0.0f, 0.0f), handedness)
            : ProceduralMeshes::generateSphere(SphereRadius, SphereSlices, SphereStacks);
        return scene.uploadMesh(sphere.positions, sphere.normals, sphere.tangents, sphere.uvs, sphere.indices);
    };
    const u32 handle = upload(-1.0f);
    u32 handlePlus = handle;
    if (std::find(s.tangentHandedness.begin(), s.tangentHandedness.end(), 1.0f) != s.tangentHandedness.end()) handlePlus = upload(1.0f);
    u32 materialIndex = 0;
    for (u32 row = 0; row < s.metallic.size(); ++row)
        for (u32 column = 0; column < s.roughness.size(); ++column) {
            const EntityID e = ecs.createEntity();
            spheres_.push_back(e);
            TransformComponent transform;
            transform.position = sphereCenter(s, column, row);
            if (s.quadCarriers) transform.rotation = glm::angleAxis(0.5f * pi, glm::vec3(1.0f, 0.0f, 0.0f));
            if (!s.mirroredColumn.empty() && s.mirroredColumn[column]) transform.scale = glm::vec3(-1.0f, 1.0f, 1.0f);
            transform.updateMatrix();
            ecs.addComponent(e, std::move(transform));
            MeshInstanceComponent instance;
            instance.meshHandle = (!s.tangentHandedness.empty() && s.tangentHandedness[column] > 0.0f) ? handlePlus : handle;
            instance.materialIndex = materialIndex++;
            instance.setVisible(true);
            instance.setCastsShadows(true);
            instance.setStatic(true);
            ecs.addComponent(e, std::move(instance));
            MaterialComponent material;
            if (s.hasSpecularGrid()) {
                auto description = std::make_shared<MaterialDescription>();
                description->specular.emplace();
                description->specular->ior            = s.ior[column];
                description->specular->specularFactor = s.specularFactor[row];
                const glm::vec3 c = s.specularColor[row];
                description->specular->specularColor = {c.x, c.y, c.z};
                material.extension = std::move(description);
            }
            if (s.hasClearcoatGrid()) {
                const bool hasIor = !s.baseIor.empty() && s.baseIor[row] > 0.0f;
                if (s.clearcoatFactor[row] > 0.0f || hasIor) {
                    auto description = std::make_shared<MaterialDescription>();
                    description->clearcoat.emplace();
                    description->clearcoat->factor    = s.clearcoatFactor[row];
                    description->clearcoat->roughness = s.clearcoatRoughness[column];
                    if (hasIor) {
                        description->specular.emplace();
                        description->specular->ior = s.baseIor[row];
                    }
                    material.extension = std::move(description);
                }
            }
            if (s.hasSheenGrid() && s.sheenColor[row] != glm::vec3(0.0f)) {
                auto description = std::make_shared<MaterialDescription>();
                description->sheen.emplace();
                const glm::vec3 c = s.sheenColor[row];
                description->sheen->color     = {c.x, c.y, c.z};
                description->sheen->roughness = s.sheenRoughness[column];
                material.extension = std::move(description);
            }
            if (s.hasAnisotropyGrid() && s.anisotropyStrength[row] > 0.0f) {
                auto description = std::make_shared<MaterialDescription>();
                description->anisotropy.emplace();
                description->anisotropy->strength = s.anisotropyStrength[row];
                description->anisotropy->rotation = s.anisotropyRotation[column];
                material.extension = std::move(description);
            }
            if (s.hasSubsurfaceGrid() && s.subsurfaceWeight[row] > 0.0f) {
                auto description = std::make_shared<MaterialDescription>();
                description->subsurface.emplace();
                description->subsurface->mode   = SubsurfaceMode::Thin;
                description->subsurface->weight = s.subsurfaceWeight[row];
                const glm::vec3 c = s.subsurfaceColor[column];
                description->subsurface->color = {c.x, c.y, c.z};
                material.extension = std::move(description);
            }
            if (s.hasVolumetricGrid() && s.volumetricWeight[column] > 0.0f) {
                auto description = std::make_shared<MaterialDescription>();
                description->subsurface.emplace();
                auto& block = *description->subsurface;
                block.mode   = SubsurfaceMode::Volumetric;
                block.weight = s.volumetricWeight[column];
                const glm::vec3 c = s.volumetricColor[row], scale = s.volumetricRadiusScale[row];
                block.color       = {c.x, c.y, c.z};
                block.radius      = s.volumetricRadius[column];
                block.radiusScale = {scale.x, scale.y, scale.z};
                block.anisotropy  = s.volumetricAnisotropy[row];
                description->volume.emplace(); // a solid: thickness = the sphere diameter, neutral attenuation
                description->volume->thickness = 2.0f * SphereRadius;
                material.extension = std::move(description);
            }
            if (s.hasComboGrid()) {
                // F15.3 (ticket 19): one sphere, any subset of the lobes (the cell of this row and column).
                const auto& cell = s.combo[row * s.roughness.size() + column];
                auto description = std::make_shared<MaterialDescription>();
                const glm::vec3 sc = cell.specularColor, sh = cell.sheenColor, tc = cell.thinColor;
                if (cell.hasSpecular) {
                    description->specular.emplace();
                    description->specular->ior            = cell.ior;
                    description->specular->specularFactor = cell.specularFactor;
                    description->specular->specularColor  = {sc.x, sc.y, sc.z};
                }
                if (cell.coatFactor > 0.0f) {
                    description->clearcoat.emplace();
                    description->clearcoat->factor    = cell.coatFactor;
                    description->clearcoat->roughness = cell.coatRoughness;
                }
                if (sh != glm::vec3(0.0f)) {
                    description->sheen.emplace();
                    description->sheen->color     = {sh.x, sh.y, sh.z};
                    description->sheen->roughness = cell.sheenRoughness;
                }
                if (cell.anisoStrength > 0.0f) {
                    description->anisotropy.emplace();
                    description->anisotropy->strength = cell.anisoStrength;
                    description->anisotropy->rotation = cell.anisoRotation;
                }
                if (cell.iridFactor > 0.0f) {
                    description->iridescence.emplace();
                    description->iridescence->factor       = cell.iridFactor;
                    description->iridescence->ior          = cell.iridIor;
                    description->iridescence->thicknessMax = cell.iridThickness;
                }
                if (cell.thinWeight > 0.0f) {
                    description->subsurface.emplace();
                    description->subsurface->mode   = SubsurfaceMode::Thin;
                    description->subsurface->weight = cell.thinWeight;
                    description->subsurface->color  = {tc.x, tc.y, tc.z};
                }
                if (featureMask(*description) != 0) material.extension = std::move(description);
            }
            if (s.hasIridescenceGrid()) {
                auto description = std::make_shared<MaterialDescription>();
                description->iridescence.emplace();
                description->iridescence->factor       = s.iridescenceFactor[row];
                description->iridescence->ior          = s.iridescenceIor[row];
                description->iridescence->thicknessMax = s.iridescenceThickness[column]; // thicknessMin keeps its default (100 nm)
                material.extension = std::move(description);
            }
            material.baseColorFactor = glm::vec4(s.baseColor, 1.0f);
            material.metallicFactor = s.metallic[row];
            material.roughnessFactor = s.roughness[column];
            // INVALID texture indices intentionally use the exact shader
            // fallbacks: the default normal/MR textures would perturb factors.
            ecs.addComponent(e, std::move(material));
        }
    if (s.giBackWall) {
        // The wall faces the camera (generatePlane normal +Y rotated +90 degrees about X); the anchor is an unlit-black speck out of view.
        auto addPlane = [&](float side, glm::vec3 position, glm::vec3 albedo) {
            const auto plane = ProceduralMeshes::generatePlane(side, side, 8, 8);
            const u32 mesh = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
            const EntityID e = ecs.createEntity();
            extras_.push_back(e);
            TransformComponent transform;
            transform.position = position;
            transform.rotation = glm::angleAxis(0.5f * pi, glm::vec3(1.0f, 0.0f, 0.0f));
            transform.updateMatrix();
            ecs.addComponent(e, std::move(transform));
            MeshInstanceComponent instance;
            instance.meshHandle = mesh;
            instance.materialIndex = materialIndex++;
            instance.setVisible(true);
            instance.setCastsShadows(true);
            instance.setStatic(true);
            ecs.addComponent(e, std::move(instance));
            MaterialComponent material;
            material.baseColorFactor = glm::vec4(albedo, 1.0f);
            material.metallicFactor  = 0.0f;
            material.roughnessFactor = 1.0f;
            ecs.addComponent(e, std::move(material));
        };
        addPlane(8.0f, glm::vec3(0.0f, 0.0f, -0.6f), glm::vec3(s.giWallAlbedo));
        addPlane(0.02f, glm::vec3(-6.5f, -4.0f, 1.2f), glm::vec3(0.0f));
    }
    if (s.sun) {
        sun_ = ecs.createEntity();
        TransformComponent transform;
        transform.rotation = orientLight(s.towardSun);
        transform.updateMatrix();
        ecs.addComponent(sun_, std::move(transform));
        LightComponent light;
        light.type = LightType::Directional;
        light.color = glm::vec3(1.0f);
        light.intensity = s.sunIrradiance;
        ecs.addComponent(sun_, std::move(light));
    }
    for (const MaterialPointLight& point : s.pointLights) {
        const EntityID e = ecs.createEntity();
        pointLights_.push_back(e);
        TransformComponent transform;
        transform.position = point.position;
        transform.updateMatrix();
        ecs.addComponent(e, std::move(transform));
        LightComponent light;
        light.type = LightType::Point;
        light.color = glm::vec3(1.0f);
        light.intensity = point.intensity;
        light.range = point.range;
        ecs.addComponent(e, std::move(light));
    }
}

void MaterialValidation::update(float, ECS& ecs) {
    if (toggleEvery_ == 0 || spheres_.empty()) return;
    if (++updates_ % toggleEvery_ != 0) return;
    const u32 event = updates_ / toggleEvery_; // 1, 2, 3 ...
    auto edited = [&](u32 index, auto&& change) {
        auto& m = ecs.getComponent<MaterialComponent>(spheres_[index]); // mutable access marks the change
        auto d = std::make_shared<MaterialDescription>(*m.extension);
        change(*d->specular);
        m.extension = std::move(d);
    };
    // Odd events: the feature sphere becomes active (ior 1.33), even events neutral again.
    const bool strong = spec(scenario_).strongEdit, odd = event % 2 == 1;
    edited(toggleFeatureSphere(), [&](SpecularBlock& b) { b.ior = strong ? (odd ? 2.4f : 1.5f) : (odd ? 1.33f : 1.5f); });
    // The parameter sphere alternates its weight 0.5 / 0.6 (a record rewrite without a bit change).
    edited(toggleParameterSphere(), [&](SpecularBlock& b) {
        if (strong) b.specularColor = odd ? std::array<float, 3>{25.0f, 25.0f, 25.0f} : std::array<float, 3>{1.0f, 1.0f, 1.0f};
        else b.specularFactor = odd ? 0.6f : 0.5f;
    });
}

void MaterialValidation::teardown(ECS& ecs, GpuScene&) {
    for (EntityID e : spheres_) ecs.destroyEntity(e);
    if (sun_ != INVALID_ENTITY) ecs.destroyEntity(sun_);
    for (EntityID e : pointLights_) ecs.destroyEntity(e);
    for (EntityID e : extras_) ecs.destroyEntity(e);
    pointLights_.clear();
    extras_.clear();
    spheres_.clear();
    sun_ = INVALID_ENTITY;
}

CameraSetup MaterialValidation::getDefaultCamera() const {
    // 16:9 frame, 60 degree vertical field of view (engine default): the
    // distance keeps the whole grid plus a margin in view.
    const auto& s = spec(scenario_);
    const float centreY = 0.5f * Spacing * float(s.metallic.size() - 1);
    const float distance = s.cameraDistance > 0.0f ? s.cameraDistance : (s.roughness.size() > 2 ? 5.5f : 4.0f);
    return {{0.0f, centreY, distance}, {0.0f, centreY, 0.0f}, distance, false};
}
} // namespace phosphor
