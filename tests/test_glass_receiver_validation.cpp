#include <doctest/doctest.h>
#include "testbench/glass_receiver_validation.h"
#include "testbench/material_validation.h"
#include "testbench/transmission_validation.h"
#include "testbench/transmissive_shadow_validation.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "null_texture_manager.h"
#include "renderer/shadow_settings.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
using namespace phosphor;

TEST_CASE("F15 closure (tickets 13/15): the glass-receiver scenarios are the four declared names, in their own bench class") {
    const auto names = GlassReceiverValidation::scenarios();
    REQUIRE(names.size() == 4);
    for (auto name : {"glass-receiver-stack", "glass-receiver-single", "glass-btdf-sun", "glass-btdf-point"}) {
        CHECK(GlassReceiverValidation::validScenario(name));
        CHECK_FALSE(MaterialValidation::validScenario(name));
        CHECK_FALSE(TransmissionValidation::validScenario(name));
        CHECK_FALSE(TransmissiveShadowValidation::validScenario(name));
        CHECK(MaterialValidation::environmentRadiance(name) == 0.0f); // the environment is 0: only the delta light lights the panes
    }
    CHECK_FALSE(GlassReceiverValidation::validScenario("transmissive-shadow-pane-sun"));
    CHECK_THROWS(GlassReceiverValidation{"unknown"});
}

TEST_CASE("F15 closure (ticket 15): the stack fixture has the B pane directly visible and wholly inside the shadow of A; single is the same scene without A") {
    const auto& stack = GlassReceiverValidation::spec("glass-receiver-stack");
    const auto& single = GlassReceiverValidation::spec("glass-receiver-single");
    REQUIRE(stack.panes.size() == 2);
    REQUIRE(single.panes.size() == 1);
    const auto& a = stack.panes[0];
    const auto& b = stack.panes[1];
    // B is identical in both scenes (the baseline of the transmittance ratio).
    CHECK(single.panes[0].center == b.center);
    CHECK(single.panes[0].size == b.size);
    CHECK(single.panes[0].roughness == b.roughness);
    CHECK(single.panes[0].baseColor == b.baseColor);
    CHECK(stack.cameraDistance == single.cameraDistance);
    CHECK(stack.towardSun == single.towardSun);
    CHECK(a.horizontal);
    CHECK(b.horizontal);
    CHECK(a.roughness == 0.0f);
    CHECK(a.baseColor == glm::vec3(0.6f, 0.8f, 0.9f)); // tinted-clear: every channel loses more than the 5 % gate (the no-attenuation control fails)
    CHECK(b.baseColor != glm::vec3(1.0f)); // a tinted B: its own tint would show if it were counted twice
    // Geometry of the claim: sun ray from a B point and view ray from the same point, intersected with A's plane.
    const float height = a.center.y - b.center.y;
    REQUIRE(height > 0.0f);
    const float sunShift = height / stack.towardSun.y * stack.towardSun.z; // z displacement of the sun ray at A's plane
    const float el = glm::radians(stack.cameraElevationDegrees);
    const float viewShift = height / std::sin(el) * std::cos(el);          // z displacement of the camera ray at A's plane (towards +Z)
    const float aMinZ = a.center.z - 0.5f * a.size.y, aMaxZ = a.center.z + 0.5f * a.size.y;
    const float aMinX = a.center.x - 0.5f * a.size.x, aMaxX = a.center.x + 0.5f * a.size.x;
    for (float u : {-0.5f, 0.0f, 0.5f})
        for (float v : {-0.5f, 0.0f, 0.5f}) {
            const float x = b.center.x + u * b.size.x, z = b.center.z + v * b.size.y;
            CHECK(z + sunShift > aMinZ + 0.1f);   // the sun ray crosses A, with a margin of 0.1 m
            CHECK(z + sunShift < aMaxZ - 0.1f);
            CHECK(x > aMinX + 0.1f);
            CHECK(x < aMaxX - 0.1f);
            CHECK(z + viewShift > aMaxZ + 0.1f);  // the view ray misses A: B is seen directly
        }
    // The sun is the mirror of the camera about the vertical: the sun reflection of the horizontal panes falls on the camera.
    CHECK(stack.towardSun.y == doctest::Approx(std::sin(el)).epsilon(1e-5));
    CHECK(stack.towardSun.z == doctest::Approx(-std::cos(el)).epsilon(1e-5));
}

TEST_CASE("F15 closure (ticket 13): the BTDF fixtures light vertical thin-walled panes from behind with one delta light") {
    for (auto name : {"glass-btdf-sun", "glass-btdf-point"}) {
        const auto& s = GlassReceiverValidation::spec(name);
        REQUIRE(s.panes.size() == 7);
        CHECK((s.sun ? 1 : 0) + (s.point ? 1 : 0) == 1);
        for (const auto& p : s.panes) {
            CHECK_FALSE(p.horizontal);
            CHECK(p.center.z == 0.0f);
            CHECK(p.transmission > 0.0f);
        }
        if (s.sun) CHECK(s.towardSun.z < -0.9f); // behind the panes (they face the camera at +Z)
        if (s.point) {
            CHECK(s.pointPosition.z < -1.0f);
            CHECK(s.pointIntensity == 60.0f);
        }
        // Roughness {0.1, 0.3, 0.6} per column, white row 0, tinted row 1, one half-transmission pane.
        CHECK(s.panes[0].roughness == 0.1f);
        CHECK(s.panes[1].roughness == 0.3f);
        CHECK(s.panes[2].roughness == 0.6f);
        CHECK(s.panes[0].baseColor == glm::vec3(1.0f));
        CHECK(s.panes[3].baseColor != glm::vec3(1.0f));
        CHECK(s.panes[6].transmission == 0.5f);
    }
}

TEST_CASE("F15 closure (tickets 13/15): panes are thin-walled transmissive materials with exact factors, the light is the only emitter") {
    for (auto name : GlassReceiverValidation::scenarios()) {
        ECS ecs;
        GpuScene scene;
        test::NullTextureManager textures;
        GlassReceiverValidation f{std::string(name)};
        f.setup(ecs, scene, textures);
        const auto& s = GlassReceiverValidation::spec(name);
        REQUIRE(f.panes().size() == s.panes.size());
        CHECK((f.light() != INVALID_ENTITY) == (s.sun || s.point));
        REQUIRE(f.anchor() != INVALID_ENTITY);
        const auto& anchor = std::as_const(ecs).getComponent<MaterialComponent>(f.anchor());
        CHECK_FALSE(anchor.extension);                       // an opaque comparison object: nothing but black
        CHECK(anchor.baseColorFactor == glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        CHECK(anchor.emissiveFactor == glm::vec3(0.0f));
        for (size_t k = 0; k < s.panes.size(); ++k) {
            const auto& m = std::as_const(ecs).getComponent<MaterialComponent>(f.panes()[k]);
            REQUIRE(m.extension);
            REQUIRE(m.extension->transmission);
            REQUIRE(m.extension->volume);
            CHECK(m.extension->transmission->factor == s.panes[k].transmission);
            CHECK(m.extension->volume->thickness == 0.0f);
            CHECK(m.baseColorFactor == glm::vec4(s.panes[k].baseColor, 1.0f));
            CHECK(m.roughnessFactor == s.panes[k].roughness);
            CHECK(m.metallicFactor == 0.0f);
            CHECK(m.emissiveFactor == glm::vec3(0.0f));
            CHECK(m.doubleSided);
        }
        f.teardown(ecs, scene);
        CHECK(f.panes().empty());
        CHECK(f.light() == INVALID_ENTITY);
        CHECK(f.anchor() == INVALID_ENTITY);
    }
}

TEST_CASE("F15 closure (ticket 15): the covering cascade's texel of the edge-quality gate (tools/f15_glass_receiver_check.py cascade_texel)") {
    // The engine camera (near 0.05, fov 60 degrees, 16:9), default settings (distance 120, lambda 0.6) at --shadow-map-size 2048: the floor of
    // the pane-sun fixture (view depth 4.5 .. 8 m) lies in cascade 0, texel 15.33 mm (30.67 mm in the divisor-2 transmittance map).
    ShadowSettings settings;
    settings.mapResolution = 2048;
    ShadowCamera camera;
    camera.position = glm::vec3(0.0f);
    camera.forward = glm::vec3(0.0f, 0.0f, -1.0f);
    camera.nearPlane = 0.05f;
    camera.inverseViewProjection = glm::inverse(glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.05f, 120.0f));
    const auto cascades = makeShadowCascades(camera, glm::vec3(0.0f, 1.0f, 0.0f), settings, {});
    CHECK(cascades[0].splitNear == doctest::Approx(0.05f));
    CHECK(cascades[0].splitFar == doctest::Approx(12.225f).epsilon(1e-3));
    CHECK(cascades[0].splitFar > 8.0f); // the whole gated floor is in cascade 0
    CHECK(cascades[0].radius == doctest::Approx(15.7028f).epsilon(2e-4));
    CHECK(cascades[0].texelWorld == doctest::Approx(0.0153348f).epsilon(2e-4));
}
