#include <doctest/doctest.h>

#include "pipeline/forward_variants.h"
#include "pipeline/pipeline_key.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_extension_pack.h"
#include "renderer/scene_store.h"
#include "renderer/temporal_layout.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"
#include "null_texture_manager.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>

using namespace phosphor;

// F15.1 (ticket 22): per-slot KHR_texture_transform, CPU side (analysis, packing, store allocation, pipeline constant).

namespace {

UvTransform makeTransform(float ox, float oy, float rotation, float sx, float sy) {
    UvTransform t;
    t.offset = {ox, oy};
    t.rotation = rotation;
    t.scale = {sx, sy};
    return t;
}

MaterialDescription perSlotDescription() {
    MaterialDescription d;
    d.base.baseColorTexture.index = 3;
    d.base.baseColorTexture.transform = makeTransform(0.2f, 0.1f, 0.3f, 2.0f, 1.5f);
    d.base.emissiveTexture.index = 4;
    d.base.emissiveTexture.transform = makeTransform(0.5f, 0.25f, 1.5707963f, 1.25f, 1.25f);
    d.base.occlusionTexture.index = 5; // present, no transform
    d.clearcoat = ClearcoatBlock{};
    d.clearcoat->factor = 1.0f;
    d.clearcoat->normalTexture.index = 6;
    d.clearcoat->normalTexture.transform = makeTransform(0.0f, 0.0f, -0.7f, 3.0f, 3.0f);
    return d;
}

GPUTextureTransformRecord record(const TextureTransformBlock& b, u32 i) {
    GPUTextureTransformRecord r;
    std::memcpy(&r, &b[i], sizeof(r));
    return r;
}

} // namespace

TEST_CASE("F15.1 ticket 22: the transform block packs one transform per slot at record slot / 3, mask bits only for transformed slots") {
    const MaterialDescription d = perSlotDescription();
    REQUIRE(analyzeTextureTransforms(d).status == TextureTransformStatus::PerSlot);
    const TextureTransformBlock b = packTextureTransforms(d);
    const GPUTextureTransformRecord r0 = record(b, 0);
    const u32 expectedMask = (1u << TT_SLOT_BASE_COLOR) | (1u << TT_SLOT_EMISSIVE) | (1u << TT_SLOT_COAT_NORMAL);
    CHECK(r0.slotMask == expectedMask);
    // Independent evaluation of the README matrices on three points for each slot.
    struct Case { u32 slot; UvTransform t; };
    const Case cases[] = {{TT_SLOT_BASE_COLOR, *d.base.baseColorTexture.transform},
                          {TT_SLOT_EMISSIVE, *d.base.emissiveTexture.transform},
                          {TT_SLOT_COAT_NORMAL, *d.clearcoat->normalTexture.transform}};
    for (const Case& c : cases) {
        const GPUTextureTransformRecord r = record(b, c.slot / 3);
        const GPUTextureTransform g = r.t[c.slot % 3];
        for (const std::array<float, 2> uv : {std::array<float, 2>{0.0f, 0.0f}, {0.25f, 0.5f}, {1.7f, -0.3f}}) {
            const auto want = applyUvTransform(c.t, uv[0], uv[1]);
            const float gx = g.m[0] * uv[0] + g.m[1] * uv[1] + g.o[0];
            const float gy = g.m[2] * uv[0] + g.m[3] * uv[1] + g.o[1];
            CHECK(gx == doctest::Approx(want[0]).epsilon(1e-5).scale(1.0));
            CHECK(gy == doctest::Approx(want[1]).epsilon(1e-5).scale(1.0));
        }
    }
    // Slots without a transform keep zero entries; the occlusion texture (present, plain) has no bit.
    CHECK((r0.slotMask & (1u << TT_SLOT_OCCLUSION)) == 0u);
    const GPUTextureTransform none = record(b, TT_SLOT_OCCLUSION / 3).t[TT_SLOT_OCCLUSION % 3];
    CHECK(none.m[0] == 0.0f);
    // A material that is not PerSlot packs nothing.
    MaterialDescription uniform;
    uniform.base.baseColorTexture.index = 1;
    uniform.base.baseColorTexture.transform = makeTransform(0.1f, 0.2f, 0.0f, 2.0f, 2.0f);
    CHECK(analyzeTextureTransforms(uniform).status == TextureTransformStatus::Uniform);
    CHECK_FALSE(needsTextureTransformBlock(uniform));
    const TextureTransformBlock empty = packTextureTransforms(uniform);
    for (const auto& r : empty)
        for (u32 w : r.w) CHECK(w == 0u);
}

TEST_CASE("F15.1 ticket 22: every slot name of forEachTexture maps to a distinct transform slot (layer textures to none)") {
    MaterialDescription d = perSlotDescription();
    d.specular = SpecularBlock{};
    d.sheen = SheenBlock{};
    d.anisotropy = AnisotropyBlock{};
    d.iridescence = IridescenceBlock{};
    d.transmission = TransmissionBlock{};
    d.volume = VolumeBlock{};
    d.subsurface = SubsurfaceBlock{};
    d.layers = LayerStack{};
    d.layers->layers.resize(MATERIAL_LAYER_MAX); // the mask and the four height textures have slots of the stack's own block (F15.4)
    u32 seen = 0, mapped = 0, unmapped = 0;
    forEachTexture(d, [&](const char* name, const MaterialTextureRef&) {
        const u32 s = textureTransformSlot(name);
        if (s == ~0u) {
            ++unmapped;
            return;
        }
        REQUIRE(s < TT_SLOT_COUNT);
        CHECK((seen & (1u << s)) == 0u);
        seen |= 1u << s;
        ++mapped;
    });
    CHECK(mapped == TT_SLOT_COUNT);
    CHECK(seen == (1u << TT_SLOT_COUNT) - 1u);
    CHECK(unmapped == 0);
}

namespace {

struct TtWorld {
    GpuScene scene;
    ECS ecs;
    SceneStore store;
    MeshHandle mesh{};

    TtWorld() {
        const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1);
        mesh = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
        GPUMaterial g{};
        scene.addMaterial(g);
    }

    EntityID add(std::shared_ptr<const MaterialDescription> desc) {
        const EntityID e = ecs.createEntity();
        TransformComponent xf{};
        xf.updateMatrix();
        ecs.addComponent(e, std::move(xf));
        MeshInstanceComponent mi{};
        mi.meshHandle = mesh;
        mi.materialIndex = 0;
        mi.setVisible(true);
        ecs.addComponent(e, std::move(mi));
        MaterialComponent mc{};
        mc.extension = std::move(desc);
        ecs.addComponent(e, std::move(mc));
        return e;
    }

    void frame() {
        store.sync(ecs, scene);
        const std::string v = store.verifyAgainstEcs(ecs, scene);
        REQUIRE_MESSAGE(v.empty(), v);
        ecs.endFrame();
    }
    const GPUMaterial& material(EntityID e) const { return store.materials()[store.materialIndexOf(e)]; }
};

} // namespace

TEST_CASE("F15.1 ticket 22: the store allocates a transform block, flags the material, frees it and carries the pipeline marker") {
    TtWorld w;
    // A scene without transforms: no block, no marker, no extension group.
    const EntityID plain = w.add(nullptr);
    w.frame();
    CHECK(w.store.extensionGroups() == 0);
    CHECK_FALSE(w.store.textureTransformInUse());
    CHECK(w.store.pipelineFeatureUnion() == 0u);
    CHECK((w.material(plain).flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) == 0u);
    CHECK(w.material(plain).textureTransformIndex == 0u);

    const auto desc = std::make_shared<const MaterialDescription>(perSlotDescription());
    const EntityID a = w.add(desc);
    w.frame();
    const GPUMaterial& ma = w.material(a);
    CHECK((ma.flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) != 0u);
    CHECK(ma.extensionIndex != MATERIAL_EXTENSION_INVALID); // the clearcoat feature is active (factor 1): an extension group exists too
    CHECK(w.store.textureTransformInUse());
    CHECK(w.store.pipelineFeatureUnion() == (w.store.featureUnion() | FEATURE_UNION_TEXTURE_TRANSFORM));
    CHECK((w.store.pipelineFeatureUnion() & MATERIAL_FEATURES_ALL) == w.store.featureUnion());
    const u32 block = ma.textureTransformIndex;
    CHECK(block % MATERIAL_EXT_RECORDS == 0u);
    GPUTextureTransformRecord r0;
    std::memcpy(&r0, &w.store.extensions()[block], sizeof(r0));
    CHECK(r0.slotMask == ((1u << TT_SLOT_BASE_COLOR) | (1u << TT_SLOT_EMISSIVE) | (1u << TT_SLOT_COAT_NORMAL)));

    // A second material gets its own, non-overlapping block.
    const EntityID b = w.add(desc);
    w.frame();
    const u32 block2 = w.material(b).textureTransformIndex;
    CHECK((block2 >= block + MATERIAL_TT_RECORDS || block >= block2 + MATERIAL_TT_RECORDS));

    // The transforms go away (a uniform description): flag clear, block kept and zeroed, still verified.
    MaterialDescription uniform;
    uniform.base.baseColorTexture.index = 3;
    uniform.base.baseColorTexture.transform = makeTransform(0.1f, 0.0f, 0.0f, 2.0f, 2.0f);
    {
        auto& mc = w.ecs.getComponent<MaterialComponent>(a);
        mc.extension = std::make_shared<const MaterialDescription>(uniform);
    }
    w.frame();
    CHECK((w.material(a).flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) == 0u);
    CHECK(w.material(a).textureTransformIndex == 0u);
    CHECK(w.store.textureTransformInUse()); // b still has one

    // Removing both entities returns the blocks (zeroed: verify checks it) and clears the marker.
    w.ecs.destroyEntity(b);
    w.ecs.destroyEntity(a);
    w.frame();
    CHECK_FALSE(w.store.textureTransformInUse());
    // A new material reuses a freed block instead of growing the buffer.
    const u32 groups = w.store.extensionGroups();
    const EntityID c = w.add(desc);
    w.frame();
    CHECK(w.store.extensionGroups() == groups);
    CHECK((w.material(c).flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) != 0u);
}

TEST_CASE("F15.1 ticket 22: withTextureTransform appends constant 26 once; a descriptor without it keeps its key") {
    using namespace pipe::forward;
    CHECK(textureTransformConstantIndex() == 26);
    const pipe::PipelineDesc base = genericDesc(rg::Format::BGRA8Srgb);
    const pipe::PipelineDesc tt = withTextureTransform(base);
    CHECK(tt.constantCount == base.constantCount + 1);
    CHECK(pipe::canonicalString(tt) != pipe::canonicalString(base));
    const pipe::PipelineDesc again = withTextureTransform(tt);
    CHECK(again.constantCount == tt.constantCount);
    CHECK(pipe::canonicalString(again) == pipe::canonicalString(tt));
    // Composes with a feature union without disturbing constant 22.
    const pipe::PipelineDesc both = withTextureTransform(withMaterialFeatures(base, MATERIAL_FEATURE_CLEARCOAT));
    CHECK(both.constantCount == base.constantCount + 2);
}

TEST_CASE("F15.4 ticket 22: a layer stack with a transform is PerSlot (never baked); a layer sub-material gets its block even when uniform") {
    MaterialDescription stack;
    stack.layers = LayerStack{};
    stack.layers->mask.index = 3;
    stack.layers->mask.transform = makeTransform(0.5f, 0.0f, 0.0f, 1.0f, 1.0f);
    stack.layers->layers.resize(2);
    CHECK(analyzeTextureTransforms(stack).status == TextureTransformStatus::PerSlot);
    const TextureTransformBlock b = packTextureTransforms(stack);
    CHECK(record(b, 0).slotMask == (1u << TT_SLOT_LAYER_MASK));
    MaterialDescription sub;
    sub.base.baseColorTexture.index = 4;
    sub.base.baseColorTexture.transform = makeTransform(0.1f, 0.0f, 0.0f, 2.0f, 2.0f);
    CHECK(analyzeTextureTransforms(sub).status == TextureTransformStatus::Uniform);
    CHECK_FALSE(needsTextureTransformBlock(sub, false));
    CHECK(needsTextureTransformBlock(sub, true)); // a layer target: no primitive bakes it
    const TextureTransformBlock forced = packTextureTransforms(sub, true);
    CHECK(record(forced, 0).slotMask == (1u << TT_SLOT_BASE_COLOR));
    // the temporal parameters carry the extension buffer pointer at an 8-aligned offset, size unchanged
    CHECK(offsetof(GPUTemporalParams, materialExtensions) == 184);
    CHECK(sizeof(GPUTemporalParams) == 192);
}

TEST_CASE("F15.4 ticket 22: a material used directly AND as a layer is never transformed twice") {
    for (const bool perSlot : {false, true}) {
        GpuScene scene;
        ECS ecs;
        SceneStore store;
        const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1);
        const MeshHandle mesh = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);
        MaterialDescription sub;
        sub.base.baseColorTexture.index = 4;
        sub.base.baseColorTexture.transform = makeTransform(0.1f, 0.0f, 0.0f, 2.0f, 2.0f);
        if (perSlot) {
            sub.base.emissiveTexture.index = 5;
            sub.base.emissiveTexture.transform = makeTransform(0.0f, 0.3f, 0.5f, 1.0f, 1.0f);
        }
        const u32 subIdx = scene.addMaterial(GPUMaterial{}, sub, "sub", MaterialSupportReport{});
        MaterialDescription stack;
        stack.layers = LayerStack{};
        stack.layers->layers.resize(2);
        stack.layers->layers[0].materialIndex = subIdx;
        stack.layers->layers[1].materialIndex = subIdx;
        const u32 stackIdx = scene.addMaterial(GPUMaterial{}, stack, "stack", MaterialSupportReport{});
        for (const u32 material : {subIdx, stackIdx}) {
            const EntityID e = ecs.createEntity();
            TransformComponent xf{};
            xf.updateMatrix();
            ecs.addComponent(e, std::move(xf));
            MeshInstanceComponent mi{};
            mi.meshHandle = mesh;
            mi.materialIndex = material;
            mi.setVisible(true);
            ecs.addComponent(e, std::move(mi));
        }
        store.sync(ecs, scene);
        const std::string v = store.verifyAgainstEcs(ecs, scene);
        REQUIRE_MESSAGE(v.empty(), v);
        const GPUMaterial& m = store.materials()[subIdx];
        // The layer use has its block; the DIRECT use applies it only when the material is per-slot (a uniform transform is baked into the
        // vertex uvs of its direct users: the block must not be applied to them again).
        CHECK((m.flags & MATERIAL_FLAG_LAYER_TRANSFORM) != 0u);
        CHECK(((m.flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) != 0u) == perSlot);
        CHECK(m.textureTransformIndex % MATERIAL_EXT_RECORDS == 0u);
        CHECK(store.textureTransformInUse());
        const GPUMaterial& s = store.materials()[stackIdx];
        CHECK((s.flags & (MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM)) == 0u);
    }
}
