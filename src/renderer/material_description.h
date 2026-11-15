#pragma once

// F15.1 (ticket 04): the material description.
//
// The versioned, renderer-independent record of one material: base
// metallic-roughness parameters, one optional block per material feature and a
// bounded layer stack (see GLOSSARY.md).  GPUMaterial and the extension
// records of later tickets are representations of it.  This header is
// portable (phosphor_core): no Metal, no glTF types.
//
// Parameter defaults, ranges, texture channels and colour spaces follow the
// glTF KHR_materials_* specifications (.scratch/f15/research/khr-materials.md);
// subsurface follows OpenPBR (ADR-0005).

#include "core/types.h"
#include "renderer/gpu_types.h"

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace phosphor {

constexpr u32 MATERIAL_DESCRIPTION_VERSION = 1;
constexpr u32 MATERIAL_LAYER_MAX = 4;

enum class TextureColorSpace : u8 { Linear, SRGB };

/// Channels of a texture a parameter is read from.  Gb = glTF
/// metallicRoughness (G roughness, B metallic); Rgb also covers
/// anisotropyTexture (rg direction, b strength) and normal maps.
enum class TextureChannel : u8 { R, G, B, A, Gb, Rgb, Rgba };

/// KHR_texture_transform: uv' = T * R * S * uv (offset, rotation in radians
/// counter-clockwise, scale).
struct UvTransform {
    std::array<float, 2> offset{0.0f, 0.0f};
    float                rotation = 0.0f;
    std::array<float, 2> scale{1.0f, 1.0f};

    [[nodiscard]] bool isIdentity() const {
        return offset[0] == 0.0f && offset[1] == 0.0f && rotation == 0.0f && scale[0] == 1.0f &&
               scale[1] == 1.0f;
    }
    bool operator==(const UvTransform&) const = default;
};

/// A texture slot of a parameter.  `index` is the resolved bindless index
/// (INVALID_TEXTURE_INDEX when absent).  `path` is only used by the material
/// description file (relative to that file) until the caller resolves it.
struct MaterialTextureRef {
    u32                          index      = INVALID_TEXTURE_INDEX;
    TextureColorSpace            colorSpace = TextureColorSpace::Linear;
    TextureChannel               channel    = TextureChannel::Rgba;
    u32                          texCoord   = 0;
    std::optional<UvTransform>   transform;
    std::string                  path;

    [[nodiscard]] bool present() const { return index != INVALID_TEXTURE_INDEX || !path.empty(); }
};

/// Slot with the channel and colour space the specification defines.
[[nodiscard]] inline MaterialTextureRef textureSlot(TextureColorSpace cs, TextureChannel ch) {
    MaterialTextureRef ref;
    ref.colorSpace = cs;
    ref.channel    = ch;
    return ref;
}

constexpr float DESCRIPTION_INFINITY = std::numeric_limits<float>::infinity();

/// glTF core metallic-roughness (Appendix B base model).  `emissive` has
/// KHR_materials_emissive_strength already folded in.
struct BaseMaterial {
    std::array<float, 4> baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    float                metallic          = 1.0f;
    float                roughness         = 1.0f;
    float                normalScale       = 1.0f;
    float                occlusionStrength = 1.0f;
    std::array<float, 3> emissive{0.0f, 0.0f, 0.0f};
    float                alphaCutoff = 0.0f; // 0 = no alpha test
    bool                 doubleSided = false;

    MaterialTextureRef baseColorTexture         = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgba);
    MaterialTextureRef metallicRoughnessTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::Gb);
    MaterialTextureRef normalTexture            = textureSlot(TextureColorSpace::Linear, TextureChannel::Rgb);
    MaterialTextureRef occlusionTexture         = textureSlot(TextureColorSpace::Linear, TextureChannel::R);
    MaterialTextureRef emissiveTexture          = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgb);
};

/// KHR_materials_ior + KHR_materials_specular.  ior = 0 is the permanent
/// special mode (effective IOR +inf, Fresnel evaluates to 1).
struct SpecularBlock {
    float                ior            = 1.5f;
    float                specularFactor = 1.0f;
    std::array<float, 3> specularColor{1.0f, 1.0f, 1.0f};
    MaterialTextureRef   specularTexture      = textureSlot(TextureColorSpace::Linear, TextureChannel::A);
    MaterialTextureRef   specularColorTexture = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgb);
};

/// KHR_materials_clearcoat (coat IOR fixed at 1.5 by the ratified text).
struct ClearcoatBlock {
    float              factor          = 0.0f;
    float              roughness       = 0.0f;
    float              normalScale     = 1.0f;
    MaterialTextureRef factorTexture   = textureSlot(TextureColorSpace::Linear, TextureChannel::R);
    MaterialTextureRef roughnessTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::G);
    MaterialTextureRef normalTexture   = textureSlot(TextureColorSpace::Linear, TextureChannel::Rgb);
};

/// KHR_materials_sheen.
struct SheenBlock {
    std::array<float, 3> color{0.0f, 0.0f, 0.0f};
    float                roughness = 0.0f;
    MaterialTextureRef   colorTexture     = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgb);
    MaterialTextureRef   roughnessTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::A);
};

/// KHR_materials_anisotropy; rotation in radians counter-clockwise from the
/// tangent.
struct AnisotropyBlock {
    float              strength = 0.0f;
    float              rotation = 0.0f;
    MaterialTextureRef texture  = textureSlot(TextureColorSpace::Linear, TextureChannel::Rgb);
};

/// KHR_materials_iridescence; thickness in nanometres.  Minimum may exceed
/// maximum (the specification allows it).
struct IridescenceBlock {
    float              factor       = 0.0f;
    float              ior          = 1.3f;
    float              thicknessMin = 100.0f;
    float              thicknessMax = 400.0f;
    MaterialTextureRef factorTexture    = textureSlot(TextureColorSpace::Linear, TextureChannel::R);
    MaterialTextureRef thicknessTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::G);
};

/// KHR_materials_transmission.
struct TransmissionBlock {
    float              factor  = 0.0f;
    MaterialTextureRef texture = textureSlot(TextureColorSpace::Linear, TextureChannel::R);
};

/// KHR_materials_volume; thickness in mesh-local units, attenuation distance
/// in world units (+inf: no attenuation).
struct VolumeBlock {
    float                thickness = 0.0f;
    std::array<float, 3> attenuationColor{1.0f, 1.0f, 1.0f};
    float                attenuationDistance = DESCRIPTION_INFINITY;
    MaterialTextureRef   thicknessTexture    = textureSlot(TextureColorSpace::Linear, TextureChannel::G);
};

enum class SubsurfaceMode : u8 { Thin, Volumetric };

/// Where a subsurface block came from; draft and RC inputs are reported as such.
enum class SubsurfaceSource : u8 {
    DescriptionFile,
    KhrMaterialsScatterDraft,      // PR #2579 at a6cd5b4 (draft)
    KhrMaterialsDiffuseTransmission // Release Candidate
};

/// Raw parameters of the draft KHR_materials_scatter, kept next to the
/// OpenPBR mapping.
struct ScatterDraftParams {
    float                scatterStrength = 0.0f;
    std::array<float, 3> multiscatterColor{1.0f, 1.0f, 1.0f};
    float                scatterAnisotropy = 0.0f;
};

/// Subsurface with OpenPBR semantics (ADR-0005).  The per-channel mean free
/// path in metres is `radius * radiusScale[c]`.
struct SubsurfaceBlock {
    SubsurfaceMode       mode   = SubsurfaceMode::Thin;
    float                weight = 0.0f;
    std::array<float, 3> color{0.8f, 0.8f, 0.8f}; // multiple-scattering colour
    float                radius = 1.0f;           // metres
    std::array<float, 3> radiusScale{1.0f, 0.5f, 0.25f};
    float                anisotropy = 0.0f;       // Henyey-Greenstein g
    MaterialTextureRef   weightTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::A);
    MaterialTextureRef   colorTexture  = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgb);
    // OpenPBR subsurface_radius_scale as a colour3 texture (multiplies radiusScale).
    MaterialTextureRef   radiusScaleTexture = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgb);
    SubsurfaceSource     source        = SubsurfaceSource::DescriptionFile;
    std::optional<ScatterDraftParams> draft; // set when source is the draft extension
};

/// Per-channel mean free path in metres.
[[nodiscard]] inline std::array<float, 3> meanFreePath(const SubsurfaceBlock& s) {
    return {s.radius * s.radiusScale[0], s.radius * s.radiusScale[1], s.radius * s.radiusScale[2]};
}

/// One layer of a layer stack: a material of the library blended by one
/// channel of the stack's RGBA mask.  Defined and validated here; not loaded
/// from glTF (F15.4 renders it).
///
/// F15.4 (ticket 23) semantics: a layer contributes the BASE parameters of its material (base colour factor x texture,
/// metallic, roughness, occlusion, normal map); the feature blocks of a layer material and its emissive/alpha are not
/// blended (features belong to the stack's own description: the coat finish).  `uvScale` tiles the layer's textures
/// (uv * uvScale; the mask keeps the primitive's uv).  `heightContrast` (metres, >= 0) switches the layer into the
/// height-contrast blend (renderer/layer_blend.h): its height is heightContrast * (heightTexture.r - 0.5).
struct MaterialLayer {
    std::string          materialName;
    u32                  materialIndex = ~0u; // resolved by the caller
    TextureChannel       maskChannel   = TextureChannel::R; // R, G, B or A
    std::optional<float> heightContrast; // metres, >= 0
    float                uvScale       = 1.0f; // > 0
    MaterialTextureRef   heightTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::R);
};

struct LayerStack {
    MaterialTextureRef      mask = textureSlot(TextureColorSpace::Linear, TextureChannel::Rgba);
    std::vector<MaterialLayer> layers;
};

struct MaterialDescription {
    u32          version = MATERIAL_DESCRIPTION_VERSION;
    BaseMaterial base;

    std::optional<SpecularBlock>     specular;
    std::optional<ClearcoatBlock>    clearcoat;
    std::optional<SheenBlock>        sheen;
    std::optional<AnisotropyBlock>   anisotropy;
    std::optional<IridescenceBlock>  iridescence;
    std::optional<TransmissionBlock> transmission;
    std::optional<VolumeBlock>       volume;
    std::optional<SubsurfaceBlock>   subsurface;
    std::optional<LayerStack>        layers;
};

/// Material features.  The numeric values are stable: bit n of featureMask()
/// is 1 << MaterialFeature value.
enum class MaterialFeature : u8 {
    SpecularIor   = 0,
    Clearcoat     = 1,
    Sheen         = 2,
    Anisotropy    = 3,
    Iridescence   = 4,
    Transmission  = 5,
    Volume        = 6,
    Subsurface    = 7,
    Layers        = 8,
};
constexpr u32 MATERIAL_FEATURE_COUNT = 9;

// The GPU constants of gpu_types.h mirror the enum (checked here, once).
static_assert(MATERIAL_FEATURE_SPECULAR_IOR == (1u << static_cast<u32>(MaterialFeature::SpecularIor)) &&
                  MATERIAL_FEATURE_CLEARCOAT == (1u << static_cast<u32>(MaterialFeature::Clearcoat)) &&
                  MATERIAL_FEATURE_SHEEN == (1u << static_cast<u32>(MaterialFeature::Sheen)) &&
                  MATERIAL_FEATURE_ANISOTROPY == (1u << static_cast<u32>(MaterialFeature::Anisotropy)) &&
                  MATERIAL_FEATURE_IRIDESCENCE == (1u << static_cast<u32>(MaterialFeature::Iridescence)) &&
                  MATERIAL_FEATURE_TRANSMISSION == (1u << static_cast<u32>(MaterialFeature::Transmission)) &&
                  MATERIAL_FEATURE_VOLUME == (1u << static_cast<u32>(MaterialFeature::Volume)) &&
                  MATERIAL_FEATURE_SUBSURFACE == (1u << static_cast<u32>(MaterialFeature::Subsurface)) &&
                  MATERIAL_FEATURE_LAYERS == (1u << static_cast<u32>(MaterialFeature::Layers)) &&
                  MATERIAL_FEATURES_ALL == (1u << MATERIAL_FEATURE_COUNT) - 1u,
              "GPU feature bits out of sync with MaterialFeature");

[[nodiscard]] constexpr u32 featureBit(MaterialFeature f) { return 1u << static_cast<u32>(f); }
[[nodiscard]] const char* featureName(MaterialFeature f);

/// True when the feature's block is present and changes the shading: a block
/// whose weight is zero ("the whole layer is disabled" in the specifications),
/// or a specular/ior block at its neutral values, does not count.
[[nodiscard]] bool featureActive(const MaterialDescription& desc, MaterialFeature f);

/// One bit per active feature (see MaterialFeature).  Core-only materials
/// give 0.
[[nodiscard]] u32 featureMask(const MaterialDescription& desc);

/// Shared, immutable reference to a description: the per-entity form of the
/// extension data (MaterialComponent::extension).  An edit creates a new
/// description and swaps the pointer through ECS modify access.
using MaterialDescriptionPtr = std::shared_ptr<const MaterialDescription>;

/// Human-readable problems (empty = valid): ranges per specification and
/// companion rules (volume needs transmission or a subsurface block, thin or
/// volumetric; volumetric subsurface needs a volume; layer stack bounds).
/// `materialCount` (if known) bounds resolved layer material indices.
[[nodiscard]] std::vector<std::string> validate(const MaterialDescription& desc,
                                                u32 materialCount = ~0u);

/// Values that validate() accepts but that deserve a note (not errors): the
/// subsurface colour above 1, which the draft KHR_materials_scatter leaves
/// unbounded.
[[nodiscard]] std::vector<std::string> validationWarnings(const MaterialDescription& desc);

/// Visit every texture slot (name = the slot in the description file).
void forEachTexture(MaterialDescription& desc,
                    const std::function<void(const char* slot, MaterialTextureRef&)>& fn);
void forEachTexture(const MaterialDescription& desc,
                    const std::function<void(const char* slot, const MaterialTextureRef&)>& fn);

/// KHR_texture_transform analysis of one material.
enum class TextureTransformStatus : u8 {
    None,        // no texture carries a transform
    Uniform,     // every present texture shares one transform: baked into the mesh UVs by the loader
    PerSlot,     // slots differ, or transformed and plain slots mix: per-slot transforms on the GPU (F15.1 ticket 22)
    Unsupported, // see TextureTransformAnalysis::reason
};
struct TextureTransformAnalysis {
    TextureTransformStatus status = TextureTransformStatus::None;
    UvTransform            transform; // valid when Uniform
    /// The UV set (glTF texCoord) every present texture of the material samples; the loader copies that set into UV0.
    /// Meaningful unless status is Unsupported because of mixed sets.
    u32                    uvSet = 0;
    std::string            reason; // when Unsupported
};
[[nodiscard]] TextureTransformAnalysis analyzeTextureTransforms(const MaterialDescription& desc);

/// TT_SLOT_* of a texture slot name of forEachTexture, or ~0u for a slot that has no GPU transform (layer mask / height textures).
[[nodiscard]] u32 textureTransformSlot(const char* slotName);

/// uv' = T * R * S * uv with the matrices of the KHR_texture_transform README.
[[nodiscard]] std::array<float, 2> applyUvTransform(const UvTransform& t, float u, float v);

} // namespace phosphor
