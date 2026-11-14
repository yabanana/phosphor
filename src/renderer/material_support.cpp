#include "renderer/material_support.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <array>
#include <map>

namespace phosphor {

const char* supportLevelName(SupportLevel level) {
    switch (level) {
        case SupportLevel::Implemented:  return "implemented";
        case SupportLevel::Approximated: return "approximated";
        case SupportLevel::Unsupported:  return "unsupported";
    }
    return "unsupported";
}

const char* inputLabelName(InputLabel label) {
    switch (label) {
        case InputLabel::Ratified:         return "ratified";
        case InputLabel::ReleaseCandidate: return "release-candidate";
        case InputLabel::Draft:            return "draft";
        case InputLabel::Phosphor:         return "phosphor";
    }
    return "ratified";
}

const MaterialSupportTable& currentSupportTable() {
    // F15.1: the base model, the folded emissive strength, the baked uniform
    // texture transform and (ticket 05) ior + specular are rendered.  Each later
    // ticket flips its feature here (and names its approximation).
    static const MaterialSupportTable table = [] {
        MaterialSupportTable t;
        t.base.level                    = SupportLevel::Implemented;
        t.emissiveStrength.level        = SupportLevel::Implemented;
        t.textureTransformUniform.level = SupportLevel::Implemented;
        // F15.1: ior + specular shade every primary light path: the resolve and forward paths (direct sun and
        // analytic lights, environment, GI diffuse weight), ReSTIR and clustered DI (ticket 07: lobe guide), and
        // (ticket 08) the specular transport (SSR, RT reflections, probes/environment split-sum) and the composite
        // diffuse weight.  The remaining declared approximation (decision 06): secondary RT hits (reflection, GI
        // and probe-capture hit surfaces) keep the base reduction albedo * (1 - metallic) + GGX.
        FeatureSupportEntry& specular = t.features[static_cast<u32>(MaterialFeature::SpecularIor)];
        specular.level                = SupportLevel::Approximated;
        specular.approximation        = "secondary_rt_hits_use_base_reduction";
        // F15.1 (ticket 09): clearcoat shades every primary light path: resolve/forward (analytic lights, the sun disk
        // quadrature of the coat lobe, environment split-sum, emission and base attenuated by 1 - c F_c), ReSTIR and
        // clustered DI (the coated BSDF in target and shading, the coat lobe's emitter share by the coat roughness),
        // the transport (one sample chooses the coat or the base lobe by albedo weight, mixture pdf) and the composite,
        // the denoiser (stable lobe key, sharper roughness).  Declared approximations (decision 06): secondary RT hits
        // ignore the coat, and emissive materials keep their uncoated emission as sampled DI emitters.
        FeatureSupportEntry& clearcoat = t.features[static_cast<u32>(MaterialFeature::Clearcoat)];
        clearcoat.level                = SupportLevel::Approximated;
        clearcoat.approximation        = "secondary_rt_hits_ignore_coat;di_emitters_uncoated";
        // F15.1 (ticket 10): sheen shades every primary light path: resolve/forward (Charlie D and V on analytic lights and the
        // sun, the sun lobe point-evaluated at the disk centre; base and environment scaled by the albedo scaling from the
        // sheen table, the exact min form for lights, the view-only form for the environment), ReSTIR and clustered DI
        // (target and shading), the transport weight and the composite (the sheen environment term S E L(N) lives in the
        // composition: GI irradiance / pi, the declared environment without GI).  Declared approximations (decision 06): the
        // transport does not sample the lobe, captured probes do not light the sheen term (the declared environment does),
        // secondary RT hits ignore sheen, and the Charlie visibility albedo is clamped to 1 at grazing views of narrow lobes.
        FeatureSupportEntry& sheen = t.features[static_cast<u32>(MaterialFeature::Sheen)];
        sheen.level                = SupportLevel::Approximated;
        sheen.approximation        = "lobe_not_sampled_environment_term_in_composition;sun_disk_point_evaluated;secondary_rt_hits_ignore_sheen";
        // F15.1 (ticket 11): anisotropy shades every primary light path with the exact anisotropic GGX D and V of the rotated tangent frame
        // (the vertex tangent's handedness and the mirrored-instance sign give a mirrored instance the exact mirror image): resolve/forward
        // (analytic lights, the sun disk keyed on the narrow axis alpha_b), ReSTIR and clustered DI (target and shading), and the transport
        // (anisotropic visible-normal sampling, exact weight).  Declared approximations (decision 06): the environment term keeps the bent
        // normal for the fetch direction and evaluates the split sum at the effective roughness sqrt(alpha_eff), alpha_eff =
        // sqrt((alpha_t^2 + alpha_b^2) / 2) (6-7 % above the exact quadrature at strength 1 on a metal), secondary RT hits ignore the
        // anisotropy, a primitive without tangents resolves it as strength 0, and the narrow-lobe (alpha < 0.05) sun disk and the
        // anisotropy texture are not validated against an independent reference yet.
        FeatureSupportEntry& anisotropy = t.features[static_cast<u32>(MaterialFeature::Anisotropy)];
        anisotropy.level                = SupportLevel::Approximated;
        anisotropy.approximation        = "environment_split_sum_at_effective_roughness;secondary_rt_hits_ignore_anisotropy;no_tangents_means_isotropic";
        // F15.1 (ticket 12): iridescence replaces the Fresnel of the specular lobe with the specification's thin-film Fresnel (the
        // Belcour-Barla closed form of the README, dielectric and metal layers with rgb_mix) at the V.H of every light sample: analytic
        // lights, each sun-disk sample, ReSTIR and clustered DI (target and shading) and the transport weight at the sampled V.H.
        // Declared approximations (decision 06): the environment, the composite diffuse weight 1 - max(E_irid) and the probe split sum
        // evaluate the film once at N.V and hold it constant over the lobe (3.8 % worst on the fixture films against the exact quadrature),
        // secondary RT hits ignore the film.  Interpretation of the unspecified d = 0 (orchestrator decision 2026-11-13): thickness 0
        // disables the film exactly (the base model); positive thicknesses use the specification's function as written, without the
        // smoothstep towards the outside ior of the Sample Viewer.  Combinations with the other lobes are composed (the film only replaces
        // the Fresnel; the coat ignores the interface between coat and film as the specification says) but not validated; the
        // factor and thickness textures are not validated against an independent reference yet.
        FeatureSupportEntry& iridescence = t.features[static_cast<u32>(MaterialFeature::Iridescence)];
        iridescence.level                = SupportLevel::Approximated;
        iridescence.approximation        = "environment_film_at_nv_constant_over_lobe;thickness_zero_disables_film;secondary_rt_hits_ignore_iridescence";
        // F15.1 (ticket 16): the subsurface feature in THIN mode (KHR_materials_diffuse_transmission, the draft scatter without a volume, the
        // description file): the Lambert BTDF of lights behind the surface (analytic lights, the sun disk centre, ReSTIR and clustered DI in
        // target and shading with visibility rays leaving from the light's side), the back-side environment (declared weight
        // w color (1 - E_d) (1 - 0.16 N.V), exact within 3 % of the quadrature), and the composite.  Declared approximations (decision 06):
        // the GI irradiance of the visible pixel stands for the irradiance at -N (no probe lookup at -N), secondary RT hits ignore the
        // transmission, the draft's anisotropic scatter split and the weight/colour textures are not validated against an independent
        // reference, and combinations with other lobes are not validated.  The volumetric mode is ticket 17 (unsupported).
        FeatureSupportEntry& subsurface = t.features[static_cast<u32>(MaterialFeature::Subsurface)];
        subsurface.level                = SupportLevel::Approximated;
        subsurface.approximation        = "thin_mode_only;gi_irradiance_at_n_stands_for_minus_n;secondary_rt_hits_ignore_transmission";
        // F15.1 (ticket 13): KHR_materials_transmission + KHR_materials_volume, ray-traced (ADR-0004).  The diffuse base of the visible surface is
        // mixed by 1 - t in every primary path; the reflected lobes keep DI, RT reflections and the denoiser.  With --rt on the transport draws
        // ONE Fresnel-chosen sample per pixel: reflection (today's lobes) or refraction through a GGX microfacet (rough BTDF, separable Smith),
        // traced with RT whatever the reflection roughness band, crossing at most 4 interface events (the visible surface included) that accept the
        // inside back faces of the medium (single-sided closed meshes too), Beer-Lambert c^(x/d) along the real inside segment, thin-walled
        // surfaces without bend or offset, glass behind glass, emitters hit by refracted rays owned by the transport (thin-walled DI covers
        // delta lights behind the pane through the glTF thin BTDF); the denoiser gets the refract lobe id/axis and the IOR-scaled roughness;
        // MetalFX temporal treats the pixel as reactive.  Declared approximations (decisions 06/09): without RT the transmitted radiance is the
        // probe/environment along the refracted direction with the volume thickness hint as path length (never SSR); the transmission factor
        // applies at the camera-visible surface only (interior crossings are pure dielectric interfaces); an event beyond the 4th ends in the
        // environment; Schlick Fresnel (the exact one differs by <= 1.2 % mean on the fixtures); direct light through two interfaces is
        // excluded (caustics); non-shadow GI/probe/secondary RT hits see glass as an opaque base reduction; a rough thin-walled surface and the
        // thickness/transmission textures are not validated against an independent reference.
        // F15.1 (ticket 14): RT shadow rays (sun, DI visibility ReSTIR and clustered, GI shadow rays, reflection secondary lighting) cross glass
        // on a straight line and multiply an RGB transmittance: (1 - Schlick F) per surface at the ray angle, transmission x (1 - metallic) x
        // baseColor tint once at the entry surface (a thin-walled surface counts once), Beer-Lambert from the signed entry/exit distance sum,
        // order independent; opaque and alpha-masked casters stop the ray; the shadow is NOT shifted by refraction (a planar slab shifts it by
        // centimetres) and focuses no light (caustics); the CSM path keeps opaque casters until ticket 15.
        FeatureSupportEntry& transmission = t.features[static_cast<u32>(MaterialFeature::Transmission)];
        transmission.level                = SupportLevel::Approximated;
        transmission.approximation        = "rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;"
                                            "at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;"
                                            "rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque";
        FeatureSupportEntry& volume = t.features[static_cast<u32>(MaterialFeature::Volume)];
        volume.level                = SupportLevel::Approximated;
        volume.approximation        = "beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering";
        return t;
    }();
    return table;
}

namespace {

bool hasExtension(const std::vector<std::string>& list, const char* name) {
    return std::find(list.begin(), list.end(), name) != list.end();
}

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + v[i];
    return out;
}

void describeSource(MaterialFeature f, const MaterialDescription& desc, const MaterialSourceInfo& info,
                    std::string& source, InputLabel& label) {
    std::vector<std::string> names;
    auto add = [&](const char* ext) {
        if (hasExtension(info.extensions, ext)) names.emplace_back(ext);
    };
    switch (f) {
        case MaterialFeature::SpecularIor:
            add("KHR_materials_specular");
            add("KHR_materials_ior");
            break;
        case MaterialFeature::Clearcoat:    add("KHR_materials_clearcoat"); break;
        case MaterialFeature::Sheen:        add("KHR_materials_sheen"); break;
        case MaterialFeature::Anisotropy:   add("KHR_materials_anisotropy"); break;
        case MaterialFeature::Iridescence:  add("KHR_materials_iridescence"); break;
        case MaterialFeature::Transmission: add("KHR_materials_transmission"); break;
        case MaterialFeature::Volume:       add("KHR_materials_volume"); break;
        case MaterialFeature::Subsurface:
            if (desc.subsurface) {
                switch (desc.subsurface->source) {
                    case SubsurfaceSource::KhrMaterialsScatterDraft:
                        source = "KHR_materials_scatter";
                        label  = InputLabel::Draft;
                        return;
                    case SubsurfaceSource::KhrMaterialsDiffuseTransmission:
                        source = "KHR_materials_diffuse_transmission";
                        label  = InputLabel::ReleaseCandidate;
                        return;
                    case SubsurfaceSource::DescriptionFile: break;
                }
            }
            break;
        case MaterialFeature::Layers: break;
    }
    if (names.empty()) {
        source = "material-description-file";
        label  = InputLabel::Phosphor;
    } else {
        source = join(names);
        label  = InputLabel::Ratified;
    }
}

FeatureSupport entry(const char* feature, const FeatureSupportEntry& e, std::string source, InputLabel label) {
    FeatureSupport f;
    f.feature       = feature;
    f.level         = e.level;
    f.approximation = e.level == SupportLevel::Approximated ? e.approximation : "";
    f.source        = std::move(source);
    f.label         = label;
    return f;
}

} // namespace

MaterialSupportReport buildSupportReport(const MaterialDescription& desc, const MaterialSourceInfo& info,
                                         const MaterialSupportTable& table) {
    MaterialSupportReport r;
    r.name              = info.name;
    r.unknownExtensions = info.unknownExtensions;
    r.warnings          = info.warnings;

    r.base = entry("base", table.base, "glTF 2.0 core", InputLabel::Ratified);

    const u32 mask = featureMask(desc);
    for (u32 i = 0; i < MATERIAL_FEATURE_COUNT; ++i) {
        const auto f = static_cast<MaterialFeature>(i);
        if (!(mask & featureBit(f))) continue;
        std::string source;
        InputLabel  label = InputLabel::Ratified;
        describeSource(f, desc, info, source, label);
        const bool volumetric = f == MaterialFeature::Subsurface && desc.subsurface && desc.subsurface->mode == SubsurfaceMode::Volumetric;
        r.features.push_back(entry(featureName(f), volumetric ? table.subsurfaceVolumetric : table.features[i], source, label));
    }
    // glTF encodings that carry no material feature of their own.
    if (info.emissiveStrengthUsed)
        r.encodings.push_back(entry("emissive_strength", table.emissiveStrength, "KHR_materials_emissive_strength",
                                    InputLabel::Ratified));
    switch (info.transformStatus) {
        case TextureTransformStatus::None: break;
        case TextureTransformStatus::Uniform:
            r.encodings.push_back(entry("texture_transform", table.textureTransformUniform, "KHR_texture_transform",
                                        InputLabel::Ratified));
            break;
        case TextureTransformStatus::NonUniform:
            r.encodings.push_back(entry("texture_transform", FeatureSupportEntry{}, "KHR_texture_transform",
                                        InputLabel::Ratified));
            r.warnings.push_back("texture_transform: textures of this material use different transforms or UV sets");
            break;
    }
    return r;
}

namespace {

using LevelCounts = std::array<u32, 3>;

std::string levelList(const std::map<std::string, LevelCounts>& perName) {
    std::string out;
    bool first = true;
    for (const auto& [name, c] : perName)
        for (size_t level = 0; level < 3; ++level) {
            if (!c[level]) continue;
            out += (first ? "" : " ") + name + ":" + supportLevelName(static_cast<SupportLevel>(level)) + "(" +
                   std::to_string(c[level]) + ")";
            first = false;
        }
    return out;
}

void addUnknown(std::vector<std::string>& unknown, const std::vector<std::string>& more) {
    for (const auto& e : more)
        if (!hasExtension(unknown, e.c_str())) unknown.push_back(e);
}

} // namespace

std::string formatSupportSummary(const std::vector<MaterialSupportReport>& materials,
                                 const std::vector<std::string>& assetUnknown) {
    LevelCounts totals{}, base{};
    std::map<std::string, LevelCounts> perFeature, perEncoding;
    std::vector<std::string> unknown = assetUnknown;
    for (const auto& m : materials) {
        if (m.valid()) ++base[static_cast<size_t>(m.base.level)];
        for (const auto& f : m.features) {
            ++totals[static_cast<size_t>(f.level)];
            ++perFeature[f.feature][static_cast<size_t>(f.level)];
        }
        for (const auto& e : m.encodings) ++perEncoding[e.feature][static_cast<size_t>(e.level)];
        addUnknown(unknown, m.unknownExtensions);
    }
    std::string baseText;
    for (size_t level = 0; level < 3; ++level)
        if (base[level])
            baseText += (baseText.empty() ? "" : " ") + std::string(supportLevelName(static_cast<SupportLevel>(level))) +
                        "(" + std::to_string(base[level]) + ")";
    return "MATERIALS count=" + std::to_string(materials.size()) + " base=[" + baseText + "]" +
           " implemented=" + std::to_string(totals[0]) + " approximated=" + std::to_string(totals[1]) +
           " unsupported=" + std::to_string(totals[2]) + " features=[" + levelList(perFeature) + "] encodings=[" +
           levelList(perEncoding) + "] unknown_extensions=[" + join(unknown) + "]";
}

std::string formatSupportDetail(const MaterialSupportReport& report) {
    auto one = [](const FeatureSupport& f) {
        std::string out = " " + f.feature + "=" + supportLevelName(f.level);
        if (!f.approximation.empty()) out += "(" + f.approximation + ")";
        if (f.label != InputLabel::Ratified) out += "[" + std::string(inputLabelName(f.label)) + "]";
        return out;
    };
    std::string out = "MATERIAL '" + report.name + "':" + one(report.base);
    if (!report.features.empty()) {
        out += " features:";
        for (const auto& f : report.features) out += one(f);
    }
    if (!report.encodings.empty()) {
        out += " encodings:";
        for (const auto& f : report.encodings) out += one(f);
    }
    if (!report.unknownExtensions.empty()) out += " unknown_extensions=" + join(report.unknownExtensions);
    for (const auto& w : report.warnings) out += " warning: " + w;
    return out;
}

std::string supportToJson(const std::vector<MaterialSupportReport>& materials,
                          const std::vector<std::string>& assetUnknown) {
    using Json = nlohmann::json;
    auto supportJson = [](const FeatureSupport& f) {
        Json e;
        e["feature"]       = f.feature;
        e["support"]       = supportLevelName(f.level);
        e["approximation"] = f.approximation;
        e["source"]        = f.source;
        e["label"]         = inputLabelName(f.label);
        return e;
    };
    auto countsJson = [](const LevelCounts& c) {
        return Json{{"implemented", c[0]}, {"approximated", c[1]}, {"unsupported", c[2]}};
    };
    Json root;
    root["schema"] = "phosphor.materials.v2";
    root["count"]  = materials.size();
    LevelCounts base{}, features{}, encodings{};
    Json list = Json::array();
    std::vector<std::string> unknown = assetUnknown;
    for (const auto& m : materials) {
        Json mj;
        mj["name"] = m.name;
        if (m.valid()) {
            ++base[static_cast<size_t>(m.base.level)];
            mj["base"] = supportJson(m.base);
        }
        Json fj = Json::array(), ej = Json::array();
        for (const auto& f : m.features) {
            ++features[static_cast<size_t>(f.level)];
            fj.push_back(supportJson(f));
        }
        for (const auto& f : m.encodings) {
            ++encodings[static_cast<size_t>(f.level)];
            ej.push_back(supportJson(f));
        }
        mj["features"]           = std::move(fj);
        mj["encodings"]          = std::move(ej);
        mj["unknown_extensions"] = m.unknownExtensions;
        mj["warnings"]           = m.warnings;
        addUnknown(unknown, m.unknownExtensions);
        list.push_back(std::move(mj));
    }
    root["base"]                     = countsJson(base);
    root["features"]                 = countsJson(features);
    root["encodings"]                = countsJson(encodings);
    root["unknown_extensions"]       = unknown;
    root["asset_unknown_extensions"] = assetUnknown;
    root["materials"]                = std::move(list);
    return root.dump(2);
}

} // namespace phosphor
