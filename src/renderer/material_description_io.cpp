#include "renderer/material_description_io.h"

#include "renderer/gpu_scene.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace phosphor {

namespace {

using Json = nlohmann::json;

constexpr const char* SCHEMA_DESCRIPTION = "phosphor-material-description";
constexpr const char* SCHEMA_OVERRIDES   = "phosphor-material-overrides";

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

const char* const MODE_NAMES[]   = {"thin", "volumetric"};
const char* const SOURCE_NAMES[] = {"description-file", "KHR_materials_scatter", "KHR_materials_diffuse_transmission"};
const char* const CHANNEL_NAMES[] = {"R", "G", "B", "A"};

// ---------------------------------------------------------------------------
// Visitors: one field list serves both writing and (partial) reading.
// ---------------------------------------------------------------------------

struct Writer {
    Json&       out;
    std::string path;

    void f(const char* key, float& v) { out[key] = std::isinf(v) ? Json("inf") : Json(v); }
    template <size_t N>
    void arr(const char* key, std::array<float, N>& v) {
        Json a = Json::array();
        for (float x : v) a.push_back(x);
        out[key] = std::move(a);
    }
    void b(const char* key, bool& v) { out[key] = v; }
    void e(const char* key, u8& v, const char* const* names, size_t n) {
        if (v >= n) fail(path + "." + key + ": invalid enum value");
        out[key] = names[v];
    }
    void tex(const char* key, MaterialTextureRef& ref) {
        if (!ref.present()) return;
        if (ref.path.empty())
            fail(path + "." + key + ": texture has a bindless index but no path; it cannot be written to a file");
        Json t;
        t["path"] = ref.path;
        if (ref.texCoord != 0) t["texCoord"] = ref.texCoord;
        if (ref.transform) {
            t["transform"] = {{"offset", {ref.transform->offset[0], ref.transform->offset[1]}},
                              {"rotation", ref.transform->rotation},
                              {"scale", {ref.transform->scale[0], ref.transform->scale[1]}}};
        }
        out[key] = std::move(t);
    }
    void optf(const char* key, std::optional<float>& v) {
        if (v) out[key] = *v;
    }
    template <typename T, typename Fn>
    void block(const char* key, std::optional<T>& o, Fn fn) {
        if (!o) return;
        Json sub = Json::object();
        Writer w{sub, path + "." + key};
        fn(w, *o);
        out[key] = std::move(sub);
    }
};

struct Reader {
    const Json&           in;
    std::string           path;
    std::set<std::string> used;

    const Json* get(const char* key) {
        if (!in.contains(key)) return nullptr;
        used.insert(key);
        return &in.at(key);
    }
    void f(const char* key, float& v) {
        const Json* j = get(key);
        if (!j) return;
        if (j->is_number()) v = j->get<float>();
        else if (j->is_string() && j->get<std::string>() == "inf") v = DESCRIPTION_INFINITY;
        else fail(path + "." + key + ": expected a number");
    }
    template <size_t N>
    void arr(const char* key, std::array<float, N>& v) {
        const Json* j = get(key);
        if (!j) return;
        if (!j->is_array() || j->size() != N) fail(path + "." + key + ": expected an array of " + std::to_string(N));
        for (size_t i = 0; i < N; ++i) {
            if (!(*j)[i].is_number()) fail(path + "." + key + ": expected numbers");
            v[i] = (*j)[i].get<float>();
        }
    }
    void b(const char* key, bool& v) {
        const Json* j = get(key);
        if (!j) return;
        if (!j->is_boolean()) fail(path + "." + key + ": expected a boolean");
        v = j->get<bool>();
    }
    void e(const char* key, u8& v, const char* const* names, size_t n) {
        const Json* j = get(key);
        if (!j) return;
        if (!j->is_string()) fail(path + "." + key + ": expected a string");
        const std::string s = j->get<std::string>();
        for (size_t i = 0; i < n; ++i)
            if (s == names[i]) {
                v = static_cast<u8>(i);
                return;
            }
        fail(path + "." + key + ": unknown value '" + s + "'");
    }
    void tex(const char* key, MaterialTextureRef& ref) {
        const Json* j = get(key);
        if (!j) return;
        if (j->is_null()) {
            ref.index = INVALID_TEXTURE_INDEX;
            ref.path.clear();
            ref.transform.reset();
            ref.texCoord = 0;
            return;
        }
        if (!j->is_object()) fail(path + "." + key + ": expected an object with a path");
        Reader r{*j, path + "." + key, {}};
        const Json* p = r.get("path");
        if (!p || !p->is_string() || p->get<std::string>().empty()) fail(r.path + ": a texture needs a non-empty path");
        ref.path  = p->get<std::string>();
        ref.index = INVALID_TEXTURE_INDEX; // the caller resolves the path
        if (const Json* tc = r.get("texCoord")) {
            if (!tc->is_number_unsigned()) fail(r.path + ".texCoord: expected a non-negative integer");
            ref.texCoord = tc->get<u32>();
        }
        if (const Json* t = r.get("transform")) {
            if (!t->is_object()) fail(r.path + ".transform: expected an object");
            Reader tr{*t, r.path + ".transform", {}};
            UvTransform xf = ref.transform.value_or(UvTransform{});
            tr.arr("offset", xf.offset);
            tr.f("rotation", xf.rotation);
            tr.arr("scale", xf.scale);
            tr.finish();
            ref.transform = xf.isIdentity() ? std::nullopt : std::optional<UvTransform>(xf);
        }
        r.finish();
    }
    void optf(const char* key, std::optional<float>& v) {
        const Json* j = get(key);
        if (!j) return;
        if (j->is_null()) v.reset();
        else if (j->is_number()) v = j->get<float>();
        else fail(path + "." + key + ": expected a number or null");
    }
    template <typename T, typename Fn>
    void block(const char* key, std::optional<T>& o, Fn fn) {
        const Json* j = get(key);
        if (!j) return;
        if (j->is_null()) {
            o.reset();
            return;
        }
        if (!j->is_object()) fail(path + "." + key + ": expected an object or null");
        if (!o) o.emplace();
        Reader r{*j, path + "." + key, {}};
        fn(r, *o);
        r.finish();
    }
    void finish() const {
        for (auto it = in.begin(); it != in.end(); ++it)
            if (!used.count(it.key())) fail(path + ": unknown key '" + it.key() + "'");
    }
};

// Field lists -----------------------------------------------------------------

template <class V> void visitBase(V& v, BaseMaterial& b) {
    v.arr("baseColor", b.baseColor);
    v.f("metallic", b.metallic);
    v.f("roughness", b.roughness);
    v.f("normalScale", b.normalScale);
    v.f("occlusionStrength", b.occlusionStrength);
    v.arr("emissive", b.emissive);
    v.f("alphaCutoff", b.alphaCutoff);
    v.b("doubleSided", b.doubleSided);
    v.tex("baseColorTexture", b.baseColorTexture);
    v.tex("metallicRoughnessTexture", b.metallicRoughnessTexture);
    v.tex("normalTexture", b.normalTexture);
    v.tex("occlusionTexture", b.occlusionTexture);
    v.tex("emissiveTexture", b.emissiveTexture);
}
template <class V> void visitSpecular(V& v, SpecularBlock& s) {
    v.f("ior", s.ior);
    v.f("specularFactor", s.specularFactor);
    v.arr("specularColor", s.specularColor);
    v.tex("specularTexture", s.specularTexture);
    v.tex("specularColorTexture", s.specularColorTexture);
}
template <class V> void visitClearcoat(V& v, ClearcoatBlock& c) {
    v.f("factor", c.factor);
    v.f("roughness", c.roughness);
    v.f("normalScale", c.normalScale);
    v.tex("factorTexture", c.factorTexture);
    v.tex("roughnessTexture", c.roughnessTexture);
    v.tex("normalTexture", c.normalTexture);
}
template <class V> void visitSheen(V& v, SheenBlock& s) {
    v.arr("color", s.color);
    v.f("roughness", s.roughness);
    v.tex("colorTexture", s.colorTexture);
    v.tex("roughnessTexture", s.roughnessTexture);
}
template <class V> void visitAnisotropy(V& v, AnisotropyBlock& a) {
    v.f("strength", a.strength);
    v.f("rotation", a.rotation);
    v.tex("texture", a.texture);
}
template <class V> void visitIridescence(V& v, IridescenceBlock& i) {
    v.f("factor", i.factor);
    v.f("ior", i.ior);
    v.f("thicknessMin", i.thicknessMin);
    v.f("thicknessMax", i.thicknessMax);
    v.tex("factorTexture", i.factorTexture);
    v.tex("thicknessTexture", i.thicknessTexture);
}
template <class V> void visitTransmission(V& v, TransmissionBlock& t) {
    v.f("factor", t.factor);
    v.tex("texture", t.texture);
}
template <class V> void visitVolume(V& v, VolumeBlock& b) {
    v.f("thickness", b.thickness);
    v.arr("attenuationColor", b.attenuationColor);
    v.f("attenuationDistance", b.attenuationDistance);
    v.tex("thicknessTexture", b.thicknessTexture);
}
template <class V> void visitSubsurface(V& v, SubsurfaceBlock& s) {
    u8 mode = static_cast<u8>(s.mode), source = static_cast<u8>(s.source);
    v.e("mode", mode, MODE_NAMES, 2);
    v.f("weight", s.weight);
    v.arr("color", s.color);
    v.f("radius", s.radius);
    v.arr("radiusScale", s.radiusScale);
    v.f("anisotropy", s.anisotropy);
    v.tex("weightTexture", s.weightTexture);
    v.tex("colorTexture", s.colorTexture);
    v.e("source", source, SOURCE_NAMES, 3);
    v.block("scatterDraft", s.draft, [](auto& sub, ScatterDraftParams& p) {
        sub.f("scatterStrength", p.scatterStrength);
        sub.arr("multiscatterColor", p.multiscatterColor);
        sub.f("scatterAnisotropy", p.scatterAnisotropy);
    });
    s.mode   = static_cast<SubsurfaceMode>(mode);
    s.source = static_cast<SubsurfaceSource>(source);
}

// Layers: an array, handled by hand.
void writeLayers(Json& out, LayerStack& stack, const std::string& path) {
    Writer w{out, path};
    w.tex("mask", stack.mask);
    Json list = Json::array();
    for (MaterialLayer& l : stack.layers) {
        Json lj;
        lj["material"] = l.materialName;
        const u8 ch = static_cast<u8>(l.maskChannel);
        if (ch > 3) fail(path + ": layer mask channel must be R, G, B or A");
        lj["maskChannel"] = CHANNEL_NAMES[ch];
        if (l.heightContrast) lj["heightContrast"] = *l.heightContrast;
        list.push_back(std::move(lj));
    }
    out["layers"] = std::move(list);
}

void readLayers(const Json& in, LayerStack& stack, const std::string& path) {
    Reader r{in, path, {}};
    r.tex("mask", stack.mask);
    if (const Json* list = r.get("layers")) {
        if (!list->is_array()) fail(path + ".layers: expected an array");
        stack.layers.clear();
        size_t i = 0;
        for (const Json& lj : *list) {
            const std::string lp = path + ".layers[" + std::to_string(i++) + "]";
            if (!lj.is_object()) fail(lp + ": expected an object");
            Reader lr{lj, lp, {}};
            MaterialLayer l;
            const Json* m = lr.get("material");
            if (!m || !m->is_string() || m->get<std::string>().empty()) fail(lp + ": a layer needs a material name");
            l.materialName = m->get<std::string>();
            u8 ch = 0;
            lr.e("maskChannel", ch, CHANNEL_NAMES, 4);
            l.maskChannel = static_cast<TextureChannel>(ch);
            lr.optf("heightContrast", l.heightContrast);
            lr.finish();
            stack.layers.push_back(std::move(l));
        }
    }
    r.finish();
}

template <class V> void visitMaterial(V& v, MaterialDescription& d) {
    auto base = [](auto& sub, BaseMaterial& b) { visitBase(sub, b); };
    // The base block always exists: wrap it as an optional-like for the visitor.
    std::optional<BaseMaterial> wrapper = d.base;
    v.block("base", wrapper, base);
    if (wrapper) d.base = *wrapper;
    v.block("specular", d.specular, [](auto& s, SpecularBlock& b) { visitSpecular(s, b); });
    v.block("clearcoat", d.clearcoat, [](auto& s, ClearcoatBlock& b) { visitClearcoat(s, b); });
    v.block("sheen", d.sheen, [](auto& s, SheenBlock& b) { visitSheen(s, b); });
    v.block("anisotropy", d.anisotropy, [](auto& s, AnisotropyBlock& b) { visitAnisotropy(s, b); });
    v.block("iridescence", d.iridescence, [](auto& s, IridescenceBlock& b) { visitIridescence(s, b); });
    v.block("transmission", d.transmission, [](auto& s, TransmissionBlock& b) { visitTransmission(s, b); });
    v.block("volume", d.volume, [](auto& s, VolumeBlock& b) { visitVolume(s, b); });
    v.block("subsurface", d.subsurface, [](auto& s, SubsurfaceBlock& b) { visitSubsurface(s, b); });
}

Json writeMaterial(const NamedMaterial& m) {
    MaterialDescription d = m.description; // visitors take mutable references
    Json j = Json::object();
    j["name"] = m.name;
    Writer w{j, "material '" + m.name + "'"};
    visitMaterial(w, d);
    if (d.layers) {
        Json lj = Json::object();
        writeLayers(lj, *d.layers, w.path + ".layers");
        j["layers"] = std::move(lj);
    }
    return j;
}

/// Read the keys of one material object over `d` (name is handled by the caller).
void readMaterialInto(const Json& j, MaterialDescription& d, const std::string& path) {
    Reader r{j, path, {}};
    r.used.insert("name");
    visitMaterial(r, d);
    if (const Json* lj = r.get("layers")) {
        if (lj->is_null()) {
            d.layers.reset();
        } else {
            if (!lj->is_object()) fail(path + ".layers: expected an object or null");
            if (!d.layers) d.layers.emplace();
            readLayers(*lj, *d.layers, path + ".layers");
        }
    }
    r.finish();
}

Json parseDocument(const std::string& text, const char* schema) {
    Json doc = Json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded()) fail("not valid JSON");
    if (!doc.is_object() || doc.value("schema", "") != schema) fail(std::string("not a ") + schema + " file");
    if (!doc.contains("version") || !doc.at("version").is_number_unsigned() ||
        doc.at("version").get<u32>() != MATERIAL_DESCRIPTION_VERSION)
        fail("unsupported " + std::string(schema) + " version (expected " +
             std::to_string(MATERIAL_DESCRIPTION_VERSION) + ")");
    return doc;
}

std::string joinProblems(const std::string& name, const std::vector<std::string>& problems) {
    std::string out;
    for (const auto& p : problems) out += (out.empty() ? "" : "; ") + std::string("material '") + name + "': " + p;
    return out;
}

} // namespace

bool saveMaterialDescriptions(const std::vector<NamedMaterial>& materials, std::string& json, std::string& error) {
    try {
        Json doc;
        doc["schema"]  = SCHEMA_DESCRIPTION;
        doc["version"] = MATERIAL_DESCRIPTION_VERSION;
        Json list = Json::array();
        for (const auto& m : materials) list.push_back(writeMaterial(m));
        doc["materials"] = std::move(list);
        json = doc.dump(2) + "\n";
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool loadMaterialDescriptions(const std::string& json, std::vector<NamedMaterial>& out, std::string& error) {
    try {
        const Json doc = parseDocument(json, SCHEMA_DESCRIPTION);
        if (!doc.contains("materials") || !doc.at("materials").is_array()) fail("missing 'materials' array");
        std::vector<NamedMaterial> result;
        std::set<std::string> names;
        for (const Json& mj : doc.at("materials")) {
            if (!mj.is_object() || !mj.contains("name") || !mj.at("name").is_string() ||
                mj.at("name").get<std::string>().empty())
                fail("every material needs a non-empty name");
            NamedMaterial m;
            m.name = mj.at("name").get<std::string>();
            if (!names.insert(m.name).second) fail("duplicate material name '" + m.name + "'");
            readMaterialInto(mj, m.description, "material '" + m.name + "'");
            const auto problems = validate(m.description);
            if (!problems.empty()) fail(joinProblems(m.name, problems));
            result.push_back(std::move(m));
        }
        out = std::move(result);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool applyMaterialOverrides(const std::string& json, std::vector<NamedMaterial>& materials, std::string& error,
                            std::vector<size_t>* changed) {
    try {
        const Json doc = parseDocument(json, SCHEMA_OVERRIDES);
        if (!doc.contains("overrides") || !doc.at("overrides").is_array()) fail("missing 'overrides' array");
        std::vector<NamedMaterial> work = materials;
        std::set<size_t> touched;
        for (const Json& oj : doc.at("overrides")) {
            if (!oj.is_object() || !oj.contains("name") || !oj.at("name").is_string())
                fail("every override needs a name");
            const std::string name = oj.at("name").get<std::string>();
            bool found = false;
            for (size_t i = 0; i < work.size(); ++i) {
                if (work[i].name != name) continue;
                found = true;
                readMaterialInto(oj, work[i].description, "override '" + name + "'");
                touched.insert(i);
            }
            if (!found) fail("override names unknown material '" + name + "'");
        }
        for (size_t i : touched) {
            const auto problems = validate(work[i].description);
            if (!problems.empty()) fail(joinProblems(work[i].name, problems));
        }
        materials = std::move(work);
        if (changed) changed->assign(touched.begin(), touched.end());
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool loadMaterialDescriptionFile(const std::string& file, std::vector<NamedMaterial>& out, std::string& error) {
    std::ifstream in(file);
    if (!in) {
        error = "cannot open " + file;
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<NamedMaterial> result;
    if (!loadMaterialDescriptions(ss.str(), result, error)) {
        error = file + ": " + error;
        return false;
    }
    const std::filesystem::path dir = std::filesystem::path(file).parent_path();
    for (auto& m : result)
        forEachTexture(m.description, [&](const char*, MaterialTextureRef& ref) {
            if (!ref.path.empty() && std::filesystem::path(ref.path).is_relative() && !dir.empty())
                ref.path = (dir / ref.path).generic_string();
        });
    out = std::move(result);
    return true;
}

bool saveMaterialDescriptionFile(const std::string& file, const std::vector<NamedMaterial>& materials,
                                 std::string& error) {
    std::string json;
    if (!saveMaterialDescriptions(materials, json, error)) return false;
    std::ofstream out(file);
    if (!out) {
        error = "cannot write " + file;
        return false;
    }
    out << json;
    return true;
}

bool applyMaterialOverridesToScene(const std::string& json, GpuScene& scene, const TexturePathResolver& resolve,
                                   std::string& error) {
    std::vector<NamedMaterial> named;
    std::vector<size_t> sceneIndex;
    const auto& names = scene.materialNames();
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i].empty()) continue;
        named.push_back({names[i], scene.materialDescriptions()[i]});
        sceneIndex.push_back(i);
    }
    std::vector<size_t> changed;
    if (!applyMaterialOverrides(json, named, error, &changed)) return false;
    for (size_t c : changed) {
        MaterialDescription d = named[c].description;
        forEachTexture(d, [&](const char* slot, MaterialTextureRef& ref) {
            if (ref.path.empty() || ref.index != INVALID_TEXTURE_INDEX) return;
            ref.index = resolve(ref.path, ref.colorSpace == TextureColorSpace::SRGB);
            (void)slot;
        });
        scene.setMaterialDescription(static_cast<u32>(sceneIndex[c]), d);
    }
    return true;
}

} // namespace phosphor
