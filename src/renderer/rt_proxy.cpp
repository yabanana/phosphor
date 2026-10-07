#include "renderer/rt_proxy.h"
#include "renderer/gpu_scene.h"

#include <meshoptimizer.h>
#include <json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace phosphor {
namespace {

using Json = nlohmann::json;
constexpr u64 kMaxManifestBytes = 16 * 1024 * 1024;
constexpr u32 kSchema = 1;
constexpr const char* kScope = "opaque-geometry-s4";

bool validLevel(RtProxyLevel level) { return u32(level) <= u32(RtProxyLevel::Full); }
bool validPercent(double v) { return std::isfinite(v) && v >= 0 && v <= 100; }
bool validError(double v) { return std::isfinite(v) && v >= 0; }

class Fingerprint {
public:
    void word(u32 value) {
        for (u32 shift = 0; shift < 32; shift += 8) {
            state_ ^= (value >> shift) & 255;
            state_ *= 1099511628211ull;
        }
    }
    void count(size_t value) { word(u32(value)); word(u32(u64(value) >> 32)); }
    void real(float value) { word(std::bit_cast<u32>(value)); }
    std::string str() const {
        std::ostringstream out;
        out << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << state_;
        return out.str();
    }
private:
    u64 state_ = 14695981039346656037ull;
};

bool validFingerprint(const std::string& value) {
    return value.size() == 24 && value.starts_with("fnv1a64:") &&
        std::all_of(value.begin() + 8, value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

std::span<const u32> meshIndices(const GpuScene& scene, const GPUMeshInfo& mesh) {
    if (u64(mesh.indexOffset) + mesh.indexCount > scene.indices().size())
        throw std::invalid_argument("RT proxy mesh index range is outside the scene");
    return std::span(scene.indices()).subspan(mesh.indexOffset, mesh.indexCount);
}

std::span<const GPUVertex> meshVertices(const GpuScene& scene, const GPUMeshInfo& mesh) {
    auto indices = meshIndices(scene, mesh);
    if (indices.empty() || indices.size() % 3 != 0)
        throw std::invalid_argument("RT proxy mesh must be a non-empty triangle list");
    const u64 count = u64(*std::max_element(indices.begin(), indices.end())) + 1;
    if (u64(mesh.vertexOffset) + count > scene.vertices().size())
        throw std::invalid_argument("RT proxy vertex range is outside the scene");
    return std::span(scene.vertices()).subspan(mesh.vertexOffset, size_t(count));
}

std::string validateManifest(const RtProxyManifest& manifest) {
    if (manifest.schema != kSchema) return "unsupported RT proxy manifest schema";
    if (manifest.meshoptimizerVersion != MESHOPTIMIZER_VERSION) return "RT proxy meshoptimizer version mismatch";
    if (!validFingerprint(manifest.geometryFingerprint)) return "invalid RT proxy geometry fingerprint";
    if (manifest.measurementScope != kScope) return "unsupported RT proxy measurement scope";
    if (manifest.scene.empty() || manifest.measurementCorpus.empty()) return "missing RT proxy measurement provenance";
    const RtProxyThresholds limits;
    const auto& t = manifest.thresholds;
    if (!validPercent(t.shadowPercent) || !validPercent(t.primaryBadPercent) ||
        !validPercent(t.acnePercent) || !validError(t.distance95Cm) ||
        t.shadowPercent > limits.shadowPercent || t.primaryBadPercent > limits.primaryBadPercent ||
        t.acnePercent > limits.acnePercent || t.distance95Cm > limits.distance95Cm)
        return "RT proxy thresholds exceed the S4 acceptance policy";
    if (!rtProxyMeetsThresholds(manifest.measured, t)) return "RT proxy measurements do not meet the acceptance policy";
    if (manifest.meshes.empty()) return "RT proxy manifest has no meshes";
    for (size_t i = 0; i < manifest.meshes.size(); ++i) {
        const auto& mesh = manifest.meshes[i];
        if (mesh.mesh != i || !validLevel(mesh.level)) return "RT proxy mesh order/level is invalid";
        if (!mesh.sourceIndexCount || mesh.sourceIndexCount % 3 || !mesh.proxyIndexCount ||
            mesh.proxyIndexCount % 3 || mesh.proxyIndexCount > mesh.sourceIndexCount)
            return "RT proxy mesh index count is invalid";
        if (!validError(mesh.relativeError) || !validError(mesh.objectSpaceError) ||
            !validFingerprint(mesh.indexFingerprint)) return "RT proxy mesh error/fingerprint is invalid";
        if (mesh.level == RtProxyLevel::Full && (mesh.sourceIndexCount != mesh.proxyIndexCount ||
            mesh.relativeError != 0 || mesh.objectSpaceError != 0)) return "full RT proxy mesh is inconsistent";
    }
    return {};
}

RtProxyMesh declaration(u32 mesh, RtProxyLevel level, u32 sourceCount, const RtProxyCookedMesh& cooked) {
    return {mesh, level, sourceCount, u32(cooked.indices.size()), rtProxyIndexFingerprint(cooked.indices),
            cooked.relativeError, cooked.objectSpaceError};
}

void append(RtProxyGeometry& geometry, GPUMeshInfo info, const RtProxyCookedMesh& cooked) {
    if (geometry.indices.size() + cooked.indices.size() > std::numeric_limits<u32>::max())
        throw std::length_error("RT proxy index stream exceeds 32-bit offsets");
    info.indexOffset = u32(geometry.indices.size());
    info.indexCount = u32(cooked.indices.size());
    info.meshletOffset = info.meshletCount = 0;
    geometry.indices.insert(geometry.indices.end(), cooked.indices.begin(), cooked.indices.end());
    geometry.meshes.push_back(info);
    geometry.proxyTriangles += cooked.indices.size() / 3;
}

RtProxyGeometry fullGeometry(const GpuScene& scene, std::string diagnostic) {
    RtProxyGeometry result;
    result.diagnostic = std::move(diagnostic);
    for (u32 i = 0; i < scene.getMeshCount(); ++i) {
        const auto& mesh = scene.meshInfos()[i];
        auto cooked = rtCookProxyMesh(meshVertices(scene, mesh), meshIndices(scene, mesh), RtProxyLevel::Full);
        result.selections.push_back(declaration(i, RtProxyLevel::Full, mesh.indexCount, cooked));
        result.fullTriangles += mesh.indexCount / 3;
        append(result, mesh, cooked);
    }
    return result;
}

// nlohmann numeric conversions otherwise accept negative/wrapped/fractional
// integers; reject those before conversion for all persisted counts and IDs.
u64 readUnsigned(const Json& value) {
    if (!value.is_number_unsigned() && !(value.is_number_integer() && value.get<int64_t>() >= 0))
        throw std::invalid_argument("RT proxy integer field is not unsigned");
    return value.get<u64>();
}
u32 readU32(const Json& value) {
    const auto v = readUnsigned(value);
    if (v > std::numeric_limits<u32>::max()) throw std::invalid_argument("RT proxy integer field overflows u32");
    return u32(v);
}

Json measurementJson(const RtProxyMeasurements& m) {
    return {{"primary_rays", m.primaryRays}, {"shadow_receivers", m.shadowReceivers},
            {"shadow_percent", m.shadowPercent}, {"primary_bad_percent", m.primaryBadPercent},
            {"distance95_cm", m.distance95Cm}, {"acne_percent", m.acnePercent}};
}

} // namespace

const char* rtProxyLevelName(RtProxyLevel level) {
    switch (level) {
        case RtProxyLevel::R10: return "r10b";
        case RtProxyLevel::R25: return "r25b";
        case RtProxyLevel::R50: return "r50b";
        case RtProxyLevel::Full: return "full";
    }
    return "invalid";
}

bool rtProxyParseLevel(std::string_view name, RtProxyLevel& level) {
    for (u32 i = 0; i <= u32(RtProxyLevel::Full); ++i) {
        if (name == rtProxyLevelName(RtProxyLevel(i))) { level = RtProxyLevel(i); return true; }
    }
    return false;
}

RtProxyCookedMesh rtCookProxyMesh(std::span<const GPUVertex> vertices,
                                std::span<const u32> indices, RtProxyLevel level) {
    if (!validLevel(level) || vertices.empty() || indices.empty() || indices.size() % 3 != 0 ||
        indices.size() > std::numeric_limits<u32>::max())
        throw std::invalid_argument("invalid RT proxy triangle geometry or level");
    const u64 vertexCount = u64(*std::max_element(indices.begin(), indices.end())) + 1;
    if (vertexCount > vertices.size()) throw std::invalid_argument("RT proxy index is outside the vertex range");
    for (const auto& v : vertices.first(size_t(vertexCount))) {
        if (!std::isfinite(v.px) || !std::isfinite(v.py) || !std::isfinite(v.pz))
            throw std::invalid_argument("RT proxy position is not finite");
    }
    RtProxyCookedMesh result;
    result.indices.assign(indices.begin(), indices.end());
    if (level == RtProxyLevel::Full || indices.size() <= 12) return result;
    constexpr float ratios[] = {0.1f, 0.25f, 0.5f};
    const auto target = std::max<size_t>(12, size_t(double(indices.size()) * ratios[u32(level)]) / 3 * 3);
    const auto count = meshopt_simplify(result.indices.data(), indices.data(), indices.size(),
        &vertices[0].px, size_t(vertexCount), sizeof(GPUVertex), target, 1.0f,
        meshopt_SimplifyLockBorder, &result.relativeError);
    if (count < 3 || count % 3 != 0 || !validError(result.relativeError)) {
        result.indices.assign(indices.begin(), indices.end());
        result.relativeError = 0;
        return result;
    }
    result.indices.resize(count);
    result.objectSpaceError = result.relativeError *
        meshopt_simplifyScale(&vertices[0].px, size_t(vertexCount), sizeof(GPUVertex));
    if (!validError(result.objectSpaceError)) throw std::invalid_argument("RT proxy simplification error is not finite");
    return result;
}

bool rtProxyMeetsThresholds(const RtProxyMeasurements& m, const RtProxyThresholds& t) {
    return m.primaryRays > 0 && m.shadowReceivers > 0 &&
        validPercent(m.shadowPercent) && validPercent(m.primaryBadPercent) &&
        validError(m.distance95Cm) && validPercent(m.acnePercent) &&
        validPercent(t.shadowPercent) && validPercent(t.primaryBadPercent) &&
        validError(t.distance95Cm) && validPercent(t.acnePercent) &&
        m.shadowPercent <= t.shadowPercent && m.primaryBadPercent <= t.primaryBadPercent &&
        m.distance95Cm <= t.distance95Cm && m.acnePercent <= t.acnePercent;
}

bool rtProxyPromote(std::span<RtProxyLevel> levels, std::span<const double> blame) {
    if (levels.size() != blame.size()) throw std::invalid_argument("RT proxy blame/level count mismatch");
    double total = 0, best = 0;
    size_t top = 0;
    for (size_t i = 0; i < levels.size(); ++i) {
        if (!validLevel(levels[i]) || !validError(blame[i])) throw std::invalid_argument("invalid RT proxy promotion input");
        if (levels[i] == RtProxyLevel::Full) continue;
        total += blame[i];
        if (blame[i] > best) { best = blame[i]; top = i; }
    }
    if (total <= 0) return false;
    if (!std::isfinite(total)) throw std::invalid_argument("RT proxy blame sum overflow");
    bool changed = false;
    for (size_t i = 0; i < levels.size(); ++i) {
        if (levels[i] != RtProxyLevel::Full && blame[i] >= 0.02 * total) {
            levels[i] = RtProxyLevel(u32(levels[i]) + 1);
            changed = true;
        }
    }
    if (!changed) levels[top] = RtProxyLevel(u32(levels[top]) + 1);
    return true;
}

std::string rtProxyGeometryFingerprint(const GpuScene& scene) {
    Fingerprint hash;
    hash.word(1); // fingerprint layout version
    hash.count(scene.vertices().size());
    for (const auto& v : scene.vertices()) {
        for (float f : {v.px, v.py, v.pz, v.nx, v.ny, v.nz, v.tx, v.ty, v.tz, v.tw, v.u, v.v}) hash.real(f);
    }
    hash.count(scene.indices().size());
    for (u32 i : scene.indices()) hash.word(i);
    hash.count(scene.meshInfos().size());
    for (const auto& mesh : scene.meshInfos()) {
        hash.word(mesh.vertexOffset); hash.word(mesh.indexOffset); hash.word(mesh.indexCount);
    }
    return hash.str();
}

std::string rtProxyIndexFingerprint(std::span<const u32> indices) {
    Fingerprint hash;
    hash.count(indices.size());
    for (u32 i : indices) hash.word(i);
    return hash.str();
}

u32 rtProxyMeshoptimizerVersion() { return MESHOPTIMIZER_VERSION; }

RtProxyManifest rtMakeProxyManifest(const GpuScene& scene, std::span<const RtProxyLevel> levels,
                                   const RtProxyMeasurements& measured, std::string sceneName,
                                   std::string measurementCorpus) {
    if (levels.size() != scene.getMeshCount()) throw std::invalid_argument("RT proxy level count does not match scene");
    RtProxyManifest result;
    result.meshoptimizerVersion = MESHOPTIMIZER_VERSION;
    result.scene = std::move(sceneName);
    result.geometryFingerprint = rtProxyGeometryFingerprint(scene);
    result.measurementCorpus = std::move(measurementCorpus);
    result.measured = measured;
    for (u32 i = 0; i < scene.getMeshCount(); ++i) {
        const auto& mesh = scene.meshInfos()[i];
        const auto cooked = rtCookProxyMesh(meshVertices(scene, mesh), meshIndices(scene, mesh), levels[i]);
        result.meshes.push_back(declaration(i, levels[i], mesh.indexCount, cooked));
    }
    const auto error = validateManifest(result);
    if (!error.empty()) throw std::invalid_argument(error);
    return result;
}

bool rtReadProxyManifest(const std::string& path, RtProxyManifest& out, std::string& error) {
    out = {};
    error.clear();
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open RT proxy manifest: " + path);
        // Read a bounded stream rather than trusting a racy file_size check.
        std::string text;
        char chunk[4096];
        while (input.read(chunk, sizeof chunk) || input.gcount()) {
            if (text.size() + size_t(input.gcount()) > kMaxManifestBytes)
                throw std::runtime_error("RT proxy manifest exceeds 16 MiB");
            text.append(chunk, size_t(input.gcount()));
        }
        if (input.bad()) throw std::runtime_error("cannot read RT proxy manifest");
        const Json doc = Json::parse(text);
        RtProxyManifest m;
        m.schema = readU32(doc.at("schema"));
        m.meshoptimizerVersion = readU32(doc.at("meshoptimizer_version"));
        m.scene = doc.at("scene").get<std::string>();
        m.geometryFingerprint = doc.at("geometry_fingerprint").get<std::string>();
        m.measurementScope = doc.at("measurement_scope").get<std::string>();
        m.measurementCorpus = doc.at("measurement_corpus").get<std::string>();
        const auto& t = doc.at("thresholds");
        m.thresholds = {t.at("shadow_percent").get<double>(), t.at("primary_bad_percent").get<double>(),
                        t.at("distance95_cm").get<double>(), t.at("acne_percent").get<double>()};
        const auto& e = doc.at("measured");
        m.measured = {readUnsigned(e.at("primary_rays")), readUnsigned(e.at("shadow_receivers")),
            e.at("shadow_percent").get<double>(), e.at("primary_bad_percent").get<double>(),
            e.at("distance95_cm").get<double>(), e.at("acne_percent").get<double>()};
        const auto& meshes = doc.at("meshes");
        if (!meshes.is_array()) throw std::invalid_argument("RT proxy meshes must be an array");
        for (const auto& e : meshes) {
            RtProxyMesh mesh;
            mesh.mesh = readU32(e.at("mesh"));
            if (!rtProxyParseLevel(e.at("level").get<std::string>(), mesh.level))
                throw std::invalid_argument("unknown RT proxy level");
            mesh.sourceIndexCount = readU32(e.at("source_indices"));
            mesh.proxyIndexCount = readU32(e.at("proxy_indices"));
            mesh.indexFingerprint = e.at("index_fingerprint").get<std::string>();
            mesh.relativeError = e.at("relative_error").get<float>();
            mesh.objectSpaceError = e.at("object_space_error").get<float>();
            m.meshes.push_back(std::move(mesh));
        }
        error = validateManifest(m);
        if (!error.empty()) return false;
        out = std::move(m);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool rtWriteProxyManifest(const std::string& path, const RtProxyManifest& m, std::string& error) {
    error = validateManifest(m);
    if (!error.empty()) return false;
    try {
        Json meshes = Json::array();
        for (const auto& mesh : m.meshes) {
            meshes.push_back({{"mesh", mesh.mesh}, {"level", rtProxyLevelName(mesh.level)},
                {"source_indices", mesh.sourceIndexCount}, {"proxy_indices", mesh.proxyIndexCount},
                {"index_fingerprint", mesh.indexFingerprint}, {"relative_error", mesh.relativeError},
                {"object_space_error", mesh.objectSpaceError}});
        }
        Json doc = {{"schema", m.schema}, {"meshoptimizer_version", m.meshoptimizerVersion}, {"scene", m.scene},
            {"geometry_fingerprint", m.geometryFingerprint}, {"measurement_scope", m.measurementScope},
            {"measurement_corpus", m.measurementCorpus}, {"thresholds", {
                {"shadow_percent", m.thresholds.shadowPercent}, {"primary_bad_percent", m.thresholds.primaryBadPercent},
                {"distance95_cm", m.thresholds.distance95Cm}, {"acne_percent", m.thresholds.acnePercent}}},
            {"measured", measurementJson(m.measured)}, {"meshes", std::move(meshes)}};
        const auto text = doc.dump(2) + "\n";
        if (text.size() > kMaxManifestBytes) throw std::runtime_error("RT proxy manifest exceeds 16 MiB");
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create RT proxy manifest: " + path);
        output << text;
        output.close();
        if (!output) throw std::runtime_error("cannot write RT proxy manifest: " + path);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

std::vector<u32> rtProxyProtectedMeshes(u32 meshCount, std::span<const GPUInstance> instances,
                                      std::span<const GPUMaterial> materials) {
    std::vector<u32> protectedMeshes;
    for (const auto& instance : instances) {
        if (instance.meshIndex >= meshCount) continue;
        bool protect = instance.materialIndex >= materials.size();
        if (!protect) {
            const auto& material = materials[instance.materialIndex];
            // Emission is factor * texture. GltfLoader binds the default white
            // texture when emissiveTexture is absent; that valid index alone
            // must not protect every non-emissive mesh in the scene.
            protect = material.alphaCutoff > 0 || std::any_of(material.emissive, material.emissive + 3,
                [](float factor) { return !std::isfinite(factor) || factor != 0.0f; });
        }
        if (protect) protectedMeshes.push_back(instance.meshIndex);
    }
    std::sort(protectedMeshes.begin(), protectedMeshes.end());
    protectedMeshes.erase(std::unique(protectedMeshes.begin(), protectedMeshes.end()), protectedMeshes.end());
    return protectedMeshes;
}

RtProxyGeometry rtBuildProxyGeometry(const GpuScene& scene, const RtProxyManifest* manifest,
                                    std::span<const u32> protectedMeshes) {
    if (!manifest) return fullGeometry(scene, "no RT proxy manifest; full geometry");
    const auto error = validateManifest(*manifest);
    if (!error.empty()) return fullGeometry(scene, error + "; full geometry");
    if (manifest->geometryFingerprint != rtProxyGeometryFingerprint(scene) || manifest->meshes.size() != scene.getMeshCount())
        return fullGeometry(scene, "RT proxy geometry fingerprint/mesh count mismatch; full geometry");
    RtProxyGeometry result;
    for (u32 i = 0; i < scene.getMeshCount(); ++i) {
        const auto& mesh = scene.meshInfos()[i];
        const auto& declared = manifest->meshes[i];
        const auto cooked = rtCookProxyMesh(meshVertices(scene, mesh), meshIndices(scene, mesh), declared.level);
        if (declared.sourceIndexCount != mesh.indexCount || declared.proxyIndexCount != cooked.indices.size() ||
            declared.indexFingerprint != rtProxyIndexFingerprint(cooked.indices) ||
            declared.relativeError != cooked.relativeError || declared.objectSpaceError != cooked.objectSpaceError)
            return fullGeometry(scene, "RT proxy cooked geometry/error differs from measurement; full geometry");
        const bool protect = std::find(protectedMeshes.begin(), protectedMeshes.end(), i) != protectedMeshes.end();
        if (protect && declared.level != RtProxyLevel::Full) {
            const auto full = rtCookProxyMesh(meshVertices(scene, mesh), meshIndices(scene, mesh), RtProxyLevel::Full);
            result.selections.push_back(declaration(i, RtProxyLevel::Full, mesh.indexCount, full));
            append(result, mesh, full);
        } else {
            result.selections.push_back(declared);
            append(result, mesh, cooked);
        }
        result.fullTriangles += mesh.indexCount / 3;
    }
    result.manifestApplied = true;
    result.diagnostic = "measured RT proxy manifest applied (opaque geometric corpus only)";
    return result;
}

} // namespace phosphor
