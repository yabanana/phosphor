#include "f9_common.h"

#include "renderer/scene_extract.h"
#include "scene/ecs.h"
#include "scene/gltf_loader.h"
#include "scene/procedural.h"
#include "scene/texture_manager.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#include <fcntl.h>
#include <unistd.h>

namespace f9 {

namespace {


std::filesystem::path shaderBase() {
    const char* env = std::getenv("SOC_SHADER_DIR");
    return env ? env : SOC_SHADER_DIR;
}

void expand(const std::filesystem::path& file, std::set<std::string>& seen, std::string& out) {
    std::ifstream in(file);
    if (!in) throw soc::BenchError("cannot read " + file.string());
    std::string line;
    while (std::getline(in, line)) {
        const size_t i = line.find_first_not_of(" \t");
        if (i != std::string::npos && line.compare(i, 8, "#include") == 0) {
            const size_t q0 = line.find('"', i);
            if (q0 != std::string::npos) {
                const size_t q1 = line.find('"', q0 + 1);
                const std::string rel = line.substr(q0 + 1, q1 - q0 - 1);
                std::filesystem::path p = std::filesystem::path(F9_SHADER_DIR) / rel;
                if (!std::filesystem::exists(p)) p = std::filesystem::path(SOC_SOURCE_DIR) / "src" / rel;
                if (!std::filesystem::exists(p)) p = std::filesystem::path(SOC_SOURCE_DIR) / "shaders" / rel;
                if (!std::filesystem::exists(p)) p = std::filesystem::path(F9_GENERATED_DIR) / rel;
                if (seen.insert(p.string()).second) {
                    out += "// ---- begin " + rel + "\n";
                    expand(p, seen, out);
                    out += "// ---- end " + rel + "\n";
                }
                continue;
            }
        }
        if (i != std::string::npos && line.compare(i, 12, "#pragma once") == 0) continue;
        out += line;
        out += '\n';
    }
}

/// Texture manager that keeps the RGBA pixels on the CPU.
class CpuTextureManager final : public phosphor::TextureManager {
public:
    std::vector<CpuTexture> textures;
    [[nodiscard]] u32 textureCount() const override { return static_cast<u32>(textures.size()); }

protected:
    u32 createTexture(const soc::u8* rgba, u32 width, u32 height, bool sRGB) override {
        CpuTexture t;
        t.width = width;
        t.height = height;
        t.sRGB = sRGB;
        t.rgba.assign(rgba, rgba + size_t(width) * height * 4);
        textures.push_back(std::move(t));
        return static_cast<u32>(textures.size() - 1);
    }
};

bool slab(const f64 lo[3], const f64 hi[3], const Ray& r, const V3& inv, f64 tmax) {
    f64 t0 = r.tmin, t1 = tmax;
    const f64 o[3] = {r.o.x, r.o.y, r.o.z}, id[3] = {inv.x, inv.y, inv.z};
    for (int a = 0; a < 3; ++a) {
        f64 ta = (lo[a] - o[a]) * id[a], tb = (hi[a] - o[a]) * id[a];
        if (ta > tb) std::swap(ta, tb);
        if (std::isnan(ta)) ta = -1e300; // 0 * inf: the ray lies in the slab plane
        if (std::isnan(tb)) tb = 1e300;
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        if (t0 > t1) return false;
    }
    return true;
}

} // namespace

MTL::Library* f9Library(soc::Context& ctx, const std::string& file, bool fastMath) {
    std::set<std::string> seen;
    std::string src;
    expand(std::filesystem::path(F9_SHADER_DIR) / file, seen, src);
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / ("f9_" + std::to_string(getpid()) + "_" + file);
    {
        std::ofstream o(tmp);
        o << src;
    }
    return ctx.library(std::filesystem::relative(tmp, shaderBase()).string(), fastMath);
}

MTL::ComputePipelineState* linkedPipeline(soc::Context& ctx, MTL::Library* lib, const std::string& fn,
                                          const std::vector<std::string>& linked,
                                          const MTL::FunctionConstantValues* constants) {
    MTL4::ComputePipelineDescriptor* d = MTL4::ComputePipelineDescriptor::alloc()->init();
    if (constants) {
        MTL4::SpecializedFunctionDescriptor* s = MTL4::SpecializedFunctionDescriptor::alloc()->init();
        s->setFunctionDescriptor(ctx.function(lib, fn));
        s->setConstantValues(constants);
        ctx.keep(s);
        d->setComputeFunctionDescriptor(s);
    } else {
        d->setComputeFunctionDescriptor(ctx.function(lib, fn));
    }
    if (!linked.empty()) {
        std::vector<NS::Object*> fns;
        for (const std::string& name : linked) fns.push_back(ctx.function(lib, name));
        MTL4::StaticLinkingDescriptor* sl = MTL4::StaticLinkingDescriptor::alloc()->init();
        sl->setFunctionDescriptors(NS::Array::array(fns.data(), fns.size()));
        ctx.keep(sl);
        d->setStaticLinkingDescriptor(sl);
    }
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = ctx.compiler()->newComputePipelineState(d, nullptr, &err);
    d->release();
    if (!pso)
        throw soc::BenchError("pipeline " + fn + ": " +
                              (err ? err->localizedDescription()->utf8String() : std::string("unknown error")));
    ctx.keep(pso);
    return pso;
}

QuietStderr::QuietStderr() {
    std::fflush(stderr);
    saved_ = dup(2);
    const int devNull = open("/dev/null", O_WRONLY);
    if (devNull >= 0) {
        dup2(devNull, 2);
        close(devNull);
    }
}

QuietStderr::~QuietStderr() {
    std::fflush(stderr);
    if (saved_ >= 0) {
        dup2(saved_, 2);
        close(saved_);
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

f64 rayTriangle(const Ray& r, V3 a, V3 b, V3 c, f64* uo, f64* vo) {
    // Watertight test (Woop, Benthin, Wald 2013) in double precision: a ray
    // through a shared edge or vertex hits at least one of the triangles,
    // like the hardware traversal (Moller-Trumbore can miss both).
    const f64 d[3] = {r.d.x, r.d.y, r.d.z};
    int kz = 0;
    if (std::fabs(d[1]) > std::fabs(d[kz])) kz = 1;
    if (std::fabs(d[2]) > std::fabs(d[kz])) kz = 2;
    int kx = (kz + 1) % 3, ky = (kx + 1) % 3;
    if (d[kz] < 0) std::swap(kx, ky);
    const f64 sx = d[kx] / d[kz], sy = d[ky] / d[kz], sz = 1.0 / d[kz];
    const V3 A = a - r.o, B = b - r.o, C = c - r.o;
    auto comp = [](const V3& p, int k) { return k == 0 ? p.x : k == 1 ? p.y : p.z; };
    const f64 ax = comp(A, kx) - sx * comp(A, kz), ay = comp(A, ky) - sy * comp(A, kz);
    const f64 bx = comp(B, kx) - sx * comp(B, kz), by = comp(B, ky) - sy * comp(B, kz);
    const f64 cx = comp(C, kx) - sx * comp(C, kz), cy = comp(C, ky) - sy * comp(C, kz);
    const f64 U = cx * by - cy * bx, V = ax * cy - ay * cx, W = bx * ay - by * ax;
    if ((U < 0 || V < 0 || W < 0) && (U > 0 || V > 0 || W > 0)) return -1;
    const f64 det = U + V + W;
    if (det == 0.0) return -1;
    const f64 T = U * sz * comp(A, kz) + V * sz * comp(B, kz) + W * sz * comp(C, kz);
    const f64 t = T / det;
    if (!(t > r.tmin && t < r.tmax)) return -1;
    if (uo) *uo = V / det;
    if (vo) *vo = W / det;
    return t;
}

f64 rayTriangleLoose(const Ray& r, V3 a, V3 b, V3 c, f64 eps) {
    const V3 e1 = b - a, e2 = c - a, p = cross(r.d, e2);
    const f64 det = dot(e1, p);
    if (det == 0.0) return -1;
    const f64 inv = 1.0 / det;
    const V3 s = r.o - a;
    const f64 u = dot(s, p) * inv;
    const V3 q = cross(s, e1);
    const f64 v = dot(r.d, q) * inv;
    if (u < -eps || v < -eps || u + v > 1 + eps) return -1;
    return dot(e2, q) * inv;
}

CpuBvh::CpuBvh(const TriangleSoup& soup) : soup_(soup) {
    order_.resize(soup.count());
    std::iota(order_.begin(), order_.end(), 0u);
    nodes_.reserve(2 * size_t(soup.count()) / 2 + 1);
    if (soup.count()) build(0, soup.count(), 0);
}

u32 CpuBvh::build(u32 first, u32 count, u32 depth) {
    const u32 idx = static_cast<u32>(nodes_.size());
    nodes_.push_back({});
    Node n;
    for (int a = 0; a < 3; ++a) { n.lo[a] = 1e300; n.hi[a] = -1e300; }
    for (u32 i = first; i < first + count; ++i)
        for (int k = 0; k < 3; ++k) {
            const V3& p = soup_.v[3 * size_t(order_[i]) + k];
            const f64 c[3] = {p.x, p.y, p.z};
            for (int a = 0; a < 3; ++a) { n.lo[a] = std::min(n.lo[a], c[a]); n.hi[a] = std::max(n.hi[a], c[a]); }
        }
    for (int a = 0; a < 3; ++a) { // conservative: slabs in double can still round
        const f64 e = 1e-9 * std::max(1.0, std::max(std::fabs(n.lo[a]), std::fabs(n.hi[a])));
        n.lo[a] -= e;
        n.hi[a] += e;
    }
    if (count <= 4 || depth > 60) {
        n.first = first;
        n.count = count;
        nodes_[idx] = n;
        return idx;
    }
    int axis = 0;
    f64 ext = -1;
    for (int a = 0; a < 3; ++a)
        if (n.hi[a] - n.lo[a] > ext) { ext = n.hi[a] - n.lo[a]; axis = a; }
    auto centre = [&](u32 t) {
        const V3& a = soup_.v[3 * size_t(t)];
        const V3& b = soup_.v[3 * size_t(t) + 1];
        const V3& c = soup_.v[3 * size_t(t) + 2];
        const V3 s = a + b + c;
        return axis == 0 ? s.x : axis == 1 ? s.y : s.z;
    };
    const u32 mid = first + count / 2;
    std::nth_element(order_.begin() + first, order_.begin() + mid, order_.begin() + first + count,
                     [&](u32 x, u32 y) { return centre(x) < centre(y); });
    build(first, mid - first, depth + 1);
    const u32 right = build(mid, first + count - mid, depth + 1);
    n.first = right;
    n.count = 0;
    nodes_[idx] = n;
    return idx;
}

CpuHit CpuBvh::nearest(const Ray& r, const Filter& accept) const {
    CpuHit best;
    if (nodes_.empty()) return best;
    const V3 inv = {1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z};
    u32 stack[128];
    u32 sp = 0;
    stack[sp++] = 0;
    while (sp) {
        const u32 ni = stack[--sp];
        const Node& n = nodes_[ni];
        if (!slab(n.lo, n.hi, r, inv, best.hit() ? best.t : r.tmax)) continue;
        if (n.count) {
            for (u32 i = n.first; i < n.first + n.count; ++i) {
                const u32 t = order_[i];
                f64 u, v;
                const f64 th = rayTriangle(r, soup_.v[3 * size_t(t)], soup_.v[3 * size_t(t) + 1], soup_.v[3 * size_t(t) + 2], &u, &v);
                if (th < 0) continue;
                if (best.hit() && (th > best.t || (th == best.t && t > best.tri))) continue;
                if (accept && !accept(t, th, u, v)) continue;
                best.t = th; best.tri = t; best.u = u; best.v = v;
            }
        } else {
            if (sp + 2 > 128) throw soc::BenchError("CpuBvh stack overflow");
            stack[sp++] = ni + 1;
            stack[sp++] = n.first;
        }
    }
    return best;
}

bool CpuBvh::any(const Ray& r, const Filter& accept) const {
    if (nodes_.empty()) return false;
    const V3 inv = {1.0 / r.d.x, 1.0 / r.d.y, 1.0 / r.d.z};
    u32 stack[128];
    u32 sp = 0;
    stack[sp++] = 0;
    while (sp) {
        const u32 ni = stack[--sp];
        const Node& n = nodes_[ni];
        if (!slab(n.lo, n.hi, r, inv, r.tmax)) continue;
        if (n.count) {
            for (u32 i = n.first; i < n.first + n.count; ++i) {
                const u32 t = order_[i];
                f64 u, v;
                const f64 th = rayTriangle(r, soup_.v[3 * size_t(t)], soup_.v[3 * size_t(t) + 1], soup_.v[3 * size_t(t) + 2], &u, &v);
                if (th >= 0 && (!accept || accept(t, th, u, v))) return true;
            }
        } else {
            if (sp + 2 > 128) throw soc::BenchError("CpuBvh stack overflow");
            stack[sp++] = ni + 1;
            stack[sp++] = n.first;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Scene corpus
// ---------------------------------------------------------------------------

glm::mat4 instanceMatrix(const phosphor::GPUInstance& inst) { return glm::make_mat4(inst.modelMatrix); }

u64 SceneData::totalTriangles() const {
    u64 n = 0;
    for (const auto& i : instances) n += meshTriangles(i.meshIndex);
    return n;
}

V3 SceneData::vertex(u32 mesh, u32 i) const {
    const phosphor::GPUVertex& v = scene.vertices()[scene.meshInfos()[mesh].vertexOffset + i];
    return {v.px, v.py, v.pz};
}

TriangleSoup SceneData::soup() const {
    TriangleSoup s;
    const auto& infos = scene.meshInfos();
    const auto& idx = scene.indices();
    for (u32 inst = 0; inst < instances.size(); ++inst) {
        const glm::dmat4 m = glm::dmat4(instanceMatrix(instances[inst]));
        const phosphor::GPUMeshInfo& mi = infos[instances[inst].meshIndex];
        auto w = [&](u32 local) {
            const V3 p = vertex(instances[inst].meshIndex, local);
            const glm::dvec4 q = m * glm::dvec4(p.x, p.y, p.z, 1.0);
            return V3{q.x, q.y, q.z};
        };
        for (u32 t = 0; t < mi.indexCount / 3; ++t)
            s.add(w(idx[mi.indexOffset + 3 * t]), w(idx[mi.indexOffset + 3 * t + 1]), w(idx[mi.indexOffset + 3 * t + 2]),
                  inst, t, 0);
    }
    return s;
}

void SceneData::bounds(V3& lo, V3& hi) const {
    lo = {1e300, 1e300, 1e300};
    hi = {-1e300, -1e300, -1e300};
    const TriangleSoup s = soup();
    for (const V3& p : s.v) {
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
}

bool loadSponza(SceneData& out, std::string& error) {
    const std::filesystem::path path = std::filesystem::path(SOC_SOURCE_DIR) / "assets/sponza/Sponza.gltf";
    if (!std::filesystem::exists(path)) {
        error = "missing " + path.string() + " (tools/fetch_sponza.py)";
        return false;
    }
    CpuTextureManager tex;
    tex.createDefaultTextures();
    phosphor::ECS ecs;
    phosphor::GltfLoader loader(out.scene, tex, ecs);
    bool loaded = false;
    {
        const QuietStderr quiet; // the loader logs every mesh upload
        loaded = loader.loadFromFile(path.string());
    }
    if (!loaded) {
        error = "GltfLoader failed on " + path.string();
        return false;
    }
    phosphor::FrameScene frame;
    phosphor::extractFrameScene(ecs, out.scene, frame);
    out.name = "sponza";
    out.instances = frame.instances;
    out.materials = frame.materials;
    out.textures = std::move(tex.textures);
    return true;
}

phosphor::MeshHandle addMesh(SceneData& s, const std::vector<glm::vec3>& pos, const std::vector<u32>& idx,
                             const std::vector<glm::vec2>& uv) {
    const QuietStderr quiet;
    std::vector<glm::vec3> n(pos.size(), glm::vec3(0, 1, 0));
    std::vector<glm::vec4> t(pos.size(), glm::vec4(1, 0, 0, 1));
    std::vector<glm::vec2> u = uv.empty() ? std::vector<glm::vec2>(pos.size(), glm::vec2(0)) : uv;
    return s.scene.uploadMesh(pos, n, t, u, idx);
}

void proceduralScene(SceneData& s) {
    using namespace phosphor::ProceduralMeshes;
    const QuietStderr quiet;
    s.name = "procedural";
    const std::vector<phosphor::MeshData> meshes = {generateSphere(1.0f, 64, 32), generateTorus(1.0f, 0.35f, 96, 48),
                                                    generateCube(1.0f), generatePlane(4.0f, 4.0f, 32, 32),
                                                    generateIcosahedron(1.0f, 4, false)};
    phosphor::GPUMaterial mat{};
    mat.baseColor[0] = mat.baseColor[1] = mat.baseColor[2] = mat.baseColor[3] = 1.0f;
    mat.baseColorTex = mat.normalTex = mat.metallicRoughnessTex = mat.occlusionTex = mat.emissiveTex =
        phosphor::INVALID_TEXTURE_INDEX;
    s.materials.push_back(mat);
    for (u32 i = 0; i < meshes.size(); ++i) {
        const phosphor::MeshData& m = meshes[i];
        const phosphor::MeshHandle h = s.scene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices);
        phosphor::GPUInstance inst{};
        const glm::mat4 w = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f * float(i), 0.0f, 0.0f));
        std::memcpy(inst.modelMatrix, glm::value_ptr(w), sizeof(inst.modelMatrix));
        inst.meshIndex = h;
        inst.materialIndex = 0;
        inst.flags = phosphor::INSTANCE_FLAG_VALID;
        inst.generation = i + 1;
        s.instances.push_back(inst);
    }
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

GpuGeometry uploadGeometry(soc::Context& ctx, const SceneData& s) {
    GpuGeometry g;
    g.vertexBytes = s.scene.vertices().size() * sizeof(phosphor::GPUVertex);
    g.indexBytes = s.scene.indices().size() * sizeof(u32);
    g.vertices = ctx.buffer(std::max<u64>(g.vertexBytes, 16));
    g.indices = ctx.buffer(std::max<u64>(g.indexBytes, 16));
    std::memcpy(g.vertices->contents(), s.scene.vertices().data(), g.vertexBytes);
    std::memcpy(g.indices->contents(), s.scene.indices().data(), g.indexBytes);
    return g;
}

MTL4::PrimitiveAccelerationStructureDescriptor* blasDescriptor(soc::Context& ctx, const SceneData& s,
                                                                const GpuGeometry& g, u32 mesh,
                                                                const BlasOptions& o) {
    const phosphor::GPUMeshInfo& mi = s.scene.meshInfos()[mesh];
    auto* geo = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    geo->setVertexBuffer(range(g.vertices, u64(mi.vertexOffset) * sizeof(phosphor::GPUVertex)));
    geo->setVertexFormat(MTL::AttributeFormatFloat3);
    geo->setVertexStride(sizeof(phosphor::GPUVertex));
    geo->setIndexBuffer(range(g.indices, u64(mi.indexOffset) * sizeof(u32), u64(mi.indexCount) * sizeof(u32)));
    geo->setIndexType(MTL::IndexTypeUInt32);
    geo->setTriangleCount(mi.indexCount / 3);
    geo->setOpaque(o.opaque);
    geo->setIntersectionFunctionTableOffset(o.iftOffset);
    ctx.keep(geo);
    auto* d = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    d->setGeometryDescriptors(NS::Array::array(geo));
    d->setUsage(o.usage);
    ctx.keep(d);
    return d;
}

MTL::AccelerationStructure* newAccelerationStructure(soc::Context& ctx, u64 size, AsPlacement placement) {
    MTL::AccelerationStructure* as = nullptr;
    if (placement == AsPlacement::Device) {
        as = ctx.device()->newAccelerationStructure(size);
    } else {
        const MTL::SizeAndAlign sa = ctx.device()->heapAccelerationStructureSizeAndAlign(size);
        MTL::HeapDescriptor* hd = MTL::HeapDescriptor::alloc()->init();
        hd->setType(MTL::HeapTypePlacement);
        hd->setStorageMode(MTL::StorageModePrivate);
        hd->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
        hd->setSize((sa.size + sa.align - 1) / sa.align * sa.align);
        MTL::Heap* heap = ctx.heap(hd);
        hd->release();
        if (!heap) throw soc::BenchError("placement heap for an acceleration structure failed");
        // Leaked on purpose: the harness reuses its command buffer, and under
        // MTL_SHADER_VALIDATION a reused command buffer that saw a released
        // heap crashes at its next commit (F9-S1e; the engine rebuilds its
        // command buffers when a heap leaves residency, MetalContext::
        // refreshCommandBuffer).
        heap->retain();
        as = heap->newAccelerationStructure(sa.size, 0);
    }
    if (!as) throw soc::BenchError("newAccelerationStructure(" + std::to_string(size) + ") failed");
    ctx.adopt(as);
    return as;
}

std::vector<Blas> allocateBlases(soc::Context& ctx, const SceneData& s, const GpuGeometry& g, const BlasOptions& o) {
    std::vector<Blas> out(s.scene.getMeshCount());
    for (u32 m = 0; m < out.size(); ++m) {
        out[m].desc = blasDescriptor(ctx, s, g, m, o);
        out[m].sizes = ctx.device()->accelerationStructureSizes(out[m].desc);
        out[m].as = newAccelerationStructure(ctx, out[m].sizes.accelerationStructureSize, o.placement);
        out[m].triangles = s.meshTriangles(m);
    }
    ctx.commitResidency();
    return out;
}

MTL::Buffer* scratchFor(soc::Context& ctx, const std::vector<Blas>& blases) {
    u64 n = 4096;
    for (const Blas& b : blases) n = std::max<u64>(n, std::max(b.sizes.buildScratchBufferSize, b.sizes.refitScratchBufferSize));
    return ctx.buffer(n, MTL::ResourceStorageModePrivate);
}

double timeEncoder(soc::Context& ctx, const std::function<void(MTL4::ComputeCommandEncoder*)>& fn) {
    soc::CommandTimer t(ctx);
    MTL4::CommandBuffer* cmd = t.begin();
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    fn(e);
    e->endEncoding();
    return std::max(1e-4, t.finish() - ctx.emptySpanMs());
}

double buildBlases(soc::Context& ctx, std::vector<Blas>& blases, MTL::Buffer* scratch) {
    return timeEncoder(ctx, [&](MTL4::ComputeCommandEncoder* e) {
        for (size_t i = 0; i < blases.size(); ++i) {
            if (i) e->barrierAfterEncoderStages(MTL::StageAccelerationStructure, MTL::StageAccelerationStructure,
                                                MTL4::VisibilityOptionDevice);
            e->buildAccelerationStructure(blases[i].as, blases[i].desc, range(scratch));
        }
    });
}

InstanceDesc toInstanceDesc(const phosphor::GPUInstance& inst, MTL::ResourceID blas, u32 userID, u32 options,
                            u32 mask) {
    InstanceDesc d{};
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 3; ++r) d.m[3 * c + r] = inst.modelMatrix[4 * c + r];
    d.options = options;
    d.mask = mask;
    d.iftOffset = 0;
    d.userID = userID;
    d.blas = blas;
    return d;
}

Tlas allocateTlas(soc::Context& ctx, u32 count, MTL::AccelerationStructureUsage usage, AsPlacement placement) {
    Tlas t;
    t.count = count;
    t.instances = ctx.buffer(std::max<size_t>(size_t(count) * sizeof(InstanceDesc), 64));
    auto* d = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
    d->setInstanceCount(count);
    d->setInstanceDescriptorBuffer(range(t.instances));
    d->setInstanceDescriptorStride(sizeof(InstanceDesc));
    d->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    d->setInstanceTransformationMatrixLayout(MTL::MatrixLayoutColumnMajor);
    d->setUsage(usage);
    ctx.keep(d);
    t.desc = d;
    t.sizes = ctx.device()->accelerationStructureSizes(d);
    t.as = newAccelerationStructure(ctx, t.sizes.accelerationStructureSize, placement);
    t.scratch = ctx.buffer(std::max<u64>(std::max(t.sizes.buildScratchBufferSize, t.sizes.refitScratchBufferSize), 4096),
                           MTL::ResourceStorageModePrivate);
    ctx.commitResidency();
    return t;
}

// ---------------------------------------------------------------------------
// Hit checks
// ---------------------------------------------------------------------------

SoupIndex::SoupIndex(const TriangleSoup& soup) {
    for (u32 t = 0; t < soup.count(); ++t) {
        const u32 i = soup.instance[t], g = soup.geometry[t], p = soup.primitive[t];
        if (map.size() <= i) map.resize(i + 1);
        if (map[i].size() <= g) map[i].resize(g + 1);
        if (map[i][g].size() <= p) map[i][g].resize(p + 1, ~0u);
        map[i][g][p] = t;
    }
}

u32 SoupIndex::operator()(u32 instance, u32 geometry, u32 primitive) const {
    if (instance >= map.size() || geometry >= map[instance].size() || primitive >= map[instance][geometry].size())
        return ~0u;
    return map[instance][geometry][primitive];
}

HitCheck checkNearest(const CpuBvh& bvh, const std::vector<Ray>& rays, const std::vector<GpuHit>& gpu,
                      const std::function<u32(const GpuHit&)>& soupIndex, f64 tol, const CpuBvh::Filter& accept) {
    HitCheck c;
    c.rays = static_cast<u32>(rays.size());
    std::vector<soc::u8> kind(rays.size(), 0); // 0 ok, 1 miss mismatch, 2 t mismatch, 3 id mismatch
    std::vector<f64> rel(rays.size(), 0.0);
    const TriangleSoup& s = bvh.soup();
    parallelFor(c.rays, [&](u32 i) {
        const CpuHit h = bvh.nearest(rays[i], accept);
        const GpuHit& g = gpu[i];
        const bool gHit = g.t >= 0;
        if (h.hit() != gHit) { kind[i] = 1; return; }
        if (!gHit) return;
        const f64 scale = std::max(1.0, h.t * length(rays[i].d));
        const f64 err = std::fabs(h.t - f64(g.t)) * length(rays[i].d);
        rel[i] = err / scale;
        if (err > tol * scale) { kind[i] = 2; return; }
        const u32 tri = soupIndex(g);
        if (tri >= s.count()) { kind[i] = 3; return; }
        Ray r = rays[i];
        r.tmin = 0;
        r.tmax = 1e300;
        // The named triangle must contain the hit point up to an edge tolerance: the GPU's
        // watertight float test may report a triangle on a shared edge that the exact double
        // test assigns to the neighbour (same t).
        const f64 tg = rayTriangleLoose(r, s.v[3 * size_t(tri)], s.v[3 * size_t(tri) + 1], s.v[3 * size_t(tri) + 2], 1e-5);
        if (tg < 0 || std::fabs(tg - f64(g.t)) * length(rays[i].d) > tol * scale) kind[i] = 3;
    });
    for (u32 i = 0; i < c.rays; ++i) {
        c.maxRelErr = std::max(c.maxRelErr, rel[i]);
        if (!kind[i]) continue;
        ++c.wrong;
        if (kind[i] == 1) ++c.missMismatch;
        if (kind[i] == 2) ++c.tMismatch;
        if (kind[i] == 3) ++c.idMismatch;
        if (c.firstError.empty()) {
            const CpuHit h = bvh.nearest(rays[i], accept);
            char buf[256];
            std::snprintf(buf, sizeof buf, "ray %u: kind %u, cpu t %.6g tri %u, gpu t %.6g inst %u prim %u", i, kind[i],
                          h.t, h.tri, double(gpu[i].t), gpu[i].instance, gpu[i].primitive);
            c.firstError = buf;
        }
    }
    return c;
}

std::vector<GpuHit> traceNearest(soc::Context& ctx, MTL::AccelerationStructure* as, bool instanced,
                                 const std::vector<Ray>& rays, u32 mask) {
    MTL::Library* lib = f9Library(ctx, "f9_trace.metal");
    MTL::ComputePipelineState* pso = ctx.compute(lib, instanced ? "trace_nearest" : "trace_nearest_blas");
    std::vector<GpuHit> out(rays.size());
    constexpr u32 kBatch = 1u << 20;
    MTL::Buffer* rb = ctx.buffer(size_t(std::min<size_t>(rays.size(), kBatch)) * sizeof(GpuRay) + 64);
    MTL::Buffer* hb = ctx.buffer(size_t(std::min<size_t>(rays.size(), kBatch)) * sizeof(GpuHit) + 64);
    MTL::Buffer* pb = ctx.buffer(16);
    for (size_t first = 0; first < rays.size(); first += kBatch) {
        const u32 n = static_cast<u32>(std::min<size_t>(kBatch, rays.size() - first));
        auto* r = static_cast<GpuRay*>(rb->contents());
        for (u32 i = 0; i < n; ++i) r[i] = toGpu(rays[first + i]);
        const u32 params[4] = {n, mask, 0, 0};
        std::memcpy(pb->contents(), params, sizeof params);
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
        ctx.table()->setResource(as->gpuResourceID(), 0);
        ctx.table()->setAddress(rb->gpuAddress(), 1);
        ctx.table()->setAddress(hb->gpuAddress(), 2);
        ctx.table()->setAddress(pb->gpuAddress(), 3);
        e->setComputePipelineState(pso);
        e->setArgumentTable(ctx.table());
        e->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(64, 1, 1));
        e->endEncoding();
        ctx.submit();
        std::memcpy(out.data() + first, hb->contents(), size_t(n) * sizeof(GpuHit));
    }
    return out;
}

std::vector<Ray> cameraRays(V3 eye, V3 target, f64 vfovDeg, u32 width, u32 height, u32 stride) {
    const V3 fwd = normalize(target - eye);
    const V3 right = normalize(cross(fwd, V3{0, 1, 0}));
    const V3 up = cross(right, fwd);
    const f64 th = std::tan(vfovDeg * 0.5 * 3.14159265358979323846 / 180.0);
    const f64 aspect = f64(width) / f64(height);
    std::vector<Ray> rays;
    rays.reserve(size_t(width / stride + 1) * (height / stride + 1));
    for (u32 y = 0; y < height; y += stride)
        for (u32 x = 0; x < width; x += stride) {
            const f64 u = ((x + 0.5) / width * 2.0 - 1.0) * th * aspect;
            const f64 v = (1.0 - (y + 0.5) / height * 2.0) * th;
            rays.push_back({eye, normalize(fwd + right * u + up * v), 0.0, 1e30});
        }
    return rays;
}

} // namespace f9
