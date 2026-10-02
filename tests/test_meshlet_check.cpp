#include <doctest/doctest.h>

#include "renderer/cull_reference.h"
#include "renderer/meshlet_check.h"
#include "renderer/meshlet_cull_math.h"
#include "renderer/meshlet_cull_reference.h"
#include "renderer/meshlet_layout.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

using namespace phosphor;

// F6.7: the portable half of --debug-meshlets.  The tests build a complete,
// consistent "GPU frame" on the CPU (candidates, decisions, lists, pyramids,
// counters) and check that checkMeshletFrame accepts it and that every kind of
// corruption is reported under the right check name.

namespace {

constexpr float kNear     = 0.05f;
constexpr u32   kVw       = 64;
constexpr u32   kVh       = 48;
constexpr u32   kCapacity = 128;

glm::mat4 reverseZ(float fovY, float aspect, float nearPlane) {
    const float f = 1.0f / std::tan(fovY * 0.5f);
    glm::mat4 proj(0.0f);
    proj[0][0] = f / aspect;
    proj[1][1] = f;
    proj[2][2] = 0.0f;
    proj[2][3] = -1.0f;
    proj[3][2] = nearPlane;
    return proj;
}

std::vector<float> flatten(const HiZPyramid& p) {
    std::vector<float> v;
    for (const auto& l : p.level) v.insert(v.end(), l.begin(), l.end());
    return v;
}

struct Frame {
    GPUMeshletCullParams params{};
    std::vector<GPUInstance> instances;
    std::vector<GPUMaterial> materials;
    std::vector<u32> sceneFlags;
    std::vector<GPUMeshInfo> meshes;
    std::vector<GPUMeshletBounds> bounds;
    std::vector<GPUMeshlet> meshlets;
    std::vector<GPUMeshletCandidate> candidates, bList;
    std::vector<u32> bFlags, decisions;
    GPUMeshletDrawRange ranges[MESHLET_DRAWS] = {};
    std::vector<u32> args;
    GPUMeshletCounters counters{};
    std::vector<float> depthFinal, depthA, history, current, next;
    MeshletListRef cand, bref;
    bool twoPhase = true;
    u32 slotCount = 0;

    MeshletCheckInput input() const {
        MeshletCheckInput in;
        in.params = params;
        in.twoPhase = twoPhase;
        in.capacity = kCapacity;
        in.slotCount = slotCount;
        in.instances = instances;
        in.materials = materials;
        in.sceneFlags = sceneFlags;
        in.meshes = meshes;
        in.bounds = bounds;
        in.meshlets = meshlets;
        in.candidates = candidates;
        in.bList = bList;
        in.bFlags = bFlags;
        in.decisions = decisions;
        in.ranges = ranges;
        in.args = args.data();
        in.counters = counters;
        in.gate = 0;
        in.history = history;
        in.current = current;
        in.next = next;
        in.depth = depthFinal;
        in.width = kVw;
        in.height = kVh;
        return in;
    }

    u32 cullClass(u32 slot) const {
        if ((materials[instances[slot].materialIndex].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u) return 2u;
        return (instances[slot].flags & INSTANCE_FLAG_MIRRORED) != 0u ? 1u : 0u;
    }

    MeshletDecisionInput decisionInput(const GPUMeshletCandidate& c) const {
        return {&instances[c.slot], &bounds[c.meshlet], cullClass(c.slot)};
    }

    u32 count(u32 begin, u32 n, u32 kind) const {
        return static_cast<u32>(std::count(decisions.begin() + begin, decisions.begin() + begin + n, kind));
    }

    // Phase B decisions from `pyr`, written at decisions[capacity + j], plus the counters.
    void recomputePhaseB(const HiZPyramid& pyr) {
        for (u32 j = 0; j < bref.total; ++j)
            decisions[kCapacity + j] = meshletDecisionPhaseB(params, decisionInput(bref.list[j]), pyr);
        recount();
    }

    void recount() {
        counters = {};
        counters.candidates      = cand.total;
        counters.drawnA          = count(0, cand.total, MESHLET_DECISION_DRAWN_A);
        counters.frustum         = count(0, cand.total, MESHLET_DECISION_FRUSTUM);
        counters.cone            = count(0, cand.total, MESHLET_DECISION_CONE);
        counters.historyRejected = count(0, cand.total, MESHLET_DECISION_HISTORY);
        counters.testedB         = bref.total;
        counters.drawnB          = count(kCapacity, bref.total, MESHLET_DECISION_DRAWN_B);
        counters.occludedB       = count(kCapacity, bref.total, MESHLET_DECISION_OCCLUDED);
    }
};

struct Spec {
    u32 mesh, material;
    float x, y, z, scale;
    bool mirrored, valid, visible;
};

Frame buildFrame() {
    Frame f;
    // Materials: 0 single sided, 1 double sided.
    f.materials.assign(2, GPUMaterial{});
    f.materials[1].flags = MATERIAL_FLAG_DOUBLE_SIDED;

    // 4 meshes of 1, 2, 3 and 5 meshlets (11 in total).  Meshlets with a global
    // index = 1 mod 4 carry a normal cone facing -z (rejected from a camera in
    // front of the instance looking down -z), the others have no usable cone.
    const u32 counts[4] = {1, 2, 3, 5};
    u32 off = 0;
    for (u32 c : counts) {
        GPUMeshInfo m{};
        m.meshletCount  = c;
        m.meshletOffset = off;
        f.meshes.push_back(m);
        off += c;
    }
    for (u32 i = 0; i < off; ++i) {
        GPUMeshletBounds b{};
        b.center[0] = (float(i % 3) - 1.0f) * 0.4f;
        b.center[1] = (float(i % 2) - 0.5f) * 0.4f;
        b.radius    = 0.8f;
        for (int k = 0; k < 3; ++k) b.coneApex[k] = b.center[k];
        if (i % 4 == 1) {
            b.coneAxis[2] = -1.0f;
            b.coneCutoff  = 0.3f;
        } else {
            b.coneAxis[2] = 1.0f;
            b.coneCutoff  = 1.0f;
        }
        f.bounds.push_back(b);
        f.meshlets.push_back({i * 8, 8, i * 36, 12});
    }

    // Camera at (0, 0, 10) looking down -z: half-extents at z = 0 are 7.7 x 5.8
    // world units, ~4.2 pixels per unit.
    const Spec specs[] = {
        {0, 0, -6.0f, 0.0f, 0, 1.0f, false, true, true},    //  0 behind the occluder (history + occluded)
        {1, 0, -5.0f, 2.0f, 0, 1.0f, false, true, true},    //  1
        {2, 0, -2.5f, 0.0f, 0, 1.0f, false, true, true},    //  2 left of the edge, partly outside the phase-A occluder
        {3, 0, 3.0f, 0.0f, 0, 1.0f, false, true, true},     //  3 drawn
        {1, 0, 5.0f, -2.0f, 0, 1.0f, false, true, true},    //  4
        {0, 0, -60.0f, 0.0f, 0, 1.0f, false, true, true},   //  5 frustum
        {2, 0, 0.0f, 0.0f, 0, 1.0f, false, true, true},     //  6 straddles the occluder edge
        {3, 0, 0.0f, 0.0f, 0, 1.0f, false, false, true},    //  7 invalid slot (visible bit set)
        {1, 0, 1.0f, 1.0f, 0, 1.0f, false, false, false},   //  8 invalid
        {2, 0, 2.0f, 1.0f, 0, 1.0f, false, true, false},    //  9 not visible in sceneFlags
        {3, 0, -4.0f, -3.0f, 0, 1.0f, true, true, true},    // 10 mirrored (class 1), behind the occluder
        {1, 0, 4.0f, 3.0f, 0, 1.0f, true, true, true},      // 11 mirrored, drawn
        {3, 1, 2.0f, 2.0f, 0, 1.0f, false, true, true},     // 12 double sided (class 2)
        {2, 1, -4.5f, 0.0f, 0, 1.0f, false, true, true},    // 13 double sided, behind the occluder
        {3, 0, 6.0f, 1.0f, 0, 0.5f, false, true, true},     // 14 small
        {0, 0, 0.0f, 0.0f, 20.0f, 1.0f, false, true, true}, // 15 behind the camera (frustum)
        {2, 0, -8.0f, -3.0f, 0, 2.0f, false, true, true},   // 16 big, at the left border
        {1, 0, 0.0f, 40.0f, 0, 1.0f, false, true, true},    // 17 above the frustum
        {3, 0, -3.5f, 4.0f, 0, 1.5f, false, true, true},    // 18
        {0, 0, 1.0f, -4.0f, 0, 1.0f, false, true, true},    // 19
    };
    f.slotCount = static_cast<u32>(std::size(specs));
    for (const Spec& s : specs) {
        GPUInstance gi{};
        gi.modelMatrix[0]  = s.mirrored ? -s.scale : s.scale;
        gi.modelMatrix[5]  = s.scale;
        gi.modelMatrix[10] = s.scale;
        gi.modelMatrix[12] = s.x;
        gi.modelMatrix[13] = s.y;
        gi.modelMatrix[14] = s.z;
        gi.modelMatrix[15] = 1.0f;
        gi.meshIndex     = s.mesh;
        gi.materialIndex = s.material;
        gi.flags         = (s.valid ? INSTANCE_FLAG_VALID : 0u) | (s.mirrored ? INSTANCE_FLAG_MIRRORED : 0u) | 1u;
        f.instances.push_back(gi);
        f.sceneFlags.push_back(s.visible ? 1u : 0u);
    }

    // Camera.
    const glm::vec3 eye(0, 0, 10);
    const glm::mat4 vp = reverseZ(glm::radians(60.0f), float(kVw) / float(kVh), kNear) *
                         glm::lookAt(eye, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    GPUMeshletCullParams& p = f.params;
    std::memcpy(p.viewProj, glm::value_ptr(vp), sizeof(p.viewProj));
    std::memcpy(p.prevViewProj, glm::value_ptr(vp), sizeof(p.prevViewProj));
    const auto planes = extractFrustumPlanesReverseZ(vp);
    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 4; ++k) p.planes[i * 4 + k] = planes[static_cast<size_t>(i)][k];
    p.cameraPosition[0] = eye.x;
    p.cameraPosition[1] = eye.y;
    p.cameraPosition[2] = eye.z;
    p.nearPlane   = kNear;
    p.viewport[0] = float(kVw);
    p.viewport[1] = float(kVh);
    p.hizSize[0]  = hizLevel0Size(kVw);
    p.hizSize[1]  = hizLevel0Size(kVh);
    p.hizLevels   = hizLevelCount(p.hizSize[0], p.hizSize[1]);
    p.flags = MESHLET_CULL_FRUSTUM | MESHLET_CULL_CONE | MESHLET_CULL_OCCLUSION | MESHLET_CULL_HISTORY_VALID |
              MESHLET_CULL_RECORD;
    p.slotCount         = f.slotCount;
    p.candidateCapacity = kCapacity;

    // Depth: the final image has an occluder (0.5) on the left half, phase A's
    // only drew its left quarter (so current <= next texel by texel).  The
    // previous frame's occluder was wider (x < 40): objects it hid that are
    // visible now are history-rejected and recovered by phase B.
    std::vector<float> depthHist(size_t(kVw) * kVh, 0.0f);
    f.depthFinal.assign(size_t(kVw) * kVh, 0.0f);
    f.depthA = f.depthFinal;
    for (u32 y = 0; y < kVh; ++y)
        for (u32 x = 0; x < kVw; ++x) {
            if (x < 40) depthHist[size_t(y) * kVw + x] = 0.5f;
            if (x < kVw / 2) f.depthFinal[size_t(y) * kVw + x] = 0.5f;
            if (x < kVw / 4) f.depthA[size_t(y) * kVw + x] = 0.5f;
        }
    const HiZPyramid hist = buildHiZReference(depthHist.data(), kVw, kVh);
    const HiZPyramid cur  = buildHiZReference(f.depthA.data(), kVw, kVh);
    f.history = flatten(hist);
    f.current = flatten(cur);
    f.next    = flatten(buildHiZReference(f.depthFinal.data(), kVw, kVh));

    // Candidates, phase A, flags, B list.
    f.cand = buildCandidatesReference(f.instances, f.meshes, f.materials, f.sceneFlags, f.slotCount);
    f.candidates.assign(kCapacity, {});
    std::copy(f.cand.list.begin(), f.cand.list.end(), f.candidates.begin());
    f.decisions.assign(2 * kCapacity, 0);
    f.bFlags.assign(kCapacity, 0);
    for (u32 i = 0; i < f.cand.total; ++i) {
        f.decisions[i] = meshletDecisionPhaseA(p, f.decisionInput(f.cand.list[i]), &hist);
        f.bFlags[i]    = f.decisions[i] == MESHLET_DECISION_HISTORY ? 1u : 0u;
    }
    f.bref = buildBListReference(f.cand, f.bFlags);
    f.bList.assign(kCapacity, {});
    std::copy(f.bref.list.begin(), f.bref.list.end(), f.bList.begin());
    for (u32 c = 0; c < 3; ++c) {
        f.ranges[c]     = f.cand.ranges[c];
        f.ranges[3 + c] = f.bref.ranges[c];
    }
    f.args.assign(MESHLET_DRAWS * 3, 1u);
    for (u32 d = 0; d < MESHLET_DRAWS; ++d)
        f.args[d * 3] = (f.ranges[d].count + MESHLET_OBJECT_GROUP - 1) / MESHLET_OBJECT_GROUP;
    f.recomputePhaseB(cur);
    return f;
}

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

} // namespace

TEST_CASE("meshlet check: a consistent frame passes and exercises every decision") {
    const Frame f = buildFrame();
    const MeshletCheckResult r = checkMeshletFrame(f.input());
    INFO(formatMeshletCheck(r));
    CHECK(r.pass);
    CHECK(r.failures.empty());
    // Not vacuous: every decision kind occurs, in more than one class.
    CHECK(f.counters.candidates > 0);
    CHECK(r.candidates == f.cand.total);
    CHECK(f.counters.drawnA > 0);
    CHECK(f.counters.frustum > 0);
    CHECK(f.counters.cone > 0);
    CHECK(f.counters.historyRejected > 0);
    CHECK(f.counters.drawnB > 0);
    CHECK(f.counters.occludedB > 0);
    CHECK(r.occludedB == f.counters.occludedB);
    CHECK(r.ambiguousA == 0);
    CHECK(r.ambiguousB == 0);
    CHECK(f.counters.testedB == f.counters.historyRejected);
    for (u32 c = 0; c < 3; ++c) CHECK(f.cand.ranges[c].count > 0); // the three cull classes are populated
    CHECK(f.cand.total < f.slotCount * 5);                         // invalid / hidden slots were skipped
    const std::string text = formatMeshletCheck(r);
    CHECK(contains(text, "PASS"));
    CHECK_FALSE(contains(text, "FAIL"));
}

TEST_CASE("meshlet check: every corruption is reported by its check") {
    Frame f = buildFrame();
    REQUIRE(checkMeshletFrame(f.input()).pass);
    REQUIRE(f.bref.total >= 2);
    const char* expected = nullptr;
    u32 gate = 0;

    SUBCASE("overflow word") {
        f.counters.overflow = 1;
        expected = "overflow";
    }
    SUBCASE("overflow gate") {
        gate = 1;
        expected = "overflow";
    }
    SUBCASE("candidate entry") {
        f.candidates[0].meshlet += 1;
        expected = "candidates";
    }
    SUBCASE("class range") {
        f.ranges[0].count += 1;
        expected = "candidates";
    }
    SUBCASE("indirect arguments") {
        f.args[0] += 1;
        expected = "candidates";
    }
    SUBCASE("phase A decision") {
        u32 i = 0;
        while (f.decisions[i] != MESHLET_DECISION_DRAWN_A) ++i; // robust: ambiguousA == 0
        f.decisions[i] = MESHLET_DECISION_FRUSTUM;
        expected = "decisionsA";
    }
    SUBCASE("B flag") {
        u32 i = 0;
        while (f.decisions[i] != MESHLET_DECISION_DRAWN_A) ++i;
        f.bFlags[i] = 1;
        expected = "bflags";
    }
    SUBCASE("B list entry") {
        size_t other = 1;
        while (other < f.bref.total && f.bList[other].slot == f.bList[0].slot && f.bList[other].meshlet == f.bList[0].meshlet)
            ++other;
        REQUIRE(other < f.bref.total);
        std::swap(f.bList[0], f.bList[other]);
        expected = "blist";
    }
    SUBCASE("B range") {
        f.ranges[3].count += 1;
        expected = "blist";
    }
    SUBCASE("phase B decision") {
        u32& d = f.decisions[kCapacity];
        d = d == MESHLET_DECISION_DRAWN_B ? MESHLET_DECISION_OCCLUDED : MESHLET_DECISION_DRAWN_B;
        expected = "decisionsB";
    }
    SUBCASE("counter off by one") {
        f.counters.drawnA += 1;
        expected = "counters";
    }
    SUBCASE("testedB off by one") {
        f.counters.testedB += 1;
        expected = "counters";
    }
    SUBCASE("next pyramid texel") {
        f.next[0] = f.next[0] == 0.25f ? 0.75f : 0.25f;
        expected = "pyramids";
    }
    SUBCASE("current texel larger than next") {
        // Level 0 texel 0: next is 0.5 there (occluded half), current 1.0 > next.
        f.current[0] = 1.0f;
        expected = "pyramids";
    }
    SUBCASE("history not a min pyramid") {
        f.history[f.history.size() - 1] = 0.9f; // 1x1 level != min of its children
        expected = "pyramids";
    }
    SUBCASE("truncated history is treated as invalid") {
        f.history.resize(f.history.size() / 2); // the CPU then expects no HISTORY decision
        expected = "decisionsA";
    }

    REQUIRE(expected != nullptr);
    MeshletCheckInput in = f.input();
    in.gate = gate;
    const MeshletCheckResult r = checkMeshletFrame(in);
    INFO(formatMeshletCheck(r));
    CHECK_FALSE(r.pass);
    CHECK(contains(r.failures, expected));
    const std::string text = formatMeshletCheck(r);
    CHECK(contains(text, "FAIL"));
    CHECK_FALSE(contains(text, "PASS"));
}

TEST_CASE("meshlet check: a too-near current pyramid loses surfaces") {
    Frame f = buildFrame();
    // Simulate a broken phase B pyramid: every texel 1.0 (nearest possible), so
    // phase B occludes everything usable.  Decisions and counters are
    // recomputed from that pyramid, so decisionsB and counters agree; the
    // true final pyramid stays in `next`.
    HiZPyramid bad = buildHiZReference(f.depthFinal.data(), kVw, kVh);
    for (auto& lvl : bad.level) std::fill(lvl.begin(), lvl.end(), 1.0f);
    f.current = flatten(bad);
    f.recomputePhaseB(bad);
    REQUIRE(f.counters.occludedB > 0);

    const MeshletCheckResult r = checkMeshletFrame(f.input());
    INFO(formatMeshletCheck(r));
    CHECK_FALSE(r.pass);
    CHECK(contains(r.failures, "lost"));
    CHECK(contains(r.failures, "pyramids")); // current > next as well
    CHECK_FALSE(contains(r.failures, "decisionsB"));
    CHECK_FALSE(contains(r.failures, "counters"));
}

TEST_CASE("meshlet check: single-phase frames skip the B checks") {
    Frame f = buildFrame();
    f.twoPhase = false;
    // Garbage in everything phase B owns.
    std::swap(f.bList[0], f.bList[f.bref.total - 1]);
    f.ranges[3].count += 7;
    f.bFlags[0] ^= 1u;
    f.next[0] = 0.123f;
    f.current.assign(f.current.size(), 1.0f);
    f.decisions[kCapacity] = MESHLET_DECISION_OCCLUDED;
    f.counters.testedB += 5;
    f.counters.drawnB += 3;
    const MeshletCheckResult r = checkMeshletFrame(f.input());
    INFO(formatMeshletCheck(r));
    CHECK(r.pass);
    CHECK(r.occludedB == 0);

    // Phase A is still checked.
    f.counters.drawnA += 1;
    CHECK_FALSE(checkMeshletFrame(f.input()).pass);
}

TEST_CASE("meshlet check: pyramidFromReadback") {
    std::vector<float> depth(size_t(kVw) * kVh);
    for (size_t i = 0; i < depth.size(); ++i) depth[i] = float(i % 17) / 17.0f;
    const HiZPyramid ref = buildHiZReference(depth.data(), kVw, kVh);
    const std::vector<float> flat = flatten(ref);

    SUBCASE("round trip") {
        const HiZPyramid p = pyramidFromReadback(flat, ref.width0, ref.height0, ref.levels);
        CHECK(p.width0 == ref.width0);
        CHECK(p.height0 == ref.height0);
        REQUIRE(p.levels == ref.levels);
        REQUIRE(p.level.size() == ref.levels);
        for (u32 l = 0; l < p.levels; ++l) {
            CHECK(p.level[l].size() == size_t(hizLevelSize(p.width0, l)) * hizLevelSize(p.height0, l));
            CHECK(p.level[l] == ref.level[l]);
        }
        CHECK(p.at(ref.levels - 1, 0, 0) == ref.at(ref.levels - 1, 0, 0));
    }
    SUBCASE("truncated in the middle of a level keeps only complete levels") {
        const size_t keep = flat.size() - 1; // the 1x1 level is incomplete
        const HiZPyramid p =
            pyramidFromReadback(std::span<const float>(flat.data(), keep), ref.width0, ref.height0, ref.levels);
        CHECK(p.levels == ref.levels - 1);
        CHECK(p.level.size() == ref.levels - 1);
        CHECK(p.level[0] == ref.level[0]);
    }
    SUBCASE("truncated inside level 0") {
        const HiZPyramid p =
            pyramidFromReadback(std::span<const float>(flat.data(), 10), ref.width0, ref.height0, ref.levels);
        CHECK(p.levels == 0);
        CHECK(p.level.empty());
    }
    SUBCASE("empty data") {
        const HiZPyramid p = pyramidFromReadback({}, ref.width0, ref.height0, ref.levels);
        CHECK(p.levels == 0);
    }
    SUBCASE("extra trailing data is ignored") {
        std::vector<float> longer = flat;
        longer.push_back(0.5f);
        const HiZPyramid p = pyramidFromReadback(longer, ref.width0, ref.height0, ref.levels);
        CHECK(p.levels == ref.levels);
        CHECK(p.level[ref.levels - 1] == ref.level[ref.levels - 1]);
    }
}

TEST_CASE("meshlet check: a truncated current pyramid fails without reading out of bounds") {
    // Regression (found by the unit tests): pyramidFromReadback drops an
    // incomplete level, and the phase-B decisions indexed the missing level.
    Frame f = buildFrame();
    MeshletCheckInput in = f.input();
    std::vector<float> truncated(in.current.begin(), in.current.end() - 1);
    in.current = truncated;
    const MeshletCheckResult r = checkMeshletFrame(in);
    CHECK_FALSE(r.pass);
    CHECK(r.failures.find("pyramids") != std::string::npos);
}
