#include "renderer/meshlet_check.h"

#include "renderer/meshlet_cull_math.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace phosphor {

namespace {

constexpr float kBand = 1.0e-3f; // relative perturbation that defines the rounding band

struct Failures {
    MeshletCheckResult& r;
    void add(const char* name, const std::string& detail) {
        r.pass = false;
        r.failures += std::string(name) + ": " + detail + "; ";
    }
};

u32 classOf(const MeshletCheckInput& in, u32 slot) {
    const GPUInstance& gi = in.instances[slot];
    if ((in.materials[gi.materialIndex].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u) return 2u;
    return (gi.flags & INSTANCE_FLAG_MIRRORED) != 0u ? 1u : 0u;
}

/// Decisions under the nominal bounds and the two perturbed ones (loose:
/// larger radius, higher cone threshold -> fewer rejections; tight: the
/// opposite).  `phaseB` selects meshletDecisionPhaseB.
struct Band {
    u32 nominal, loose, tight;
    [[nodiscard]] bool robust() const { return nominal == loose && nominal == tight; }
    [[nodiscard]] bool accepts(u32 d) const { return d == nominal || d == loose || d == tight; }
};

Band decide(const MeshletCheckInput& in, const GPUMeshletCandidate& c, bool phaseB, const HiZPyramid* pyramid) {
    const GPUMeshletBounds& b0 = in.bounds[c.meshlet];
    GPUMeshletBounds loose = b0, tight = b0;
    loose.radius *= 1.0f + kBand;
    tight.radius *= 1.0f - kBand;
    if (b0.coneCutoff < 1.0f) {
        loose.coneCutoff = std::min(b0.coneCutoff + kBand, 1.0f);
        tight.coneCutoff = b0.coneCutoff - kBand;
    }
    const u32 cls = classOf(in, c.slot);
    auto one = [&](const GPUMeshletBounds& b) {
        const MeshletDecisionInput d{&in.instances[c.slot], &b, cls};
        return phaseB ? meshletDecisionPhaseB(in.params, d, *pyramid) : meshletDecisionPhaseA(in.params, d, pyramid);
    };
    return {one(b0), one(loose), one(tight)};
}

/// Every level of `p` is the min of its children (exact).
bool pyramidConsistent(const HiZPyramid& p, std::string& where) {
    for (u32 l = 1; l < p.levels; ++l) {
        const u32 w = hizLevelSize(p.width0, l), h = hizLevelSize(p.height0, l);
        const u32 pw = hizLevelSize(p.width0, l - 1), ph = hizLevelSize(p.height0, l - 1);
        for (u32 y = 0; y < h; ++y)
            for (u32 x = 0; x < w; ++x) {
                float m = 1.0f;
                for (u32 dy = 0; dy < 2; ++dy)
                    for (u32 dx = 0; dx < 2; ++dx)
                        if (2 * x + dx < pw && 2 * y + dy < ph) m = std::min(m, p.at(l - 1, 2 * x + dx, 2 * y + dy));
                if (std::memcmp(&m, &p.level[l][size_t(y) * w + x], 4) != 0) {
                    where = "level " + std::to_string(l) + " texel (" + std::to_string(x) + "," + std::to_string(y) + ")";
                    return false;
                }
            }
    }
    return true;
}

} // namespace

HiZPyramid pyramidFromReadback(std::span<const float> data, u32 width0, u32 height0, u32 levels) {
    HiZPyramid p;
    p.width0 = width0;
    p.height0 = height0;
    p.levels = levels;
    p.level.resize(levels);
    size_t offset = 0;
    for (u32 l = 0; l < levels; ++l) {
        const size_t n = size_t(hizLevelSize(width0, l)) * hizLevelSize(height0, l);
        if (offset + n > data.size()) {
            p.levels = l;
            p.level.resize(l);
            break;
        }
        p.level[l].assign(data.begin() + offset, data.begin() + offset + n);
        offset += n;
    }
    return p;
}

MeshletCheckResult checkMeshletFrame(const MeshletCheckInput& in) {
    MeshletCheckResult r;
    Failures fail{r};
    const GPUMeshletCullParams& p = in.params;

    // ---- overflow ---------------------------------------------------------------------
    if (in.counters.overflow != 0 || in.gate != 0) {
        fail.add("overflow", "overflow " + std::to_string(in.counters.overflow) + " gate " + std::to_string(in.gate) +
                                 " (capacity " + std::to_string(p.candidateCapacity) + ")");
        return r; // nothing else is meaningful: the indexed fallback drew the frame
    }

    // ---- candidates ---------------------------------------------------------------------
    const MeshletListRef cand = buildCandidatesReference(in.instances, in.meshes, in.materials, in.sceneFlags, in.slotCount);
    r.candidates = cand.total;
    if (cand.total > in.capacity) fail.add("candidates", "reference total exceeds the capacity");
    for (u32 c = 0; c < 3; ++c) {
        const GPUMeshletDrawRange& g = in.ranges[c];
        if (g.first != cand.ranges[c].first || g.count != cand.ranges[c].count || g.cullClass != c ||
            g.phase != MESHLET_PHASE_A) {
            fail.add("candidates", "class " + std::to_string(c) + " range GPU {" + std::to_string(g.first) + "," +
                                       std::to_string(g.count) + "} CPU {" + std::to_string(cand.ranges[c].first) + "," +
                                       std::to_string(cand.ranges[c].count) + "}");
        }
        const u32 groups = (cand.ranges[c].count + MESHLET_OBJECT_GROUP - 1) / MESHLET_OBJECT_GROUP;
        if (in.args[c * 3] != groups || in.args[c * 3 + 1] != 1 || in.args[c * 3 + 2] != 1) {
            fail.add("candidates", "class " + std::to_string(c) + " args " + std::to_string(in.args[c * 3]) + " expected " +
                                       std::to_string(groups));
        }
    }
    const u32 total = std::min<u32>(cand.total, static_cast<u32>(std::min<u64>(in.capacity, in.candidates.size())));
    for (u32 i = 0; i < total; ++i) {
        if (in.candidates[i].slot != cand.list[i].slot || in.candidates[i].meshlet != cand.list[i].meshlet) {
            fail.add("candidates", "entry " + std::to_string(i) + " GPU {" + std::to_string(in.candidates[i].slot) + "," +
                                       std::to_string(in.candidates[i].meshlet) + "} CPU {" +
                                       std::to_string(cand.list[i].slot) + "," + std::to_string(cand.list[i].meshlet) + "}");
            break;
        }
    }
    if (!r.pass) return r; // decisions are indexed by the candidate list

    // ---- phase A decisions -----------------------------------------------------------------
    const HiZPyramid history = pyramidFromReadback(in.history, p.hizSize[0], p.hizSize[1], p.hizLevels);
    const bool useHistory = (p.flags & MESHLET_CULL_HISTORY_VALID) != 0u && history.levels == p.hizLevels;
    u32 nA[7] = {};
    u32 badA = 0;
    std::string firstA;
    for (u32 i = 0; i < cand.total; ++i) {
        const u32 g = in.decisions[i];
        if (g < 7) ++nA[g];
        const Band b = decide(in, cand.list[i], false, useHistory ? &history : nullptr);
        if (!b.robust()) ++r.ambiguousA;
        if (!b.accepts(g)) {
            if (badA++ == 0) {
                firstA = "candidate " + std::to_string(i) + " (slot " + std::to_string(cand.list[i].slot) + " meshlet " +
                         std::to_string(cand.list[i].meshlet) + ") GPU " + std::to_string(g) + " CPU " +
                         std::to_string(b.nominal);
            }
        }
        if (in.twoPhase && (in.bFlags[i] != 0u) != (g == MESHLET_DECISION_HISTORY)) {
            fail.add("bflags", "candidate " + std::to_string(i) + " flag " + std::to_string(in.bFlags[i]) + " decision " +
                                   std::to_string(g));
            break;
        }
    }
    if (badA) fail.add("decisionsA", std::to_string(badA) + " mismatches, first " + firstA);
    const GPUMeshletCounters& k = in.counters;
    if (k.candidates != cand.total) fail.add("counters", "candidates " + std::to_string(k.candidates));
    if (k.drawnA != nA[MESHLET_DECISION_DRAWN_A] || k.frustum != nA[MESHLET_DECISION_FRUSTUM] ||
        k.cone != nA[MESHLET_DECISION_CONE] || k.historyRejected != nA[MESHLET_DECISION_HISTORY]) {
        fail.add("counters", "phase A counters differ from the recorded decisions");
    }
    if (k.drawnA + k.frustum + k.cone + k.historyRejected + nA[MESHLET_DECISION_SIZE] != k.candidates) {
        fail.add("counters", "drawnA + frustum + cone + history + size != candidates");
    }
    if (!in.twoPhase) return r;

    // ---- phase B list ---------------------------------------------------------------------------
    const MeshletListRef bref = buildBListReference(cand, in.bFlags);
    for (u32 c = 0; c < 3; ++c) {
        const GPUMeshletDrawRange& g = in.ranges[3 + c];
        if (g.first != bref.ranges[c].first || g.count != bref.ranges[c].count || g.phase != MESHLET_PHASE_B) {
            fail.add("blist", "class " + std::to_string(c) + " range GPU {" + std::to_string(g.first) + "," +
                                  std::to_string(g.count) + "} CPU {" + std::to_string(bref.ranges[c].first) + "," +
                                  std::to_string(bref.ranges[c].count) + "}");
        }
        const u32 groups = (bref.ranges[c].count + MESHLET_OBJECT_GROUP - 1) / MESHLET_OBJECT_GROUP;
        if (in.args[(3 + c) * 3] != groups) fail.add("blist", "class " + std::to_string(c) + " B args");
    }
    for (u32 j = 0; j < bref.total; ++j) {
        if (in.bList[j].slot != bref.list[j].slot || in.bList[j].meshlet != bref.list[j].meshlet) {
            fail.add("blist", "entry " + std::to_string(j));
            break;
        }
    }
    if (k.testedB != bref.total || k.historyRejected != bref.total) {
        fail.add("counters", "testedB " + std::to_string(k.testedB) + " history " + std::to_string(k.historyRejected) +
                                 " B list " + std::to_string(bref.total));
    }

    // ---- pyramids --------------------------------------------------------------------------------
    const HiZPyramid current = pyramidFromReadback(in.current, p.hizSize[0], p.hizSize[1], p.hizLevels);
    const HiZPyramid next    = pyramidFromReadback(in.next, p.hizSize[0], p.hizSize[1], p.hizLevels);
    const HiZPyramid finalRef = buildHiZReference(in.depth.data(), in.width, in.height);
    std::string where;
    if (next.levels != finalRef.levels) {
        fail.add("pyramids", "level count");
    } else {
        for (u32 l = 0; l < next.levels && r.pass; ++l) {
            if (std::memcmp(next.level[l].data(), finalRef.level[l].data(), next.level[l].size() * 4) != 0) {
                for (size_t t = 0; t < next.level[l].size(); ++t) {
                    if (std::memcmp(&next.level[l][t], &finalRef.level[l][t], 4) != 0) {
                        char buf[160];
                        std::snprintf(buf, sizeof buf, "new history level %u texel %zu GPU %.8g CPU(final depth) %.8g", l, t,
                                      static_cast<double>(next.level[l][t]), static_cast<double>(finalRef.level[l][t]));
                        fail.add("pyramids", buf);
                        break;
                    }
                }
            }
        }
    }
    for (u32 l = 0; l < current.levels && l < next.levels; ++l) {
        bool ok = true;
        for (size_t t = 0; t < current.level[l].size() && ok; ++t) ok = current.level[l][t] <= next.level[l][t];
        if (!ok) {
            fail.add("pyramids", "current > final at level " + std::to_string(l) + " (phase A's depth nearer than the final)");
            break;
        }
    }
    if (!pyramidConsistent(current, where)) fail.add("pyramids", "current not a min pyramid at " + where);
    if (useHistory && !pyramidConsistent(history, where)) fail.add("pyramids", "history not a min pyramid at " + where);

    // ---- phase B decisions and lost surfaces -----------------------------------------------------
    u32 nB[7] = {};
    u32 badB = 0, lost = 0;
    std::string firstB, firstLost;
    for (u32 j = 0; j < bref.total; ++j) {
        const u32 g = in.decisions[in.capacity + j];
        if (g < 7) ++nB[g];
        const Band b = decide(in, bref.list[j], true, &current);
        if (!b.robust()) ++r.ambiguousB;
        if (!b.accepts(g) && badB++ == 0) {
            firstB = "B entry " + std::to_string(j) + " GPU " + std::to_string(g) + " CPU " + std::to_string(b.nominal);
        }
        if (g == MESHLET_DECISION_OCCLUDED) {
            ++r.occludedB;
            // Against the FINAL depth, with the most occlusion-friendly bounds
            // of the band: still visible = a lost surface.
            const Band f = decide(in, bref.list[j], true, &finalRef);
            if (f.tight != MESHLET_DECISION_OCCLUDED && lost++ == 0) {
                firstLost = "B entry " + std::to_string(j) + " (slot " + std::to_string(bref.list[j].slot) + " meshlet " +
                            std::to_string(bref.list[j].meshlet) + ")";
            }
        }
    }
    if (badB) fail.add("decisionsB", std::to_string(badB) + " mismatches, first " + firstB);
    if (lost) fail.add("lost", std::to_string(lost) + " occluded meshlets visible in the final depth, first " + firstLost);
    if (k.drawnB != nB[MESHLET_DECISION_DRAWN_B] || k.occludedB != nB[MESHLET_DECISION_OCCLUDED] ||
        k.drawnB + k.occludedB != k.testedB) {
        fail.add("counters", "phase B counters (drawnB " + std::to_string(k.drawnB) + " occludedB " +
                                 std::to_string(k.occludedB) + " testedB " + std::to_string(k.testedB) + ")");
    }
    return r;
}

std::string formatMeshletCheck(const MeshletCheckResult& r) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "candidates %u | ambiguous A %u B %u | occluded B %u | %s", r.candidates,
                  r.ambiguousA, r.ambiguousB, r.occludedB, r.pass ? "PASS" : "FAIL");
    std::string s = buf;
    if (!r.pass) s += " | " + r.failures;
    return s;
}

} // namespace phosphor
