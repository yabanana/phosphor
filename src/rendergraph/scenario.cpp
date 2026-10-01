#include "rendergraph/scenario.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace phosphor::rg {

namespace {

// A signal that its consumers either sample (stored) or recompute from a
// depth texture (OPT-1.2 rematerialisation).
struct Signal {
    TextureRef texture;      // valid when stored
    bool       remat = false;
    u32        seed  = 0;    // producer's seed and slot (value function)
    u32        slot  = 0;
    TextureRef depth;        // depth the value is a function of
    Format     format = Format::Unknown;
};

struct Ctx {
    const ScenarioParams&   params;
    RenderGraph&            graph;
    Scenario&               scenario;
    const SynthExecFactory& factory;
    std::vector<std::string> rematUsed;
    u32 nextSeed = 1;
    std::string suffix;      // " [v1]" for the second and later views
    u32         geometryBase = 0;

    [[nodiscard]] Format hdr() const { return params.wideHdr ? Format::RGBA32Float : Format::RGBA16Float; }
    [[nodiscard]] u32 it(u32 n) const {
        if (n == 0) return 0;
        return std::max(1u, static_cast<u32>(std::lround(static_cast<double>(n) * params.work)));
    }
    [[nodiscard]] u32 tris(u32 n) const {
        return std::max(2u, static_cast<u32>(std::lround(static_cast<double>(n) * params.work)) & ~1u);
    }
    [[nodiscard]] bool remat(const std::string& name) {
        if (std::find(params.remat.begin(), params.remat.end(), name) == params.remat.end()) return false;
        rematUsed.push_back(name);
        return true;
    }
    [[nodiscard]] u32 W() const { return params.width; }
    [[nodiscard]] u32 H() const { return params.height; }
};

// Accesses of one synthetic pass, recorded both in the graph and in its
// SynthPass (what the backend's shader does).
class Setup {
public:
    Setup(Ctx& c, PassBuilder& b, SynthPass& s, PassType type) : c_(c), b_(b), s_(s), type_(type) {}

    [[nodiscard]] Stages stage() const { return type_ == PassType::Compute ? StageDispatch : StageFragment; }

    TextureRef texture(const std::string& name, Format format, u32 w, u32 h) {
        return b_.createTexture(name + c_.suffix, {format, std::max(w, 1u), std::max(h, 1u)});
    }

    void sample(TextureRef t) {
        b_.read(t, Usage::ShaderRead, stage());
        SynthInput in;
        in.texture = t;
        in.depth   = isDepthFormat(c_.graph.resources()[t.resource].texture.format);
        s_.inputs.push_back(in);
    }

    void sample(const Signal& sig) {
        if (!sig.remat) {
            sample(sig.texture);
            return;
        }
        // Recompute from the depth: read the depth if the pass does not already.
        u32 depthInput = ~0u;
        for (u32 i = 0; i < s_.inputs.size(); ++i) {
            if (!s_.inputs[i].remat && s_.inputs[i].texture.resource == sig.depth.resource) depthInput = i;
        }
        if (depthInput == ~0u) {
            depthInput = static_cast<u32>(s_.inputs.size());
            sample(sig.depth);
            s_.inputs.back().rematSource = true;
        }
        SynthInput in;
        in.remat      = true;
        in.rematDepth = depthInput;
        in.rematSeed  = sig.seed;
        in.rematSlot  = sig.slot;
        in.rematFormat = sig.format;
        s_.inputs.push_back(in);
        s_.rematIterations = c_.it(c_.params.rematIterations);
    }

    TextureRef color(TextureRef t, u32 slot, LoadIntent load = LoadIntent::Discard) {
        const TextureRef v = b_.writeColor(t, slot, load);
        s_.colors.push_back({slot, v});
        if (load == LoadIntent::Preserve) s_.fetched.push_back(slot); // blends onto the previous contents
        size(t);
        return v;
    }

    /// A color output of a Geometry pass that consumers may rematerialise
    /// (`name` in ScenarioParams::remat): then nothing is written.
    Signal signal(const std::string& name, Format format, u32 slot, TextureRef depth) {
        Signal sig;
        sig.seed   = s_.seed;
        sig.slot   = slot;
        sig.depth  = depth;
        sig.format = format;
        if (c_.remat(name)) {
            sig.remat = true;
            return sig;
        }
        sig.texture = color(texture(name, format, c_.W(), c_.H()), slot);
        s_.colors.back().depthOnly = true;
        return sig;
    }

    /// Same for a storage output of a compute pass that samples `depth` (full
    /// size) as input `depthInput`.
    Signal signalStorage(const std::string& name, Format format, TextureRef depth, u32 depthInput) {
        Signal sig;
        sig.seed   = s_.seed;
        sig.slot   = static_cast<u32>(s_.storage.size());
        sig.depth  = depth;
        sig.format = format;
        s_.signalDepthInput = depthInput;
        if (c_.remat(name)) {
            sig.remat = true;
            return sig;
        }
        sig.texture = storage(texture(name, format, c_.W(), c_.H()));
        s_.storage.back().depthOnly = true;
        return sig;
    }

    void fetch(TextureRef t, u32 slot) {
        b_.readColor(t, slot);
        s_.fetched.push_back(slot);
        size(t);
    }

    TextureRef depth(TextureRef t, LoadIntent load = LoadIntent::Clear) {
        ClearValue clear;
        clear.depth = 0.0f; // reverse-Z
        const TextureRef v = b_.writeDepth(t, load, clear);
        s_.depthWrite  = true;
        s_.depthTarget = v;
        size(t);
        return v;
    }

    void depthTest(TextureRef t) {
        b_.readDepth(t);
        s_.depthTest   = true;
        s_.depthTarget = t;
        size(t);
    }

    TextureRef storage(TextureRef t) {
        const TextureRef v = b_.write(t, Usage::ShaderWrite, StageDispatch);
        s_.storage.push_back({static_cast<u32>(s_.storage.size()), v, false});
        size(t);
        return v;
    }

    void alu(u32 iterations) { s_.aluIterations = c_.it(iterations); }
    void geometry(u32 triangles, u32 geometrySeed) {
        s_.triangles    = c_.tris(triangles);
        s_.geometrySeed = geometrySeed + c_.geometryBase;
    }
    void hints(u32 h) { b_.setHints(h); }

private:
    void size(TextureRef t) {
        if (s_.width != 0) return;
        const TextureDesc& d = c_.graph.resources()[t.resource].texture;
        s_.width  = d.width;
        s_.height = d.height;
    }

    Ctx&         c_;
    PassBuilder& b_;
    SynthPass&   s_;
    PassType     type_;
};

u32 addSynth(Ctx& c, const std::string& name, SynthKind kind, Queue queue, const std::function<void(Setup&)>& fn) {
    const u32 index = static_cast<u32>(c.graph.passes().size());
    SynthPass sp;
    sp.kind = kind;
    sp.seed = (c.nextSeed++) * 0x9E3779B1u;
    const PassType type = kind == SynthKind::Compute ? PassType::Compute : PassType::Raster;
    c.graph.addPass(
        name + c.suffix, type, queue,
        [&](PassBuilder& b) {
            Setup s(c, b, sp, type);
            fn(s);
            const SynthWork w = synthWork(sp);
            PassCost cost;
            cost.intOps      = w.intOps;
            cost.triangles   = w.triangles;
            cost.invocations = w.invocations;
            b.setCost(cost);
            switch (kind) {
            case SynthKind::Compute:    b.setProfileShaders("synth_cs"); break;
            case SynthKind::Fullscreen: b.setProfileShaders("synth_fullscreen_vs,synth_fs"); break;
            case SynthKind::Geometry:
                b.setProfileShaders(sp.colors.empty() ? "synth_geometry_vs,synth_depth_fs" : "synth_geometry_vs,synth_fs");
                break;
            case SynthKind::None: break;
            }
        },
        c.factory ? c.factory(index) : ExecuteFn{});
    if (c.scenario.synth.size() <= index) c.scenario.synth.resize(index + 1);
    c.scenario.synth[index] = std::move(sp);
    return index;
}

u32 addSynth(Ctx& c, const std::string& name, SynthKind kind, const std::function<void(Setup&)>& fn) {
    return addSynth(c, name, kind, Queue::Graphics, fn);
}

// --- Building blocks -----------------------------------------------------------

std::vector<TextureRef> shadowCascades(Ctx& c, u32 count, u32 triangles) {
    std::vector<TextureRef> maps;
    for (u32 i = 0; i < count; ++i) {
        TextureRef map;
        addSynth(c, "Shadow cascade " + std::to_string(i), SynthKind::Geometry, [&](Setup& s) {
            map = s.depth(s.texture("Shadow map " + std::to_string(i), Format::Depth32Float, c.params.shadowSize,
                                    c.params.shadowSize));
            s.geometry(triangles, 100 + i);
            s.hints(HintGeometryHeavy);
        });
        maps.push_back(map);
    }
    return maps;
}

/// Bloom: `levels` compute downsamples from `hdr`, then upsamples back to
/// level 1; returns the level-1 result.
TextureRef bloom(Ctx& c, TextureRef hdr, u32 levels) {
    std::vector<TextureRef> down;
    TextureRef src = hdr;
    for (u32 l = 1; l <= levels; ++l) {
        const u32 w = std::max(c.W() >> l, 1u), h = std::max(c.H() >> l, 1u);
        TextureRef out;
        addSynth(c, "Bloom down " + std::to_string(l), SynthKind::Compute, [&](Setup& s) {
            s.sample(src);
            out = s.storage(s.texture("Bloom down " + std::to_string(l), c.hdr(), w, h));
            s.alu(l == 1 ? 8 : 4);
        });
        down.push_back(out);
        src = out;
    }
    TextureRef up = down.back();
    for (u32 l = levels - 1; l >= 1; --l) {
        const u32 w = std::max(c.W() >> l, 1u), h = std::max(c.H() >> l, 1u);
        TextureRef out;
        addSynth(c, "Bloom up " + std::to_string(l), SynthKind::Compute, [&](Setup& s) {
            s.sample(down[l - 1]);
            s.sample(up);
            out = s.storage(s.texture("Bloom up " + std::to_string(l), c.hdr(), w, h));
            s.alu(4);
        });
        up = out;
        if (l == 1) break;
    }
    return up;
}

/// TAA with a ping-pong history owned by the backend.
TextureRef taa(Ctx& c, TextureRef hdr, const Signal& velocity, TextureRef depth) {
    const TextureDesc desc{c.hdr(), c.W(), c.H()};
    const TextureRef prev = c.graph.importTexture("TAA history (previous)" + c.suffix, desc, ImportContentsDefined);
    TextureRef next = c.graph.importTexture("TAA history" + c.suffix, desc, ImportContentsDefined | ImportOutput);
    c.scenario.imports.push_back({prev.resource, ScenarioImport::Role::HistoryRead, desc});
    c.scenario.imports.push_back({next.resource, ScenarioImport::Role::HistoryWrite, desc});
    addSynth(c, "TAA", SynthKind::Compute, [&](Setup& s) {
        s.sample(hdr);
        s.sample(depth);
        s.sample(velocity);
        s.sample(prev);
        next = s.storage(next);
        s.alu(16);
    });
    return next;
}

/// Tonemap into an LDR target, then the UI blended onto it (both fullscreen,
/// same size: fusable).
TextureRef tonemapAndUi(Ctx& c, const std::vector<TextureRef>& inputs) {
    TextureRef ldr;
    addSynth(c, "Tonemap", SynthKind::Fullscreen, [&](Setup& s) {
        for (const TextureRef& t : inputs) s.sample(t);
        ldr = s.color(s.texture("LDR", Format::RGBA8Unorm, c.W(), c.H()), 0);
        s.alu(8);
    });
    addSynth(c, "UI", SynthKind::Fullscreen, [&](Setup& s) {
        ldr = s.color(ldr, 0, LoadIntent::Preserve);
        s.alu(4);
    });
    return ldr;
}

// --- Scenarios --------------------------------------------------------------------

// 0: deferred renderer: cascaded shadows, G-buffer, SSAO, deferred lighting
// (framebuffer fetch of the G-buffer), bloom, TAA, tonemap, UI.
void buildDeferred(Ctx& c) {
    const std::vector<TextureRef> shadows = shadowCascades(c, 4, 400000);
    TextureRef depth, albedo, normal, material;
    Signal velocity;
    addSynth(c, "G-buffer", SynthKind::Geometry, [&](Setup& s) {
        depth    = s.depth(s.texture("Depth", Format::Depth32Float, c.W(), c.H()));
        albedo   = s.color(s.texture("Albedo", Format::RGBA8Unorm, c.W(), c.H()), 0);
        normal   = s.color(s.texture("Normal", Format::RGB10A2Unorm, c.W(), c.H()), 1);
        material = s.color(s.texture("Material", Format::RGBA8Unorm, c.W(), c.H()), 2);
        velocity = s.signal("Velocity", Format::RG16Float, 3, depth);
        s.geometry(600000, 1);
        s.alu(24);
        s.hints(HintGeometryHeavy);
    });
    TextureRef aoRaw, ao;
    addSynth(c, "SSAO", SynthKind::Compute, [&](Setup& s) {
        s.sample(depth);
        s.sample(normal);
        aoRaw = s.storage(s.texture("SSAO raw", Format::R8Unorm, c.W() / 2, c.H() / 2));
        s.alu(48);
    });
    addSynth(c, "SSAO blur", SynthKind::Compute, [&](Setup& s) {
        s.sample(aoRaw);
        ao = s.storage(s.texture("SSAO", Format::R8Unorm, c.W() / 2, c.H() / 2));
        s.alu(8);
    });
    TextureRef hdr;
    addSynth(c, "Lighting", SynthKind::Fullscreen, [&](Setup& s) {
        s.fetch(albedo, 0);
        s.fetch(normal, 1);
        s.fetch(material, 2);
        for (const TextureRef& m : shadows) s.sample(m);
        s.sample(ao);
        hdr = s.color(s.texture("HDR", c.hdr(), c.W(), c.H()), 4);
        s.alu(96);
        s.hints(HintFragmentHeavy);
    });
    const TextureRef bloomed = bloom(c, hdr, 5);
    const TextureRef resolved = taa(c, hdr, velocity, depth);
    c.scenario.output = tonemapAndUi(c, {resolved, bloomed});
}

// 1: forward+ with cascaded shadows: depth prepass, light grid, forward
// opaque (depth test only), sky, transparents, short bloom, TAA, tonemap, UI.
void buildForwardPlus(Ctx& c) {
    const std::vector<TextureRef> shadows = shadowCascades(c, 3, 400000);
    TextureRef depth;
    addSynth(c, "Depth prepass", SynthKind::Geometry, [&](Setup& s) {
        depth = s.depth(s.texture("Depth", Format::Depth32Float, c.W(), c.H()));
        s.geometry(600000, 1);
        s.hints(HintGeometryHeavy);
    });
    TextureRef grid;
    addSynth(c, "Light grid", SynthKind::Compute, [&](Setup& s) {
        s.sample(depth);
        grid = s.storage(s.texture("Light grid", Format::RGBA16Float, (c.W() + 15) / 16, (c.H() + 15) / 16));
        s.alu(256);
    });
    TextureRef hdr;
    Signal velocity;
    addSynth(c, "Forward opaque", SynthKind::Geometry, [&](Setup& s) {
        s.depthTest(depth);
        for (const TextureRef& m : shadows) s.sample(m);
        s.sample(grid);
        hdr      = s.color(s.texture("HDR", c.hdr(), c.W(), c.H()), 0);
        velocity = s.signal("Velocity", Format::RG16Float, 1, depth);
        s.geometry(600000, 1);
        s.alu(128);
        s.hints(HintFragmentHeavy);
    });
    addSynth(c, "Sky", SynthKind::Fullscreen, [&](Setup& s) {
        hdr = s.color(hdr, 0, LoadIntent::Preserve);
        s.alu(8);
    });
    addSynth(c, "Transparents", SynthKind::Geometry, [&](Setup& s) {
        s.depthTest(depth);
        hdr = s.color(hdr, 0, LoadIntent::Preserve);
        s.geometry(100000, 1);
        s.alu(32);
    });
    const TextureRef bloomed = bloom(c, hdr, 3);
    const TextureRef resolved = taa(c, hdr, velocity, depth);
    c.scenario.output = tonemapAndUi(c, {resolved, bloomed});
}

// 2: long post-processing chain on a forward scene: depth of field, motion
// blur, bloom, lens flare, exposure, tonemap, grain, FXAA, UI.
void buildPostChain(Ctx& c) {
    TextureRef depth, hdr;
    Signal velocity;
    addSynth(c, "Scene", SynthKind::Geometry, [&](Setup& s) {
        depth    = s.depth(s.texture("Depth", Format::Depth32Float, c.W(), c.H()));
        hdr      = s.color(s.texture("HDR", c.hdr(), c.W(), c.H()), 0);
        velocity = s.signal("Velocity", Format::RG16Float, 1, depth);
        s.geometry(300000, 1);
        s.alu(32);
    });
    Signal coc;
    addSynth(c, "DoF CoC", SynthKind::Compute, [&](Setup& s) {
        s.sample(depth);
        // Rematerialised: the pass has no output left and the graph culls it.
        coc = s.signalStorage("CoC", Format::R16Float, depth, 0);
        s.alu(4);
    });
    TextureRef dofDown, near, far, dof;
    addSynth(c, "DoF downsample", SynthKind::Compute, [&](Setup& s) {
        s.sample(hdr);
        s.sample(coc);
        dofDown = s.storage(s.texture("DoF half", c.hdr(), c.W() / 2, c.H() / 2));
        s.alu(8);
    });
    addSynth(c, "DoF gather", SynthKind::Compute, [&](Setup& s) {
        s.sample(dofDown);
        near = s.storage(s.texture("DoF near", c.hdr(), c.W() / 2, c.H() / 2));
        far  = s.storage(s.texture("DoF far", c.hdr(), c.W() / 2, c.H() / 2));
        s.alu(64);
    });
    addSynth(c, "DoF composite", SynthKind::Compute, [&](Setup& s) {
        s.sample(hdr);
        s.sample(coc);
        s.sample(near);
        s.sample(far);
        dof = s.storage(s.texture("DoF", c.hdr(), c.W(), c.H()));
        s.alu(8);
    });
    TextureRef tileMax, neighborMax, blurred;
    addSynth(c, "MB tile max", SynthKind::Compute, [&](Setup& s) {
        s.sample(velocity);
        tileMax = s.storage(s.texture("MB tile max", Format::RG16Float, (c.W() + 15) / 16, (c.H() + 15) / 16));
        s.alu(16);
    });
    addSynth(c, "MB neighbor max", SynthKind::Compute, [&](Setup& s) {
        s.sample(tileMax);
        neighborMax = s.storage(s.texture("MB neighbor max", Format::RG16Float, (c.W() + 15) / 16, (c.H() + 15) / 16));
        s.alu(4);
    });
    addSynth(c, "Motion blur", SynthKind::Compute, [&](Setup& s) {
        s.sample(dof);
        s.sample(velocity);
        s.sample(neighborMax);
        blurred = s.storage(s.texture("Motion blurred", c.hdr(), c.W(), c.H()));
        s.alu(64);
    });
    const TextureRef bloomed = bloom(c, blurred, 6);
    TextureRef flare, exposure;
    addSynth(c, "Lens flare", SynthKind::Compute, [&](Setup& s) {
        s.sample(bloomed);
        flare = s.storage(s.texture("Lens flare", c.hdr(), c.W() / 4, c.H() / 4));
        s.alu(16);
    });
    addSynth(c, "Exposure", SynthKind::Compute, [&](Setup& s) {
        s.sample(flare);
        exposure = s.storage(s.texture("Exposure", Format::R32Float, 16, 16));
        s.alu(32);
    });
    TextureRef ldr;
    addSynth(c, "Tonemap", SynthKind::Fullscreen, [&](Setup& s) {
        s.sample(blurred);
        s.sample(bloomed);
        s.sample(flare);
        s.sample(exposure);
        ldr = s.color(s.texture("LDR", Format::RGBA8Unorm, c.W(), c.H()), 0);
        s.alu(16);
    });
    addSynth(c, "Film grain", SynthKind::Fullscreen, [&](Setup& s) {
        ldr = s.color(ldr, 0, LoadIntent::Preserve);
        s.alu(8);
    });
    TextureRef aa;
    addSynth(c, "FXAA", SynthKind::Compute, [&](Setup& s) {
        s.sample(ldr);
        aa = s.storage(s.texture("AA", Format::RGBA8Unorm, c.W(), c.H()));
        s.alu(24);
    });
    addSynth(c, "UI", SynthKind::Fullscreen, [&](Setup& s) {
        aa = s.color(aa, 0, LoadIntent::Preserve);
        s.alu(4);
    });
    c.scenario.output = aa;
}

// 3: async compute: shadows, G-buffer + lighting (fusable), a bandwidth-bound
// GI update and an ALU-bound particle simulation that the lighting consumes.
void buildAsync(Ctx& c) {
    const std::vector<TextureRef> shadows = shadowCascades(c, 4, 500000);
    TextureRef depth, albedo, normal, material;
    addSynth(c, "G-buffer", SynthKind::Geometry, [&](Setup& s) {
        depth    = s.depth(s.texture("Depth", Format::Depth32Float, c.W(), c.H()));
        albedo   = s.color(s.texture("Albedo", Format::RGBA8Unorm, c.W(), c.H()), 0);
        normal   = s.color(s.texture("Normal", Format::RGB10A2Unorm, c.W(), c.H()), 1);
        material = s.color(s.texture("Material", Format::RGBA8Unorm, c.W(), c.H()), 2);
        s.geometry(600000, 1);
        s.alu(24);
        s.hints(HintGeometryHeavy);
    });
    const auto queueOf = [&](const std::string& pass) {
        if (!c.params.async) return Queue::AsyncCompute; // default: every candidate async
        const auto& list = *c.params.async;
        return std::find(list.begin(), list.end(), pass) != list.end() ? Queue::AsyncCompute : Queue::Graphics;
    };
    const TextureDesc atlasDesc{c.hdr(), 4096, 4096};
    const TextureRef atlas = c.graph.importTexture("Probe atlas" + c.suffix, atlasDesc, ImportContentsDefined);
    c.scenario.imports.push_back({atlas.resource, ScenarioImport::Role::Static, atlasDesc});
    TextureRef gi, particles;
    addSynth(c, "GI probe update", SynthKind::Compute, queueOf("GI probe update"), [&](Setup& s) {
        s.sample(atlas);
        gi = s.storage(s.texture("GI volume", c.hdr(), 1024, 1024));
        s.alu(4);
    });
    addSynth(c, "Particle simulation", SynthKind::Compute, queueOf("Particle simulation"), [&](Setup& s) {
        particles = s.storage(s.texture("Particles", Format::RGBA16Float, 512, 512));
        s.alu(4096);
    });
    TextureRef hdr;
    addSynth(c, "Lighting", SynthKind::Fullscreen, [&](Setup& s) {
        s.fetch(albedo, 0);
        s.fetch(normal, 1);
        s.fetch(material, 2);
        for (const TextureRef& m : shadows) s.sample(m);
        s.sample(gi);
        s.sample(particles);
        hdr = s.color(s.texture("HDR", c.hdr(), c.W(), c.H()), 4);
        s.alu(96);
        s.hints(HintFragmentHeavy);
    });
    c.scenario.output = tonemapAndUi(c, {hdr, depth});
}

struct ScenarioInfo {
    const char* name;
    void (*build)(Ctx&);
    std::vector<std::string> remat;
    std::vector<std::string> async;
};

const std::vector<ScenarioInfo>& scenarios() {
    static const std::vector<ScenarioInfo> list = {
        {"deferred", buildDeferred, {"Velocity"}, {}},
        {"forward-plus", buildForwardPlus, {"Velocity"}, {}},
        {"post-chain", buildPostChain, {"Velocity", "CoC"}, {}},
        {"async-compute", buildAsync, {}, {"GI probe update", "Particle simulation"}},
    };
    return list;
}

} // namespace

u32 scenarioCount() { return static_cast<u32>(scenarios().size()); }

const char* scenarioName(u32 index) { return index < scenarioCount() ? scenarios()[index].name : "unknown"; }

std::vector<std::string> scenarioRematCandidates(u32 index) {
    return index < scenarioCount() ? scenarios()[index].remat : std::vector<std::string>{};
}

std::vector<std::string> scenarioAsyncCandidates(u32 index) {
    return index < scenarioCount() ? scenarios()[index].async : std::vector<std::string>{};
}

std::string scenarioFamily(u32 index, const ScenarioParams& p) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "scenario:%s:%ux%u:s%u:w%g:v%u:%s", scenarioName(index), p.width, p.height,
                  p.shadowSize, static_cast<double>(p.work), p.views, p.wideHdr ? "hdr32" : "hdr16");
    return buf;
}

bool buildScenario(u32 index, const ScenarioParams& params, RenderGraph& graph, TextureRef drawable, Scenario& out,
                   const SynthExecFactory& factory, std::string* error) {
    out = Scenario{};
    if (index >= scenarioCount()) {
        if (error) *error = "unknown scenario " + std::to_string(index);
        return false;
    }
    if (params.async) {
        for (const std::string& a : *params.async) {
            const auto& candidates = scenarios()[index].async;
            if (std::find(candidates.begin(), candidates.end(), a) == candidates.end()) {
                if (error) *error = "scenario '" + std::string(scenarios()[index].name) + "' has no async-eligible pass '" + a + "'";
                return false;
            }
        }
    }
    for (const std::string& r : params.remat) {
        const auto& candidates = scenarios()[index].remat;
        if (std::find(candidates.begin(), candidates.end(), r) == candidates.end()) {
            if (error) *error = "scenario '" + std::string(scenarios()[index].name) + "' cannot rematerialise '" + r + "'";
            return false;
        }
    }
    if (params.views < 1 || params.views > 6) {
        if (error) *error = "scenario views must be 1..6";
        return false;
    }
    out.name = scenarios()[index].name;
    Ctx c{params, graph, out, factory, {}, 1};
    std::vector<TextureRef> outputs;
    for (u32 v = 0; v < params.views; ++v) {
        // Split screen / extra cameras: independent copies of the scenario
        // (own names, seeds and geometry), composited by the present pass.
        c.suffix       = v == 0 ? std::string() : " [v" + std::to_string(v) + "]";
        c.geometryBase = v * 1000;
        scenarios()[index].build(c);
        outputs.push_back(out.output);
    }
    c.suffix.clear();
    if (drawable.valid()) {
        addSynth(c, "Present", SynthKind::Fullscreen, [&](Setup& s) {
            for (const TextureRef& t : outputs) s.sample(t);
            s.color(drawable, 0);
        });
    }
    out.synth.resize(graph.passes().size());
    if (!graph.errors().empty()) {
        if (error) *error = graph.errors().front();
        return false;
    }
    return true;
}

SynthWork synthWork(const SynthPass& p) {
    SynthWork w;
    const double invocations = static_cast<double>(p.width) * p.height;
    w.invocations = invocations;
    u32 rematInputs = 0;
    for (const SynthInput& in : p.inputs) rematInputs += in.remat ? 1 : 0;
    const double steps = p.aluIterations + static_cast<double>(p.rematIterations) * rematInputs;
    w.intOps    = invocations * (steps * 4.0 + kSynthBaseOps);
    w.triangles = p.triangles;
    return w;
}

} // namespace phosphor::rg
