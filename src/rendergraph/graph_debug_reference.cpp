#include "rendergraph/graph_debug_reference.h"

#include "rendergraph/aliasing.h"

namespace phosphor::rg {

void expectedReadback(u32 frame, std::vector<u32>& out) {
    out.resize(static_cast<size_t>(kDebugSize) * kDebugSize);
    for (u32 y = 0; y < kDebugSize; ++y) {
        u32 rowSum = 0;
        for (u32 x = 0; x < kDebugSize; ++x) rowSum += fillValue(x, y, frame);
        for (u32 x = 0; x < kDebugSize; ++x) {
            out[static_cast<size_t>(y) * kDebugSize + x] = rasterValue(expandValue(rowSum, x, y)) ^ kChecksumMask;
        }
    }
}

u64 countMismatches(u32 frame, const u32* data) {
    u64 bad = 0;
    for (u32 y = 0; y < kDebugSize; ++y) {
        u32 rowSum = 0;
        for (u32 x = 0; x < kDebugSize; ++x) rowSum += fillValue(x, y, frame);
        for (u32 x = 0; x < kDebugSize; ++x) {
            const u32 expected = rasterValue(expandValue(rowSum, x, y)) ^ kChecksumMask;
            bad += data[static_cast<size_t>(y) * kDebugSize + x] != expected ? 1 : 0;
        }
    }
    return bad;
}

void addDebugChain(RenderGraph& graph, DebugChainRefs& refs, const DebugChainExec& exec) {
    const TextureDesc image{Format::R32Uint, kDebugSize, kDebugSize};
    refs.readback = graph.importBuffer("Debug readback", {kDebugReadbackSize}, ImportOutput | ImportPerFrame);

    graph.addPass(
        "Debug fill A", PassType::Compute,
        [&](PassBuilder& b) {
            refs.a = b.write(b.createTexture("Debug A", image), Usage::ShaderWrite, StageDispatch);
        },
        exec.fill);
    graph.addPass(
        "Debug reduce B", PassType::Compute,
        [&](PassBuilder& b) {
            b.read(refs.a, Usage::ShaderRead, StageDispatch);
            refs.b = b.write(b.createBuffer("Debug B", {kDebugSize * sizeof(u32)}), Usage::ShaderWrite,
                             StageDispatch);
        },
        exec.reduce);
    graph.addPass(
        "Debug expand C", PassType::Compute,
        [&](PassBuilder& b) {
            b.read(refs.b, Usage::ShaderRead, StageDispatch);
            refs.c = b.write(b.createTexture("Debug C", image), Usage::ShaderWrite, StageDispatch);
        },
        exec.expand);
    graph.addPass(
        "Debug raster D", PassType::Raster,
        [&](PassBuilder& b) {
            b.read(refs.c, Usage::ShaderRead, StageFragment);
            refs.d = b.writeColor(b.createTexture("Debug D", image), 0, LoadIntent::Clear);
        },
        exec.raster);
    graph.addPass(
        "Debug checksum", PassType::Compute,
        [&](PassBuilder& b) {
            b.read(refs.d, Usage::ShaderRead, StageDispatch);
            b.write(refs.readback, Usage::ShaderWrite, StageDispatch);
            b.setSideEffect();
        },
        exec.checksum);
}

DebugAliasSummary summarizeAliasing(const CompiledGraph& compiled) {
    DebugAliasSummary s;
    const auto& p = compiled.aliasing.placements;
    for (const Placement& pl : p) s.aliasedFlags += pl.aliased ? 1 : 0;
    for (size_t i = 0; i < p.size(); ++i) {
        for (size_t j = i + 1; j < p.size(); ++j) {
            if (rangesIntersect(p[i], p[j])) s.sharedPairs.emplace_back(p[i].resource, p[j].resource);
        }
    }
    s.heapSize      = compiled.aliasing.heapSize;
    s.unaliasedSize = compiled.aliasing.unaliasedSize;
    return s;
}

} // namespace phosphor::rg
