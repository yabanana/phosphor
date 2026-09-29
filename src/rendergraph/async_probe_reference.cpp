#include "rendergraph/async_probe_reference.h"

namespace phosphor::rg {

void expectedAsyncReadback(u32 frame, std::vector<u32>& out) {
    std::vector<u32> s(kAsyncSeedCount);
    for (u32 i = 0; i < kAsyncSeedCount; ++i) s[i] = seedValue(i, frame);
    out.resize(kAsyncResultCount);
    for (u32 j = 0; j < kAsyncResultCount; ++j) {
        out[j] = consumeValue(reduceGroup(&s[static_cast<size_t>(j) * kAsyncGroupSize]), j, frame);
    }
}

u64 countAsyncMismatches(u32 frame, const u32* data) {
    std::vector<u32> expected;
    expectedAsyncReadback(frame, expected);
    u64 bad = 0;
    for (u32 j = 0; j < kAsyncResultCount; ++j) bad += data[j] != expected[j] ? 1 : 0;
    return bad;
}

void addAsyncProbeChain(RenderGraph& graph, AsyncProbeRefs& refs, const AsyncProbeExec& exec) {
    addAsyncProbeProducers(graph, refs, exec);
    addAsyncProbeConsumer(graph, refs, exec);
}

void addAsyncProbeProducers(RenderGraph& graph, AsyncProbeRefs& refs, const AsyncProbeExec& exec) {
    refs.readback = graph.importBuffer("Async readback", {kAsyncReadbackSize}, ImportOutput | ImportPerFrame);

    graph.addPass(
        "Async seed", PassType::Compute, Queue::Graphics,
        [&](PassBuilder& b) {
            refs.s = b.write(b.createBuffer("Async S", {kAsyncSeedBytes}), Usage::ShaderWrite, StageDispatch);
        },
        exec.seed);
    graph.addPass(
        "Async reduce", PassType::Compute, Queue::AsyncCompute,
        [&](PassBuilder& b) {
            b.read(refs.s, Usage::ShaderRead, StageDispatch);
            refs.r = b.write(b.createBuffer("Async R", {kAsyncResultBytes}), Usage::ShaderWrite, StageDispatch);
        },
        exec.reduce);
}

void addAsyncProbeConsumer(RenderGraph& graph, AsyncProbeRefs& refs, const AsyncProbeExec& exec) {
    graph.addPass(
        "Async consume", PassType::Compute, Queue::Graphics,
        [&](PassBuilder& b) {
            b.read(refs.r, Usage::ShaderRead, StageDispatch);
            b.write(refs.readback, Usage::ShaderWrite, StageDispatch);
            b.setSideEffect();
        },
        exec.consume);
}

} // namespace phosphor::rg
