#include "platform/metal/known_cost_pass.h"
#include "platform/metal/pipeline_cache.h"
#include "rendergraph/pass_context.h"

#include <stdexcept>

namespace phosphor {

namespace {
// Argument table slots; must match shaders/known_cost.metal.
constexpr NS::UInteger kBindIterations = 0;
constexpr NS::UInteger kBindOut        = 1;
} // namespace

KnownCostPass::KnownCostPass(MetalContext& context, PipelineCache& pipelines, u32 iterations)
    : context_(context), pipelines_(pipelines), iterations_(iterations) {
    pipe::PipelineDesc desc;
    desc.kind         = pipe::PipelineKind::Compute;
    desc.label        = "known_cost";
    desc.functions[0] = "known_cost";
    pipeline_ = pipelines_.request(desc);

    MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(2);
    d->setLabel(NS::String::string("Known cost arguments", NS::UTF8StringEncoding));
    NS::Error* error = nullptr;
    args_ = context_.device()->newArgumentTable(d, &error);
    d->release();
    if (!args_) throw std::runtime_error("Failed to create the known-cost argument table");
}

KnownCostPass::~KnownCostPass() {
    context_.waitIdle();
    args_->release();
}

void KnownCostPass::addToGraph(rg::RenderGraph& graph) {
    graph.addPass(
        "Known cost", rg::PassType::Compute,
        [&](rg::PassBuilder& b) {
            out_ = b.createBuffer("Known cost output", {static_cast<u64>(kThreads) * sizeof(u32)});
            out_ = b.write(out_, rg::Usage::ShaderWrite, rg::StageDispatch);
            b.setSideEffect();
            b.setProfileShaders("known_cost");
        },
        [this](rg::PassContext& ctx) {
            auto* enc = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
            const UploadRing::Slice constants = context_.frameUploads().allocate(sizeof(u32));
            *reinterpret_cast<u32*>(constants.cpu) = iterations_;
            args_->setAddress(constants.gpu, kBindIterations);
            args_->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(out_))->gpuAddress(), kBindOut);
            enc->setComputePipelineState(pipelines_.compute(pipeline_));
            enc->setArgumentTable(args_);
            enc->dispatchThreads(MTL::Size::Make(kThreads, 1, 1), MTL::Size::Make(256, 1, 1));
        });
}

} // namespace phosphor
