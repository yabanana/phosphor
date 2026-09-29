#include "pipeline/pipeline_registry.h"

namespace phosphor::pipe {

// F3 contract stub: implemented by the F3.2/F3.5 work package.
float PipelineStats::archiveMissRate() const { return 0.0f; }
std::string formatPipelineStats(const PipelineStats&) { return "PIPELINES"; }
std::string pipelineStatsJson(const PipelineStats&) { return "{}"; }

PipelineRegistry::PipelineRegistry(u32) {}
PipelineRegistry::~PipelineRegistry() = default;
void PipelineRegistry::setReleaser(ReleaseFn release, void* context) {
    releaseFn_  = release;
    releaseCtx_ = context;
}
PipelineHandle PipelineRegistry::find(PipelineKey) const { return INVALID_PIPELINE; }
PipelineHandle PipelineRegistry::add(PipelineKey, const PipelineDesc&) { return INVALID_PIPELINE; }
const PipelineDesc& PipelineRegistry::desc(PipelineHandle h) const { return entries_[h].desc; }
PipelineKey PipelineRegistry::key(PipelineHandle h) const { return entries_[h].key; }
PipelineState PipelineRegistry::state(PipelineHandle h) const { return entries_[h].state; }
void* PipelineRegistry::get(PipelineHandle) const { return nullptr; }
bool PipelineRegistry::isFinal(PipelineHandle) const { return false; }
u32 PipelineRegistry::drain() { return 0; }
u32 PipelineRegistry::beginGeneration() { return generation_; }
void PipelineRegistry::recordRenderThreadCompile(float) {}
void PipelineRegistry::post(const Completion&) {}
void PipelineRegistry::release(void*) const {}
void PipelineRegistry::applyCurrent(Entry&, const Completion&, u32&) {}
void PipelineRegistry::commitOrAbandonReload(u32&) {}

} // namespace phosphor::pipe
