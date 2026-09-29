#include "pipeline/compile_queue.h"

namespace phosphor::pipe {

// F3 contract stub: implemented by the F3.1 work package.
CompileQueue::CompileQueue(u32, ThreadInit threadInit) : threadInit_(std::move(threadInit)) {}
CompileQueue::~CompileQueue() = default;
void CompileQueue::submit(CompilePriority, u32, Job) {}
u32 CompileQueue::cancelBefore(u32) { return 0; }
void CompileQueue::waitIdle() {}
size_t CompileQueue::outstanding() const { return 0; }
u64 CompileQueue::completed() const { return 0; }
void CompileQueue::workerMain(u32) {}

} // namespace phosphor::pipe
