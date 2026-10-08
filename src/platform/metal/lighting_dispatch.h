#pragma once
#include "platform/metal/metal_context.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/upload_ring.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace phosphor::lighting {
inline pipe::PipelineDesc kernel(const char* name, bool rt = false) {
    pipe::PipelineDesc d; d.kind = pipe::PipelineKind::Compute; d.label = name; d.functions = {name,"",""};
    if (rt) d.linkedFunctions = {"rt_alpha_generic"};
    return d;
}
inline MTL4::ArgumentTable* table(MetalContext& c, u32 buffers=24, u32 textures=16) {
    auto* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(buffers); d->setMaxTextureBindCount(textures);
    NS::Error* error = nullptr; auto* t = c.device()->newArgumentTable(d,&error); d->release();
    if (!t) throw std::runtime_error("Lighting argument table allocation failed");
    return t;
}
inline MTL::Buffer* buffer(MetalContext& c, u64 bytes, const char* name, bool shared=false) {
    auto* b = c.memory().newBuffer(std::max<u64>(bytes,16), shared ? MTL::ResourceStorageModeShared : MTL::ResourceStorageModePrivate,
                                   MemoryCategory::RayTracing, name);
    if (!b) throw std::runtime_error("Lighting buffer allocation failed"); return b;
}
inline MTL::Texture* texture(MetalContext& c, u32 w, u32 h, MTL::PixelFormat f, const char* name) {
    auto* d = MTL::TextureDescriptor::texture2DDescriptor(f,w,h,false);
    d->setStorageMode(MTL::StorageModePrivate); d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | MTL::TextureUsageRenderTarget);
    auto* t = c.memory().newTexture(d,MemoryCategory::RayTracing,name);
    if (!t) throw std::runtime_error("Lighting texture allocation failed"); return t;
}
template<class T> MTL::GPUAddress upload(MetalContext& c, const T& x) {
    auto slice = c.frameUploads().allocate(sizeof(T)); std::memcpy(slice.cpu,&x,sizeof(T)); return slice.gpu;
}
inline void dispatch(MTL4::ComputeCommandEncoder* e, PipelineCache& p, pipe::PipelineHandle h, MTL4::ArgumentTable* t, u32 x, u32 y=1) {
    auto* state = p.compute(h); if (!state || !x || !y) throw std::logic_error("Lighting dispatch missing pipeline/work");
    e->setComputePipelineState(state); e->setArgumentTable(t);
    e->dispatchThreads(MTL::Size::Make(x,y,1), MTL::Size::Make(std::min<u32>(8,x),std::min<u32>(8,y),1));
}
} // namespace phosphor::lighting
