#pragma once
#include "platform/metal/shadow_passes.h"
#include "renderer/denoise_settings.h"
#include <memory>
#include <array>
namespace phosphor {
// SOURCE ONLY / NON VERIFIED. Separate physical history pairs per active
// view/signal; reusable by reflection composition or another lighting consumer.
class DenoisePasses {
public:
    DenoisePasses(MetalContext&,PipelineCache&,const LaunchOptions&);
    ~DenoisePasses();
    void prepareFrame(const ShadowPasses::Frame&,u64 signalEpoch,const std::array<u64,4>& currentRevisions);
    void invalidateAll(const char* reason);
    rg::TextureRef addSignal(rg::RenderGraph&,u32 signal,rg::TextureRef raw,rg::TextureRef motion,
                            rg::BufferRef surfaces,rg::BufferRef specularMetadata={});
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool check(u32 slot) const;
    [[nodiscard]] bool ready() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
