#pragma once
#include "platform/metal/metalfx_denoise.h"
#include <memory>
#include <string_view>
namespace phosphor {
class MetalfxDenoiseFixture {
public:
    struct Options {std::string scenario,outputDirectory;bool preExposedPolicy=false;u32 activeViews=1,prewarmTimeoutMs=120000;};
    MetalfxDenoiseFixture(MetalContext&,PipelineCache&,MetalfxDenoise::Factory,Options);
    ~MetalfxDenoiseFixture();
    static bool validScenario(std::string_view);
    void prepareFrame(const MetalfxDenoise::Frame&);
    bool initialPrewarmAttempted()const;
    rg::TextureRef addToGraph(rg::RenderGraph&);
    void bindFrame(MetalGraphExecutor&);
    bool consume(u32 slot);
    bool finish(); // caller waits GPU/drains slots; no unsupported fixture can pass
    bool ready()const;
    u64 version()const;
    std::string status()const;
    std::string reportJSON()const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
