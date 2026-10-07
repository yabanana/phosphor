#include "platform/metal/temporal_worker.h"
#include <MetalFX/MetalFX.hpp>
#include "platform/metal/pipeline_cache.h"
#ifdef PHOSPHOR_METALFX_DENOISED_GATEWAY_AVAILABLE
#include <MetalFX/MTL4FXTemporalDenoisedScaler.hpp>
#endif

#include "core/log.h"
#include "core/profile.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/metalfx_lifetime.h"

#include <pthread/qos.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>

namespace phosphor {

namespace {

using Clock = std::chrono::steady_clock;

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

float msSince(Clock::time_point start) {
    return std::chrono::duration<float, std::milli>(Clock::now() - start).count();
}

/// Metal's error text on one line (archive misses carry multi-line keys):
/// every log line stays a single line for the scripts that parse the log.
std::string reason(NS::Error* error) {
    std::string text = error ? error->localizedDescription()->utf8String() : "unknown error";
    for (char& c : text) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    const size_t end = text.find_last_not_of(' ');
    text.erase(end == std::string::npos ? 0 : end + 1);
    return text;
}

MTL::DataType dataType(pipe::ConstantType type) {
    switch (type) {
    case pipe::ConstantType::Bool:  return MTL::DataTypeBool;
    case pipe::ConstantType::UInt:  return MTL::DataTypeUInt;
    case pipe::ConstantType::Int:   return MTL::DataTypeInt;
    case pipe::ConstantType::Float: return MTL::DataTypeFloat;
    }
    return MTL::DataTypeUInt;
}

MTL::ColorWriteMask writeMask(u8 mask) {
    MTL::ColorWriteMask out = MTL::ColorWriteMaskNone;
    if (mask & 1u) out |= MTL::ColorWriteMaskRed;
    if (mask & 2u) out |= MTL::ColorWriteMaskGreen;
    if (mask & 4u) out |= MTL::ColorWriteMaskBlue;
    if (mask & 8u) out |= MTL::ColorWriteMaskAlpha;
    return out;
}

/// +1 native descriptor for a function with no specialization requirements.
MTL4::LibraryFunctionDescriptor* libraryFunctionDescriptor(const std::string& name, MTL::Library* library) {
    auto* fn = MTL4::LibraryFunctionDescriptor::alloc()->init();
    fn->setLibrary(library);
    fn->setName(str(name.c_str()));
    return fn;
}

/// +1 function descriptor for `name`, specialised with the desc's constants.
/// Main pipeline functions retain an empty specialization for generic variants:
/// declared-but-undefined function constants select the shader's runtime path.
MTL4::FunctionDescriptor* functionDescriptor(const std::string& name, const pipe::PipelineDesc& desc,
                                             MTL::Library* library) {
    auto* fn = libraryFunctionDescriptor(name, library);

    MTL::FunctionConstantValues* values = MTL::FunctionConstantValues::alloc()->init();
    for (u32 i = 0; i < desc.constantCount; ++i) {
        const pipe::FunctionConstant& c = desc.constants[i];
        if (c.type == pipe::ConstantType::Bool) {
            const bool b = c.bits != 0;
            values->setConstantValue(&b, MTL::DataTypeBool, c.index);
        } else {
            values->setConstantValue(&c.bits, dataType(c.type), c.index);
        }
    }
    MTL4::SpecializedFunctionDescriptor* spec = MTL4::SpecializedFunctionDescriptor::alloc()->init();
    spec->setFunctionDescriptor(fn);
    spec->setConstantValues(values);
    values->release();
    fn->release();
    return spec;
}

/// Metal's default blend substate, spelled out: a specialisation descriptor
/// otherwise inherits the flexible pipeline's "Unspecialized" substate, and
/// validation warns that it is ignored when blending is disabled.
void setDefaultBlendSubstate(MTL4::RenderPipelineColorAttachmentDescriptor* color) {
    color->setRgbBlendOperation(MTL::BlendOperationAdd);
    color->setAlphaBlendOperation(MTL::BlendOperationAdd);
    color->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    color->setDestinationRGBBlendFactor(MTL::BlendFactorZero);
    color->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    color->setDestinationAlphaBlendFactor(MTL::BlendFactorZero);
}

constexpr u32 kMetalColorAttachments = 8;

/// Every attachment's output state spelled out (measured: a fresh MTL4
/// descriptor reports an "Unspecialized" blend substate, which validation
/// flags on flexible pipelines and their specialisations), then the desc's.
void setColorOutputs(MTL4::RenderPipelineColorAttachmentDescriptorArray* attachments, const pipe::PipelineDesc& desc) {
    for (u32 i = 0; i < kMetalColorAttachments; ++i) {
        MTL4::RenderPipelineColorAttachmentDescriptor* color = attachments->object(i);
        color->setPixelFormat(MTL::PixelFormatInvalid);
        color->setBlendingState(MTL4::BlendStateDisabled);
        color->setWriteMask(MTL::ColorWriteMaskAll);
        setDefaultBlendSubstate(color);
    }
    for (u32 i = 0; i < desc.colorCount; ++i) {
        const pipe::ColorOutput& out = desc.color[i];
        if (out.format == rg::Format::Unknown && !out.unspecialized) continue;
        MTL4::RenderPipelineColorAttachmentDescriptor* color = attachments->object(i);
        if (out.unspecialized) {
            // Metal 4 flexible pipeline: output state chosen at specialisation.
            color->setPixelFormat(MTL::PixelFormatUnspecialized);
            color->setBlendingState(MTL4::BlendStateUnspecialized);
            color->setWriteMask(MTL::ColorWriteMaskUnspecialized);
            continue;
        }
        color->setPixelFormat(toMetalFormat(out.format));
        if (out.writeMask != 0xF) color->setWriteMask(writeMask(out.writeMask));
        if (out.blend == pipe::ColorOutput::Blend::AlphaOver) {
            color->setBlendingState(MTL4::BlendStateEnabled);
            color->setRgbBlendOperation(MTL::BlendOperationAdd);
            color->setAlphaBlendOperation(MTL::BlendOperationAdd);
            color->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
            color->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
            color->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
            color->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        }
    }
}

/// +1 MTL4 pipeline descriptor for `desc`.
MTL4::PipelineDescriptor* buildDescriptor(const pipe::PipelineDesc& desc, MTL::Library* library) {
    if (desc.kind == pipe::PipelineKind::Tile) {
        auto *d = MTL4::TileRenderPipelineDescriptor::alloc()->init();
        if (!desc.label.empty())
            d->setLabel(str(desc.label.c_str()));
        auto *fn = functionDescriptor(desc.functions[0], desc, library);
        d->setTileFunctionDescriptor(fn);
        fn->release();
        d->setThreadgroupSizeMatchesTileSize(true);
        for (u32 i = 0; i < desc.colorCount; ++i)
            d->colorAttachments()->object(i)->setPixelFormat(toMetalFormat(desc.color[i].format));
        return d;
    }
    if (desc.kind == pipe::PipelineKind::Compute) {
        MTL4::ComputePipelineDescriptor* d = MTL4::ComputePipelineDescriptor::alloc()->init();
        if (!desc.label.empty()) d->setLabel(str(desc.label.c_str()));
        MTL4::FunctionDescriptor* fn = functionDescriptor(desc.functions[0], desc, library);
        d->setComputeFunctionDescriptor(fn);
        fn->release();
        if (!desc.linkedFunctions.empty()) {
            std::vector<NS::Object*> functions;
            functions.reserve(desc.linkedFunctions.size());
            // The generic RT intersection function has no function constants.
            // Preserve its native library descriptor instead of manufacturing
            // an empty specialization; explicit constants still specialize.
            for (const auto& name : desc.linkedFunctions)
                functions.push_back(desc.constantCount ? functionDescriptor(name, desc, library)
                                                       : libraryFunctionDescriptor(name, library));
            auto* link = MTL4::StaticLinkingDescriptor::alloc()->init();
            link->setFunctionDescriptors(NS::Array::array(functions.data(), functions.size()));
            d->setStaticLinkingDescriptor(link);
            link->release();
            for (auto* function : functions) function->release();
        }
        return d;
    }
    if (desc.kind == pipe::PipelineKind::Mesh) {
        // F6.3: [optional object] + mesh + fragment; the limits are part of
        // the key (pipe::MeshPipelineLimits).
        MTL4::MeshRenderPipelineDescriptor* d = MTL4::MeshRenderPipelineDescriptor::alloc()->init();
        if (!desc.label.empty()) d->setLabel(str(desc.label.c_str()));
        if (!desc.functions[0].empty()) {
            MTL4::FunctionDescriptor* os = functionDescriptor(desc.functions[0], desc, library);
            d->setObjectFunctionDescriptor(os);
            os->release();
            d->setMaxTotalThreadsPerObjectThreadgroup(desc.mesh.objectThreads);
            d->setPayloadMemoryLength(desc.mesh.payloadBytes);
            d->setMaxTotalThreadgroupsPerMeshGrid(desc.mesh.meshGroups);
        }
        MTL4::FunctionDescriptor* ms = functionDescriptor(desc.functions[1], desc, library);
        MTL4::FunctionDescriptor* fs = functionDescriptor(desc.functions[2], desc, library);
        d->setMeshFunctionDescriptor(ms);
        d->setFragmentFunctionDescriptor(fs);
        ms->release();
        fs->release();
        d->setMaxTotalThreadsPerMeshThreadgroup(desc.mesh.meshThreads);
        setColorOutputs(d->colorAttachments(), desc);
        return d;
    }
    MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
    if (!desc.label.empty()) d->setLabel(str(desc.label.c_str()));
    MTL4::FunctionDescriptor* vs = functionDescriptor(desc.functions[0], desc, library);
    MTL4::FunctionDescriptor* fs = functionDescriptor(desc.functions[1], desc, library);
    d->setVertexFunctionDescriptor(vs);
    d->setFragmentFunctionDescriptor(fs);
    vs->release();
    fs->release();
    setColorOutputs(d->colorAttachments(), desc);
    if (desc.indirectCommandBuffers) d->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    // MTL4 render pipelines take no depth format: it is inferred from the pass.
    return d;
}

void releaseDeferred(void* cache, void* object) {
    static_cast<MetalContext*>(cache)->deferRelease(static_cast<NS::Object*>(object));
}

void releaseNow(void*, void* object) {
    static_cast<NS::Object*>(object)->release();
}

void* toOpaque(NS::Object* object) { return static_cast<void*>(object); }

/// Keeps a library alive while a job referencing it exists: queued jobs can
/// be dropped without running (cancelBefore, shutdown), so the release must
/// not depend on the job running.
std::shared_ptr<void> retainLibrary(MTL::Library* library) {
    library->retain();
    return std::shared_ptr<void>(nullptr, [library](void*) { library->release(); });
}

} // namespace

PipelineCache::PipelineCache(MetalContext& context, const Options& options)
    : context_(context), options_(options) {
    MTL::Device* device = context_.device();
    NS::Error* error = nullptr;

    MTL4::CompilerDescriptor* compilerDesc = MTL4::CompilerDescriptor::alloc()->init();
    compilerDesc->setLabel(str("Phosphor pipeline compiler"));
    if (!options_.harvestPath.empty()) {
        // F3.4: record every descriptor compiled from now on.
        MTL4::PipelineDataSetSerializerDescriptor* sd = MTL4::PipelineDataSetSerializerDescriptor::alloc()->init();
        sd->setConfiguration(MTL4::PipelineDataSetSerializerConfigurationCaptureDescriptors);
        serializer_ = device->newPipelineDataSetSerializer(sd);
        sd->release();
        if (!serializer_) throw std::runtime_error("Failed to create the pipeline data set serializer");
        compilerDesc->setPipelineDataSetSerializer(serializer_);
    }
    compiler_ = device->newCompiler(compilerDesc, &error);
    compilerDesc->release();
    if (!compiler_) throw std::runtime_error(std::string("Failed to create MTL4Compiler: ") + reason(error).c_str());

    library_ = context_.library();
    library_->retain();

    if (serializer_) {
        archiveStatus_ = "ignored (harvesting)";
    } else if (options_.archivePath.empty()) {
        archiveStatus_ = "none";
    } else if (const char* sv = std::getenv("MTL_SHADER_VALIDATION"); sv && std::strcmp(sv, "0") != 0) {
        // Measured on macOS 27.2: every lookup fails with "MTL4Archive
        // instances are not compatible with Metal shader validation".  Report
        // it as an unavailable archive instead of a stream of misses.
        archiveStatus_ = "disabled (incompatible with MTL_SHADER_VALIDATION)";
        LOG_INFO("Pipeline archive %s", archiveStatus_.c_str());
    } else {
        error = nullptr;
        archive_ = device->newArchive(NS::URL::fileURLWithPath(str(options_.archivePath.c_str())), &error);
        if (archive_) {
            archiveStatus_ = "loaded " + options_.archivePath;
        } else {
            archiveStatus_ = "unavailable (" + options_.archivePath + ": " + reason(error).c_str() + ")";
            LOG_WARN("Pipeline archive %s", archiveStatus_.c_str());
        }
    }

    registry_.setReleaser(releaseDeferred, &context_);

    const u32 workers = std::max<u32>(1, static_cast<u32>(device->maximumConcurrentCompilationTaskCount()));
    const bool interactive = options_.interactiveQos;
    queue_ = std::make_unique<pipe::CompileQueue>(workers, [interactive](u32 worker) {
        // F3.1: the compiler inherits the calling thread's QoS; keep it below
        // the render thread's (user-interactive).
        pthread_set_qos_class_self_np(interactive ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_UTILITY, 0);
        if (worker == 0) {
            LOG_INFO("Pipeline compile threads: QoS class 0x%x (%s)", static_cast<unsigned>(qos_class_self()),
                     qos_class_self() == QOS_CLASS_UTILITY ? "utility" : "not utility");
        }
    });
    LOG_INFO("Pipeline cache: %u compile threads (maximumConcurrentCompilationTaskCount), archive %s%s", workers,
             archiveStatus_.c_str(), options_.sync ? ", synchronous (--pipeline-sync)" : "");
}

PipelineCache::~PipelineCache() {
    queue_.reset(); // joins the workers; nothing posts after this
    context_.waitIdle();
    registry_.setReleaser(releaseNow, nullptr);
    registry_.drain();
    for (auto& [key, base] : flexible_) {
        if (base.pipeline) base.pipeline->release();
    }
    if (pendingLibrary_) pendingLibrary_->release();
    library_->release();
    if (archive_) archive_->release();
    compiler_->release();
    if (serializer_) serializer_->release();
    // registry_ releases what it still holds (releaseNow) when destroyed.
}

std::future<std::shared_ptr<MTL4FX::TemporalScaler>>
PipelineCache::requestTemporalScaler(MTLFX::TemporalScalerDescriptor *descriptor) {
    auto promise = std::make_shared<std::promise<std::shared_ptr<MTL4FX::TemporalScaler>>>();
    auto future = promise->get_future();
    auto copy = std::shared_ptr<MTLFX::TemporalScalerDescriptor>(descriptor->copy(), [](auto *p) { p->release(); });
    queue_->submit(pipe::CompilePriority::Urgent, ~0u, [this, promise, copy] {
        auto *pool = NS::AutoreleasePool::alloc()->init();
        MTL4FX::TemporalScaler *scaler = nullptr;
        try {
            std::lock_guard sdkFactoryLock(sdkFactoryMutex_);
            scaler = copy->newTemporalScaler(context_.device(), compiler_);
        } catch (...) {
            pool->release();
            promise->set_exception(std::current_exception());
            return;
        }
        // Drain before adopting: the ownership record counts every reference
        // left on the new scaler besides ours.
        pool->release();
        promise->set_value(metalfx::adoptTemporalScaler(scaler));
    });
    return future;
}

void PipelineCache::retireTemporalScaler(std::shared_ptr<MTL4FX::TemporalScaler> scaler) {
    if (!scaler)
        return;
    queue_->submit(pipe::CompilePriority::Prewarm, ~0u, [scaler = std::move(scaler)]() mutable {
        auto *pool = NS::AutoreleasePool::alloc()->init();
        scaler.reset();
        pool->release();
    });
}

#ifdef PHOSPHOR_METALFX_DENOISED_GATEWAY_AVAILABLE
std::future<std::shared_ptr<MTL4FX::TemporalDenoisedScaler>>
PipelineCache::requestTemporalDenoisedScaler(MTLFX::TemporalDenoisedScalerDescriptor* descriptor) {
    auto promise=std::make_shared<std::promise<std::shared_ptr<MTL4FX::TemporalDenoisedScaler>>>();
    auto future=promise->get_future();
    // Copy synchronously: the caller releases its descriptor after this returns.
    auto copy=std::shared_ptr<MTLFX::TemporalDenoisedScalerDescriptor>(descriptor->copy(),[](auto* p){p->release();});
    queue_->submit(pipe::CompilePriority::Urgent,~0u,[this,promise,copy]{
        auto* pool=NS::AutoreleasePool::alloc()->init();
        MTL4FX::TemporalDenoisedScaler* raw=nullptr;
        try {std::lock_guard sdkFactoryLock(sdkFactoryMutex_);raw=copy->newTemporalDenoisedScaler(context_.device(),compiler_);}
        catch(...) {pool->release();promise->set_exception(std::current_exception());return;}
        pool->release();
        // Ordinary ownership ONLY. DENOISED lifetime has not been measured;
        // never use metalfx::adoptTemporalScaler or private cycle logic.
        try {
            promise->set_value(std::shared_ptr<MTL4FX::TemporalDenoisedScaler>(raw,[](auto* p){if(p)p->release();}));
        } catch(...) {promise->set_exception(std::current_exception());}
    });
    return future;
}
void PipelineCache::retireTemporalDenoisedScaler(std::shared_ptr<MTL4FX::TemporalDenoisedScaler> scaler) {
    if(!scaler)return;
    // Adapter deferred this until all GPU consumers completed.
    queue_->submit(pipe::CompilePriority::Prewarm,~0u,[scaler=std::move(scaler)]()mutable{
        auto* pool=NS::AutoreleasePool::alloc()->init();scaler.reset();pool->release();
    });
}
#endif

std::future<std::shared_ptr<TemporalWorker>>
PipelineCache::requestTemporalWorker(std::shared_ptr<TemporalWorker> worker) {
    auto promise = std::make_shared<std::promise<std::shared_ptr<TemporalWorker>>>();
    auto future = promise->get_future();
    queue_->submit(pipe::CompilePriority::Urgent, ~0u, [promise, worker] {
        auto *pool = NS::AutoreleasePool::alloc()->init();
        try {
            worker->start();
            promise->set_value(worker);
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
        pool->release();
    });
    return future;
}

u32 PipelineCache::workerCount() const { return queue_ ? queue_->workerCount() : 0; }

pipe::PipelineHandle PipelineCache::request(const pipe::PipelineDesc& desc, bool loading) {
    const Clock::time_point requestStart = Clock::now();
    struct AddTime {
        double& total;
        Clock::time_point start;
        ~AddTime() { total += std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
    } addTime{requestMs_, requestStart};
    const pipe::PipelineKey key = pipe::pipelineKey(desc);
    pipe::PipelineHandle handle = registry_.find(key);
    if (handle != pipe::INVALID_PIPELINE) return handle;

    handle = registry_.add(key, desc);
    LOG_INFO("Pipeline requested: %s (key %016llx)", desc.label.c_str(), static_cast<unsigned long long>(key));
    const u32 generation = registry_.pendingGeneration();
    MTL::Library* library = registry_.reloadPending() ? pendingLibrary_ : library_;

    if (options_.sync) {
        // Negative control: the render thread resolves the request itself.
        const u32 callsBefore = registry_.stats().compilerCalls;
        const Clock::time_point start = Clock::now();
        resolve(handle, desc, generation, library, /*allowFallback*/ false);
        drain();
        const float ms = msSince(start);
        if (startupDone_ && !registry_.isFinal(handle)) LOG_ERROR("Pipeline %s failed", desc.label.c_str());
        // Only a direct archive hit avoids a compiler API invocation.
        if (startupDone_ && registry_.stats().compilerCalls != callsBefore) registry_.recordRenderThreadCompile(ms);
        return handle;
    }
    // A flexible fallback only when nothing equivalent is ready: a variant
    // whose generic pipeline (same output state) is final is drawn with that
    // generic meanwhile (the renderer's choice, zero compiles), and startup
    // requests are waited for anyway.  Every specialisation of a flexible
    // pipeline makes the validation layer print "blending substate ... is
    // ignored" for each attachment with blending disabled (measured with a
    // minimal repro, whatever the base configuration), so the path is used
    // only where it buys something.  --debug-pipeline-fallback forces it.
    bool allowFallback = options_.fallbackOnly;
    if (!allowFallback && startupDone_ && !loading && desc.kind == pipe::PipelineKind::Render) {
        const pipe::PipelineHandle generic = registry_.find(pipe::pipelineKey(desc.generic()));
        allowFallback = generic == pipe::INVALID_PIPELINE || !registry_.isFinal(generic);
    }
    submit(handle, desc, generation, library, allowFallback);
    return handle;
}

void PipelineCache::submit(pipe::PipelineHandle handle, const pipe::PipelineDesc& desc, u32 generation,
                           MTL::Library* library, bool allowFallback) {
    queue_->submit(pipe::CompilePriority::Urgent, generation,
                   [this, handle, desc, generation, library, allowFallback, ref = retainLibrary(library)] {
                       PH_ZONE("Pipeline resolve");
                       resolve(handle, desc, generation, library, allowFallback);
                   });
}

void PipelineCache::resolve(pipe::PipelineHandle handle, const pipe::PipelineDesc& desc, u32 generation,
                            MTL::Library* library, bool allowFallback) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    pipe::Completion done;
    done.handle     = handle;
    done.generation = generation;
    done.archive    = archive_ ? pipe::ArchiveOutcome::Miss : pipe::ArchiveOutcome::Unavailable;
    const bool linkedCompute = desc.kind == pipe::PipelineKind::Compute && !desc.linkedFunctions.empty();
    // The direct Archive static-link path retained a linked function in the
    // F9 lifetime checks. Route only linked compute pipelines through Compiler;
    // lookupArchives is an optional cache hint, whose hit state is not exposed.
    // Report the actual compiler API call below, never an inferred archive hit.
    if (serializer_ || linkedCompute) done.archive = pipe::ArchiveOutcome::NotTried;

    // 1. Archive (F3.4): no compiler API invocation on a direct hit.
    if (archive_ && !options_.fallbackOnly && !linkedCompute) {
        NS::Error* error = nullptr;
        if (NS::Object* object = archiveLookup(desc, library, &error)) {
            done.object  = toOpaque(object);
            done.archive = pipe::ArchiveOutcome::Hit;
            complete(done);
            pool->release();
            return;
        }
        LOG_INFO("Pipeline archive miss: %s (%s)", desc.label.c_str(), reason(error).c_str());
    }

    // 2. Fallback (F3.2): the flexible generic pipeline specialised to this
    //    output state, usable while the full variant compiles.
    const bool render = desc.kind == pipe::PipelineKind::Render;
    if (allowFallback && render && !desc.isFlexible()) {
        u32 calls = 0;
        float ms = 0.0f;
        const pipe::PipelineDesc generic = desc.generic();
        MTL::RenderPipelineState* base = flexibleBase(generic, library, calls, ms);
        if (base) {
            MTL4::PipelineDescriptor* d = buildDescriptor(generic, library);
            NS::Error* error = nullptr;
            const Clock::time_point start = Clock::now();
            MTL::RenderPipelineState* fallback = nullptr;
            {
                PH_ZONE("Pipeline fallback specialise");
                fallback = compiler_->newRenderPipelineStateBySpecialization(d, base, &error);
            }
            ms += msSince(start);
            ++calls;
            d->release();
            if (!fallback) LOG_WARN("Specialising the flexible pipeline of %s failed: %s", desc.label.c_str(), reason(error).c_str());
            pipe::Completion fb;
            fb.handle        = handle;
            fb.generation    = generation;
            fb.object        = toOpaque(fallback);
            fb.fallback      = true;
            fb.compilerCalls = calls;
            fb.compileMs     = ms;
            if (fallback) complete(fb);
        }
        if (options_.fallbackOnly && base) {
            pool->release();
            return; // debug: the final object never replaces the fallback
        }
        // The final compile is background work: let other entries' fallbacks go first.
        if (queue_ && !options_.sync) {
            queue_->submit(pipe::CompilePriority::Specialize, generation,
                           [this, done, desc, library, ref = retainLibrary(library)] {
                NS::AutoreleasePool* inner = NS::AutoreleasePool::alloc()->init();
                pipe::Completion final = done;
                NS::Error* error = nullptr;
                const Clock::time_point start = Clock::now();
                final.object        = toOpaque(compileFinal(desc, library, &error));
                final.compileMs     = msSince(start);
                final.compilerCalls = 1;
                if (!final.object) LOG_ERROR("Pipeline %s failed: %s", desc.label.c_str(), reason(error).c_str());
                complete(final);
                inner->release();
            });
            pool->release();
            return;
        }
    }

    // 3. Full compile (compute pipelines, sync mode, no fallback possible).
    NS::Error* error = nullptr;
    const Clock::time_point start = Clock::now();
    done.object        = toOpaque(compileFinal(desc, library, &error));
    done.compileMs     = msSince(start);
    done.compilerCalls = 1;
    if (!done.object) LOG_ERROR("Pipeline %s failed: %s", desc.label.c_str(), reason(error).c_str());
    complete(done);
    pool->release();
}

NS::Object* PipelineCache::compileFinal(const pipe::PipelineDesc& desc, MTL::Library* library, NS::Error** error) {
    PH_ZONE("Pipeline compile");
    PH_ZONE_TEXT(desc.label.c_str(), desc.label.size());
    auto d = NS::TransferPtr(buildDescriptor(desc, library));
    NS::Object* object = nullptr;
    if (desc.kind == pipe::PipelineKind::Compute) {
        NS::SharedPtr<MTL4::CompilerTaskOptions> task;
        if (!desc.linkedFunctions.empty() && archive_ && !serializer_) {
            task = NS::TransferPtr(MTL4::CompilerTaskOptions::alloc()->init());
            if (!task) throw std::runtime_error("Failed to create compiler archive-lookup options");
            // Public compiler cache hint. The task copies the array; its
            // autoreleased reference is drained by resolve()'s autorelease pool.
            task->setLookupArchives(NS::Array::array(archive_));
        }
        object = compiler_->newComputePipelineState(static_cast<MTL4::ComputePipelineDescriptor*>(d.get()), task.get(), error);
    } else {
        object = compiler_->newRenderPipelineState(d.get(), nullptr, error);
    }
    return object;
}

NS::Object* PipelineCache::archiveLookup(const pipe::PipelineDesc& desc, MTL::Library* library, NS::Error** error) {
    MTL4::PipelineDescriptor* d = buildDescriptor(desc, library);
    NS::Object* object = nullptr;
    if (desc.kind == pipe::PipelineKind::Compute) {
        object = archive_->newComputePipelineState(static_cast<MTL4::ComputePipelineDescriptor*>(d), error);
    } else {
        object = archive_->newRenderPipelineState(d, error);
    }
    d->release();
    return object;
}

MTL::RenderPipelineState* PipelineCache::flexibleBase(const pipe::PipelineDesc& generic, MTL::Library* library,
                                                      u32& compilerCalls, float& compileMs) {
    const pipe::PipelineDesc flexibleDesc = generic.flexible();
    const u64 key = pipe::pipelineKey(flexibleDesc) ^ (reinterpret_cast<uintptr_t>(library) * 0x9E3779B97F4A7C15ull);
    std::once_flag* once = nullptr;
    {
        std::lock_guard<std::mutex> lock(flexibleMutex_);
        auto& slot = flexibleOnce_[key];
        if (!slot) slot = std::make_unique<std::once_flag>();
        once = slot.get();
    }
    std::call_once(*once, [&] {
        NS::Error* error = nullptr;
        NS::Object* object = nullptr;
        if (archive_ && !options_.fallbackOnly) object = archiveLookup(flexibleDesc, library, &error);
        if (!object) {
            const Clock::time_point start = Clock::now();
            object = compileFinal(flexibleDesc, library, &error);
            compileMs += msSince(start);
            ++compilerCalls;
            if (!object) LOG_WARN("Flexible pipeline %s failed: %s", generic.label.c_str(), reason(error).c_str());
        }
        std::lock_guard<std::mutex> lock(flexibleMutex_);
        flexible_[key] = {library, static_cast<MTL::RenderPipelineState*>(object)};
    });
    std::lock_guard<std::mutex> lock(flexibleMutex_);
    return flexible_[key].pipeline;
}

void PipelineCache::complete(const pipe::Completion& completion) {
    registry_.post(completion);
    {
        std::lock_guard<std::mutex> lock(waitMutex_);
        ++posted_;
    }
    waitCv_.notify_all();
}

void PipelineCache::waitForCompletions(const std::function<bool()>& done) {
    for (;;) {
        u64 seen = 0;
        {
            std::lock_guard<std::mutex> lock(waitMutex_);
            seen = posted_;
        }
        drain();
        if (done()) return;
        std::unique_lock<std::mutex> lock(waitMutex_);
        waitCv_.wait(lock, [&] { return posted_ != seen; });
    }
}

void PipelineCache::waitReady(pipe::PipelineHandle handle) {
    waitForCompletions([&] {
        return registry_.get(handle) != nullptr || registry_.state(handle) == pipe::PipelineState::Failed;
    });
}

void PipelineCache::waitAllReady() {
    waitForCompletions([&] {
        for (u32 h = 0; h < registry_.size(); ++h) {
            if (registry_.get(h)) continue;
            if (registry_.state(h) == pipe::PipelineState::Failed) {
                throw std::runtime_error("Pipeline " + registry_.desc(h).label + " failed to build");
            }
            return false;
        }
        return true;
    });
}

void PipelineCache::waitAllFinal() {
    // Waits on the compile queue's idle state, not on a completion count: a
    // job posts its completion BEFORE its worker stops counting it as running,
    // so a check of outstanding() that falls between the two sees one job left
    // and would then wait for a completion that never comes (F4, measured: a
    // capture run hung here for 25 minutes with every compile thread idle).
    for (;;) {
        if (queue_) queue_->waitIdle();
        drain();
        bool final = true;
        for (u32 h = 0; h < registry_.size() && final; ++h) {
            const pipe::PipelineState s = registry_.state(h);
            if (s != pipe::PipelineState::Ready && s != pipe::PipelineState::Failed) {
                if (!options_.fallbackOnly || s == pipe::PipelineState::Pending) final = false;
            }
        }
        if (final) return;
        // Idle queue, everything posted was drained and an entry is still not
        // final: nothing can complete it any more.
        if (!queue_ || queue_->outstanding() == 0) {
            LOG_ERROR("Pipeline cache: waitAllFinal found an entry that no job will complete");
            return;
        }
    }
}

MTL::RenderPipelineState* PipelineCache::render(pipe::PipelineHandle h) const {
    return static_cast<MTL::RenderPipelineState*>(static_cast<NS::Object*>(registry_.get(h)));
}

MTL::ComputePipelineState* PipelineCache::compute(pipe::PipelineHandle h) const {
    return static_cast<MTL::ComputePipelineState*>(static_cast<NS::Object*>(registry_.get(h)));
}

u32 PipelineCache::beginFrame() { return drain(); }

u32 PipelineCache::drain() {
    const u32 changed = registry_.drain();
    if (pendingLibrary_ && !registry_.reloadPending()) {
        if (registry_.generation() == pendingGeneration_) {
            // Reload committed: the new library is served from now on.
            context_.deferRelease(library_);
            library_ = pendingLibrary_;
            pruneFlexible_ = true;
            LOG_INFO("Shaders reloaded: generation %u, %u pipelines swapped", registry_.generation(), changed);
        } else {
            context_.deferRelease(pendingLibrary_);
            LOG_ERROR("Shader reload abandoned (a pipeline failed to compile); the previous pipelines stay");
        }
        pendingLibrary_ = nullptr;
    }
    if (pruneFlexible_) pruneFlexibleBases();
    return changed;
}

void PipelineCache::pruneFlexibleBases() {
    // Jobs of the previous generation were cancelled or are finishing; new
    // ones use library_.  Once nothing runs, no job can hold an old base.
    if (queue_ && queue_->outstanding() != 0) return;
    std::lock_guard<std::mutex> lock(flexibleMutex_);
    u32 released = 0;
    for (auto it = flexible_.begin(); it != flexible_.end();) {
        if (it->second.library == library_) {
            ++it;
            continue;
        }
        if (it->second.pipeline) context_.deferRelease(it->second.pipeline);
        flexibleOnce_.erase(it->first);
        it = flexible_.erase(it);
        ++released;
    }
    pruneFlexible_ = false;
    if (released) LOG_INFO("Released %u flexible pipelines of the previous shader library", released);
}

bool PipelineCache::reload(MTL::Library* library) {
    if (registry_.reloadPending() || options_.sync) return false;
    library->retain();
    pendingLibrary_ = library;
    const u32 generation = registry_.beginGeneration();
    pendingGeneration_ = generation;
    queue_->cancelBefore(generation);
    for (u32 h = 0; h < registry_.size(); ++h) {
        const pipe::PipelineDesc desc = registry_.desc(h);
        queue_->submit(pipe::CompilePriority::Specialize, generation,
                       [this, h, desc, generation, library, ref = retainLibrary(library)] {
                           // The archive is keyed by function hash: unchanged
                           // functions still hit, changed ones compile.
                           resolve(h, desc, generation, library, /*allowFallback*/ false);
                       });
    }
    LOG_INFO("Shader reload: recompiling %u pipelines (generation %u)", registry_.size(), generation);
    return true;
}

bool PipelineCache::writeHarvest() {
    if (!serializer_) return false;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    NS::Error* error = nullptr;
    NS::Data* script = serializer_->serializeAsPipelinesScript(&error);
    bool ok = false;
    if (!script) {
        LOG_ERROR("Pipeline harvest failed: %s", reason(error).c_str());
    } else {
        std::ofstream out(options_.harvestPath, std::ios::binary);
        out.write(static_cast<const char*>(script->bytes()), static_cast<std::streamsize>(script->length()));
        ok = static_cast<bool>(out);
        if (ok) {
            LOG_INFO("Pipeline descriptors written to %s (%lu bytes, %u pipelines requested)",
                     options_.harvestPath.c_str(), static_cast<unsigned long>(script->length()), registry_.size());
        } else {
            LOG_ERROR("Failed to write %s", options_.harvestPath.c_str());
        }
    }
    pool->release();
    return ok;
}

} // namespace phosphor
