#include "platform/metal/denoise_passes.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/metal_graph_executor.h"
#include "rendergraph/pass_context.h"
#include <array>
#include <cstring>
#include <string>
#include <limits>
namespace phosphor {
struct DenoisePasses::Impl {
    static constexpr u32 Signals=4,Views=4,MaxAtrous=5;
    MetalContext& c;PipelineCache& pipelines;LaunchOptions options;DenoiseSettings settings;
    ShadowPasses::Frame frame{};u64 epoch=0,graphVersion=1;
    pipe::PipelineHandle temporal{},atrous{},checker{},clear{},corrupt{},poisonNext{};
    std::array<HistoryRegistry,Signals> registries{};
    struct Content {u64 scene=0,external=0,revision=0;bool operator==(const Content&)const=default;};
    struct History {MTL::Buffer* pair[2]{};u64 capacity=0,contentEpoch=0;u32 lastWritten=0;Content content{};bool contentKnown=false;};
    std::array<std::array<History,Signals>,Views> histories{};
    struct SlotSignal {
        std::array<MTL4::ArgumentTable*,MaxAtrous+5> tables{};
        MTL::Buffer* check=nullptr;bool used=false;u32 expected=0;
    };
    std::array<std::array<SlotSignal,Signals>,METAL_FRAMES_IN_FLIGHT> slots{};
    struct Signal {
        bool defined=false;
        u32 readSide=0,writeSide=1;
        GPUDenoiseParams params{};MTL::GPUAddress address=0;
        std::array<MTL::GPUAddress,MaxAtrous> atrousAddress{};
        rg::BufferRef previous{},next{},counts{},surface{},metadata{};
        rg::TextureRef raw{},motion{},temporalOutput{},moments{},filtered{};
        std::array<rg::TextureRef,MaxAtrous> atrousInput{},atrousOutput{};
    };
    std::array<Signal,Signals> signal{};
    std::array<u64,Signals> revisions{};
    Impl(MetalContext& context,PipelineCache& p,const LaunchOptions& o):c(context),pipelines(p),options(o){
        if(o.reducedLighting||o.forceApple9){settings.atrousIterations=2;settings.maxHistory=16;}
        temporal=p.request(lighting::kernel("denoise_temporal"));atrous=p.request(lighting::kernel("denoise_atrous"));
        checker=p.request(lighting::kernel("denoise_check"));clear=p.request(lighting::kernel("lighting_check_clear"));
        corrupt=p.request(lighting::kernel("denoise_history_corrupt_safe"));
        if(o.debugReflectionCorrupt==1)poisonNext=p.request(lighting::kernel("denoise_next_foreign_view"));
        for(auto& slot:slots)for(auto& s:slot){for(auto*& table:s.tables)table=lighting::table(c);
            s.check=lighting::buffer(c,32,"F13 per-signal denoise checks",true);}
    }
    ~Impl(){c.waitIdle();for(auto& view:histories)for(auto& h:view)for(auto* b:h.pair)c.memory().release(b,MemoryCategory::RayTracing);
        for(auto& slot:slots)for(auto& s:slot){for(auto* t:s.tables)if(t)t->release();c.memory().release(s.check,MemoryCategory::RayTracing);}}
    void invalidate(const char* reason){for(auto& r:registries)for(u32 v=0;v<Views;++v)r.invalidate(v,reason);}
    void reserve(u32 sig){auto& h=histories.at(frame.view).at(sig);const u64 count=u64(frame.backingWidth)*frame.backingHeight;
        if(count<=h.capacity)return;for(auto* b:h.pair)c.memory().release(b,MemoryCategory::RayTracing);
        h.capacity=count;h.lastWritten=0;for(auto*& b:h.pair)b=lighting::buffer(c,count*sizeof(GPUDenoiseHistory),"F13 view signal history");
        registries[sig].invalidate(frame.view,"denoise allocation growth");++graphVersion;
    }
    void updateSignal(u32 sig){reserve(sig);auto& h=histories[frame.view][sig];auto& s=signal[sig];
        const Content content{frame.scene,epoch,revisions[sig]};if(!h.contentKnown||h.content!=content){h.content=content;h.contentKnown=true;++h.contentEpoch;}
        const auto decision=registries[sig].begin(frame.view,{frame.width,frame.height,frame.backingWidth,frame.backingHeight},h.contentEpoch,frame.cut,frame.reset);
        s.params=denoiseParameters(settings,frame.width,frame.height,sig,frame.view,u32(decision.generation),u32(h.contentEpoch),decision.reset);
        s.params.frameIndex=u32(frame.index);s.address=lighting::upload(c,s.params);s.readSide=h.lastWritten;s.writeSide=1u-s.readSide;
        for(u32 i=0;i<settings.atrousIterations;++i){auto p=s.params;p.atrousStep=1u<<i;s.atrousAddress[i]=lighting::upload(c,p);}
        slots[frame.slot][sig].expected=frame.width*frame.height;
    }
    void prepare(const ShadowPasses::Frame& f,u64 signalEpoch,const std::array<u64,Signals>& currentRevisions){frame=f;epoch=signalEpoch;revisions=currentRevisions;
        if(!f.width||!f.height||f.width>f.backingWidth||f.height>f.backingHeight||f.slot>=METAL_FRAMES_IN_FLIGHT||f.view>=Views||
            u64(f.backingWidth)*f.backingHeight>std::numeric_limits<u32>::max())throw std::invalid_argument("Invalid F13 denoise frame extent or view");
        for(u32 sig=0;sig<Signals;++sig){auto& state=slots.at(frame.slot)[sig];state.used=signal[sig].defined;if(state.used)updateSignal(sig);}
        // Allocate only signals selected for this view, before graph callbacks.
        // First use/resize is a resource event; steady frames allocate nothing.
        if(options.directLighting!=DirectLightingMode::Legacy)reserve(DENOISE_SIGNAL_DI);
        if(options.gi!=GiMode::Off)reserve(DENOISE_SIGNAL_GI);
        if(options.reflections!=ReflectionMode::Off)reserve(DENOISE_SIGNAL_SPECULAR);
        if(options.ao!=AoMode::Off)reserve(DENOISE_SIGNAL_AO);
    }
    void texture(MTL4::ArgumentTable* t,rg::PassContext& ctx,rg::TextureRef r,u32 index){
        t->setTexture(static_cast<MTL::Texture*>(ctx.texture(r))->gpuResourceID(),index);
    }
    rg::TextureRef add(rg::RenderGraph& g,u32 sig,rg::TextureRef raw,rg::TextureRef motion,rg::BufferRef surfaces,rg::BufferRef metadata){
        using namespace rg;if(sig>=Signals||!raw.valid()||!motion.valid()||!surfaces.valid())throw std::invalid_argument("invalid denoise signal graph inputs");
        if(sig==DENOISE_SIGNAL_SPECULAR&&!metadata.valid())throw std::invalid_argument("specular denoise requires hit metadata");
        reserve(sig);auto& h=histories.at(frame.view).at(sig);auto& s=signal[sig];auto& state=slots[frame.slot][sig];state.used=true;state.expected=frame.width*frame.height;
        s.defined=true;updateSignal(sig);
        s.raw=raw;s.motion=motion;s.surface=surfaces;s.metadata=metadata;
        const std::string label="F13 denoise "+std::to_string(sig);
        // Both imports stay persistent even though pair sides are rebound.
        // The temporal pass reads AND writes them at Dispatch; the graph emits
        // its cross-frame queue barrier before either physical side is used.
        s.previous=g.importBuffer(label+" previous history",{h.capacity*sizeof(GPUDenoiseHistory)},ImportContentsDefined);
        s.next=g.importBuffer(label+" next history",{h.capacity*sizeof(GPUDenoiseHistory)},ImportOutput);
        if(options.debugHistoryCorrupt)g.addPass(label+" negative view",PassType::Compute,[this,sig](PassBuilder& b){
            auto& s=signal[sig];b.read(s.previous,Usage::ShaderRead,StageDispatch);s.previous=b.write(s.previous,Usage::ShaderWrite,StageDispatch);
        },[this,sig](PassContext& ctx){auto& s=signal[sig];auto& h=histories[frame.view][sig];auto* t=slots[frame.slot][sig].tables[MaxAtrous+3];
            t->setAddress(s.address,0);t->setAddress(h.pair[s.readSide]->gpuAddress(),1);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,corrupt,t,frame.width*frame.height);});
        g.addPass(label+" temporal",PassType::Compute,[this,sig](PassBuilder& b){auto& s=signal[sig];
            b.read(s.raw,Usage::ShaderRead,StageDispatch);b.read(s.motion,Usage::ShaderRead,StageDispatch);b.read(s.surface,Usage::ShaderRead,StageDispatch);b.read(s.previous,Usage::ShaderRead,StageDispatch);
            if(s.metadata.valid())b.read(s.metadata,Usage::ShaderRead,StageDispatch);s.next=b.write(s.next,Usage::ShaderWrite,StageDispatch);
            s.temporalOutput=b.createTexture("F13 temporal signal "+std::to_string(sig),{Format::RGBA32Float,frame.width,frame.height});s.temporalOutput=b.write(s.temporalOutput,Usage::ShaderWrite,StageDispatch);
            s.moments=b.createTexture("F13 moments variance "+std::to_string(sig),{Format::RGBA32Float,frame.width,frame.height});s.moments=b.write(s.moments,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("denoise_temporal");
        },[this,sig](PassContext& ctx){auto& s=signal[sig];auto& h=histories[frame.view][sig];auto* t=slots[frame.slot][sig].tables[0];
            t->setAddress(s.address,0);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(s.surface))->gpuAddress(),1);
            t->setAddress(h.pair[s.readSide]->gpuAddress(),2);if(s.metadata.valid())t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(s.metadata))->gpuAddress(),3);
            else t->setAddress(h.pair[s.readSide]->gpuAddress(),3);t->setAddress(h.pair[s.writeSide]->gpuAddress(),4);
            texture(t,ctx,s.raw,0);texture(t,ctx,s.motion,1);texture(t,ctx,s.temporalOutput,2);texture(t,ctx,s.moments,3);
            lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,temporal,t,frame.width*frame.height);
            // Encoding is not GPU completion. Persistent imports/order preserve
            // retirement; registry records the submission identity only.
            registries[sig].read(frame.view,frame.index+1);registries[sig].write(frame.view,frame.index+1,frame.unjitteredVP);h.lastWritten=s.writeSide;
        });
        s.filtered=s.temporalOutput;
        for(u32 iteration=0;iteration<settings.atrousIterations;++iteration){s.atrousInput[iteration]=s.filtered;
            g.addPass(label+" atrous "+std::to_string(iteration),PassType::Compute,[this,sig,iteration](PassBuilder& b){auto& s=signal[sig];
                b.read(s.atrousInput[iteration],Usage::ShaderRead,StageDispatch);b.read(s.moments,Usage::ShaderRead,StageDispatch);b.read(s.surface,Usage::ShaderRead,StageDispatch);
                if(s.metadata.valid())b.read(s.metadata,Usage::ShaderRead,StageDispatch);s.atrousOutput[iteration]=b.createTexture("F13 filtered signal "+std::to_string(sig)+" "+std::to_string(iteration),{Format::RGBA32Float,frame.width,frame.height});
                s.atrousOutput[iteration]=b.write(s.atrousOutput[iteration],Usage::ShaderWrite,StageDispatch);b.setProfileShaders("denoise_atrous");
            },[this,sig,iteration](PassContext& ctx){auto& s=signal[sig];auto* t=slots[frame.slot][sig].tables[1+iteration];t->setAddress(s.atrousAddress[iteration],0);
                t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(s.surface))->gpuAddress(),1);t->setAddress(s.metadata.valid()?static_cast<MTL::Buffer*>(ctx.buffer(s.metadata))->gpuAddress():histories[frame.view][sig].pair[s.readSide]->gpuAddress(),3);
                texture(t,ctx,s.atrousInput[iteration],0);texture(t,ctx,s.moments,1);texture(t,ctx,s.atrousOutput[iteration],2);
                lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,atrous,t,frame.width*frame.height);});s.filtered=s.atrousOutput[iteration];}
        if(options.debugReflectionCorrupt==1)g.addPass(label+" negative NEXT foreign view",PassType::Compute,[this,sig](PassBuilder& b){auto& s=signal[sig];
            b.read(s.next,Usage::ShaderRead,StageDispatch);s.next=b.write(s.next,Usage::ShaderWrite,StageDispatch);b.setProfileShaders("denoise_next_foreign_view");
        },[this,sig](PassContext& ctx){auto& s=signal[sig];auto* t=slots[frame.slot][sig].tables[MaxAtrous+4];t->setAddress(s.address,0);t->setAddress(histories[frame.view][sig].pair[s.writeSide]->gpuAddress(),1);
            lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,poisonNext,t,frame.width*frame.height);});
        {s.counts=g.importBuffer(label+" check counts",{32},ImportPerFrame|ImportOutput);
            g.addPass(label+" checks clear",PassType::Compute,[this,sig](PassBuilder& b){signal[sig].counts=b.write(signal[sig].counts,Usage::ShaderWrite,StageDispatch);},[this,sig](PassContext& ctx){auto* t=slots[frame.slot][sig].tables[MaxAtrous+1];t->setAddress(slots[frame.slot][sig].check->gpuAddress(),0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,clear,t,8);});
            g.addPass(label+" independent state check",PassType::Compute,[this,sig](PassBuilder& b){auto& s=signal[sig];b.read(s.next,Usage::ShaderRead,StageDispatch);b.read(s.filtered,Usage::ShaderRead,StageDispatch);b.read(s.counts,Usage::ShaderRead,StageDispatch);s.counts=b.write(s.counts,Usage::ShaderWrite,StageDispatch);},[this,sig](PassContext& ctx){auto& s=signal[sig];auto* t=slots[frame.slot][sig].tables[MaxAtrous+2];t->setAddress(s.address,0);t->setAddress(histories[frame.view][sig].pair[s.writeSide]->gpuAddress(),1);t->setAddress(slots[frame.slot][sig].check->gpuAddress(),2);texture(t,ctx,s.filtered,0);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),pipelines,checker,t,frame.width*frame.height);});}
        return s.filtered;
    }
    void bind(MetalGraphExecutor& e){for(u32 sig=0;sig<Signals;++sig){if(!slots[frame.slot][sig].used)continue;const auto& s=signal[sig];const auto& h=histories[frame.view][sig];
        e.bindBuffer(s.previous,h.pair[s.readSide]);e.bindBuffer(s.next,h.pair[s.writeSide]);e.bindBuffer(s.counts,slots[frame.slot][sig].check);}}
};
DenoisePasses::DenoisePasses(MetalContext& c,PipelineCache& p,const LaunchOptions& o):impl_(std::make_unique<Impl>(c,p,o)){}
DenoisePasses::~DenoisePasses()=default;
void DenoisePasses::prepareFrame(const ShadowPasses::Frame& f,u64 epoch,const std::array<u64,4>& revisions){impl_->prepare(f,epoch,revisions);}
void DenoisePasses::invalidateAll(const char* reason){impl_->invalidate(reason);}
rg::TextureRef DenoisePasses::addSignal(rg::RenderGraph& g,u32 s,rg::TextureRef r,rg::TextureRef m,rg::BufferRef b,rg::BufferRef metadata){return impl_->add(g,s,r,m,b,metadata);}
void DenoisePasses::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}u64 DenoisePasses::version()const{return impl_->graphVersion;}
bool DenoisePasses::check(u32 slot)const{for(const auto& s:impl_->slots.at(slot)){if(!s.used)continue;const auto* words=static_cast<const u32*>(s.check->contents());if(words[0]!=s.expected||words[2]||words[3])return false;}return true;}
bool DenoisePasses::ready()const{return impl_->pipelines.compute(impl_->temporal)&&impl_->pipelines.compute(impl_->atrous)&&
    impl_->pipelines.compute(impl_->checker)&&impl_->pipelines.compute(impl_->clear)&&
    (!impl_->options.debugHistoryCorrupt||impl_->pipelines.compute(impl_->corrupt))&&
    (impl_->options.debugReflectionCorrupt!=1||impl_->pipelines.compute(impl_->poisonNext));}
} // namespace phosphor
