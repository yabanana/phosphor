#include "platform/metal/volume_diagnostics.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/gpu_memory.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"
#include <json.hpp>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace phosphor {
namespace {
MTL::Buffer* readbackBuffer(MetalContext& c,u64 size,const char* name){auto* p=c.memory().newBuffer(size,MTL::ResourceStorageModeShared,MemoryCategory::Other,name);if(!p)throw std::runtime_error("Volume diagnostic allocation failed");std::memset(p->contents(),0,size);return p;}
std::string sourceHash(){u64 hash=1469598103934665603ull;
#ifdef PHOSPHOR_SHADER_SOURCE_DIR
    const std::filesystem::path shaders(PHOSPHOR_SHADER_SOURCE_DIR),src=shaders.parent_path()/"src/renderer";
    for(const auto& path:{shaders/"atmosphere.metal",shaders/"fog.metal",shaders/"clouds.metal",shaders/"atmosphere_common.h",shaders/"volume_diagnostics.metal",src/"gpu_types.h",src/"volume_noise.h",src/"volume_oracle.cpp"}){
        std::ifstream in(path,std::ios::binary);if(!in)return "unavailable";char block[4096];while(in){in.read(block,sizeof(block));for(std::streamsize i=0;i<in.gcount();++i){hash^=u8(block[i]);hash*=1099511628211ull;}}}
#else
    return "unavailable";
#endif
    std::ostringstream text;text<<std::hex<<hash;return text.str();}
}
struct VolumeDiagnostics::Impl {
    MetalContext& c;PipelineCache& p;Config config;u32 slot=0,view=0;bool capture=false;
    pipe::PipelineHandle stamp,fixture,foreignFog,foreignCloud,collector,solar;
    MTL::Buffer* globalProduced=nullptr;std::array<MTL::Buffer*,4> skyProduced{};
    struct Slot {
        MTL::Buffer* samples=nullptr;MTL::Buffer* counters=nullptr;
        std::array<MTL4::ArgumentTable*,3> stampTables{};std::array<MTL4::ArgumentTable*,2> foreignTables{};
        MTL4::ArgumentTable *fixtureTable=nullptr,*collectTable=nullptr,*solarTable=nullptr;
        MTL::GPUAddress atmosphereAddress=0,diagnosticAddress=0;
        std::array<MTL::GPUAddress,3> stampAddress{};
        GPUAtmosphereParams submitted{};VolumeOracleInput expected;u64 frame=0;u32 view=0,generation=0;
        bool pending=false,armed=false;std::string sourceHash;
        nlohmann::json provenance;
    };std::array<Slot,METAL_FRAMES_IN_FLIGHT> slots{};
    rg::BufferRef globalRef{},skyRef{},samplesRef{};rg::TextureRef solarRef{};
    GPUVolumeDiagnosticParams params{};
    Impl(MetalContext& context,PipelineCache& pipelines,Config cfg):c(context),p(pipelines),config(std::move(cfg)) {
        if(!config.every)throw std::invalid_argument("Volume oracle interval must be positive");
        stamp=p.request(lighting::kernel("volume_lut_stamp"));fixture=p.request(lighting::kernel("volume_homogeneous_fog"));
        foreignFog=p.request(lighting::kernel("volume_foreign_fog_history"));foreignCloud=p.request(lighting::kernel("volume_foreign_cloud_history"));
        collector=p.request(lighting::kernel("volume_numeric_collect"));solar=p.request(lighting::kernel("volume_solar_wide_probe"));
        globalProduced=readbackBuffer(c,16,"Actual LUT produced revisions");for(auto*& b:skyProduced)b=readbackBuffer(c,16,"Actual sky produced revision per view");
        for(auto& s:slots){s.samples=readbackBuffer(c,11*sizeof(GPUVolumeNumericSample),"Sparse same-frame F14 numerical readback");for(auto*& t:s.stampTables)t=lighting::table(c);
            for(auto*& t:s.foreignTables)t=lighting::table(c);s.fixtureTable=lighting::table(c);s.collectTable=lighting::table(c);s.solarTable=lighting::table(c);}
    }
    ~Impl(){c.waitIdle();c.memory().release(globalProduced,MemoryCategory::Other);for(auto* b:skyProduced)c.memory().release(b,MemoryCategory::Other);
        for(auto& s:slots){c.memory().release(s.samples,MemoryCategory::Other);for(auto* t:s.stampTables)t->release();for(auto* t:s.foreignTables)t->release();s.fixtureTable->release();s.collectTable->release();s.solarTable->release();}}
    void tex(MTL4::ArgumentTable* t,rg::PassContext& ctx,rg::TextureRef ref,u32 index){t->setTexture(static_cast<MTL::Texture*>(ctx.texture(ref))->gpuResourceID(),index);}
    bool prepare(u32 index,u64 frame,u32 currentView,const GPUAtmosphereParams& expected,const GPUAtmosphereParams& submitted,const GPUFogParams& fog,bool armed){
        slot=index;view=currentView;capture=(frame+1)%config.every==0;auto& s=slots.at(index);if(s.pending)throw std::logic_error("Consume immutable volume snapshot before slot reuse");
        s.frame=frame;s.view=currentView;s.generation=p.generation();s.armed=armed;s.submitted=submitted;s.expected={expected,fog,{},config.homogeneousFog};s.sourceHash=capture?sourceHash():std::string();
        s.provenance=nullptr;if(capture&&!config.path.empty()){std::ifstream manifest(std::filesystem::path(config.path)/"provenance.json");if(manifest)manifest>>s.provenance;}
        params={};params.transWidth=expected.transmittanceWidth;params.transHeight=expected.transmittanceHeight;params.multiWidth=expected.multiWidth;params.multiHeight=expected.multiHeight;
        params.skyWidth=expected.skyWidth;params.skyHeight=expected.skyHeight;params.fogX=fog.gridX;params.fogY=fog.gridY;params.fogZ=fog.gridZ;
        params.frameLo=u32(frame);params.frameHi=u32(frame>>32);params.expectedPhysicsRevision=expected.parameterRevision;params.expectedSkyRevision=expected.skyRevision;
        params.sampleCount=config.homogeneousFog?11u:8u;params.corruption=config.corruption;params.homogeneous=config.homogeneousFog;
        params.fixtureExtinction=0.01f;params.fixtureSource[0]=0.01f;params.fixtureSource[1]=0.02f;params.fixtureSource[2]=0.03f;
        if(config.homogeneousFog){const u32 xy=(fog.gridY/2)*fog.gridX+fog.gridX/2;params.fogIndices[0]=xy+fog.gridX*fog.gridY;params.fogIndices[1]=xy+(fog.gridZ/2)*fog.gridX*fog.gridY;params.fogIndices[2]=xy+(fog.gridZ-1)*fog.gridX*fog.gridY;}
        s.expected.diagnostics=params;s.atmosphereAddress=lighting::upload(c,submitted);s.diagnosticAddress=lighting::upload(c,params);
        for(u32 kind=0;kind<s.stampAddress.size();++kind){auto stampParams=params;stampParams.stampKind=kind;s.stampAddress[kind]=lighting::upload(c,stampParams);}return capture;
    }
    void begin(rg::RenderGraph& g){globalRef=g.importBuffer("Actual atmosphere produced epochs",{16},rg::ImportContentsDefined|rg::ImportOutput);
        skyRef=g.importBuffer("Actual sky produced epoch per view",{16},rg::ImportContentsDefined|rg::ImportOutput);samplesRef={};}
    void stampProducer(rg::RenderGraph& g,rg::TextureRef texture,u32 kind){using namespace rg;
        auto& ref=kind==2?skyRef:globalRef;
        g.addPass("Actual LUT producer revision "+std::to_string(kind),PassType::Compute,[&](PassBuilder& b){b.read(texture,Usage::ShaderRead,StageDispatch);ref=b.write(ref,Usage::ShaderWrite,StageDispatch);},
            [this,kind](PassContext& ctx){auto& s=slots[slot];auto* t=s.stampTables[kind];t->setAddress(s.atmosphereAddress,0);t->setAddress(s.stampAddress[kind],17);t->setAddress((kind==2?skyProduced[view]:globalProduced)->gpuAddress(),18);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,stamp,t,1);});}
    void homogeneous(rg::PassContext& ctx,MTL::Buffer* cells,MTL::GPUAddress fogAddress){auto& s=slots[slot];auto* t=s.fixtureTable;t->setAddress(fogAddress,0);t->setAddress(cells->gpuAddress(),1);t->setAddress(s.diagnosticAddress,17);
        auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());e->setComputePipelineState(p.compute(fixture));e->setArgumentTable(t);e->dispatchThreads(MTL::Size::Make(params.fogX,params.fogY,params.fogZ),MTL::Size::Make(4,4,4));}
    void foreign(rg::RenderGraph& g,rg::BufferRef& history,MTL::Buffer* buffer,u32 count,bool cloud){if(!capture||config.corruption!=VOLUME_CORRUPT_HISTORY)return;using namespace rg;
        (void)buffer;const BufferRef input=history;
        g.addPass(cloud?"Negative actual foreign cloud history":"Negative actual foreign fog history",PassType::Compute,[&](PassBuilder& b){b.read(history,Usage::ShaderRead,StageDispatch);history=b.write(history,Usage::ShaderWrite,StageDispatch);},
            [this,input,count,cloud](PassContext& ctx){auto& s=slots[slot];auto diag=s.expected.diagnostics;if(cloud)diag.fogIndices[0]=count;
                auto* t=s.foreignTables[cloud?1:0];t->setAddress(lighting::upload(c,diag),17);t->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(input))->gpuAddress(),18);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,cloud?foreignCloud:foreignFog,t,count);});}
    void collect(rg::RenderGraph& g,const Sources& src){if(!capture)return;using namespace rg;
        samplesRef=g.importBuffer("Immutable same-frame F14 oracle samples",{11*sizeof(GPUVolumeNumericSample)},ImportPerFrame|ImportOutput);
        g.addPass("Actual shared solar disk wide HDR probe",PassType::Compute,[&](PassBuilder& b){solarRef=b.createTexture("Solar toward-away-tangent RGBA32",{Format::RGBA32Float,3,1});solarRef=b.write(solarRef,Usage::ShaderWrite,StageDispatch);},
            [this](PassContext& ctx){auto& s=slots[slot];auto* t=s.solarTable;t->setAddress(s.atmosphereAddress,0);tex(t,ctx,solarRef,5);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,solar,t,3);});
        g.addPass("Same-frame independent volume oracle readback",PassType::Compute,[&](PassBuilder& b){for(auto r:{src.transmittance,src.multiple,src.sky,solarRef})b.read(r,Usage::ShaderRead,StageDispatch);b.read(globalRef,Usage::ShaderRead,StageDispatch);b.read(skyRef,Usage::ShaderRead,StageDispatch);
            b.read(src.counters,Usage::ShaderRead,StageDispatch);
            if(config.homogeneousFog){b.read(src.fogCells,Usage::ShaderRead,StageDispatch);b.read(src.fogIntegrated,Usage::ShaderRead,StageDispatch);}samplesRef=b.write(samplesRef,Usage::ShaderWrite,StageDispatch);},
            [this,src](PassContext& ctx){auto& s=slots[slot];auto* t=s.collectTable;t->setAddress(s.atmosphereAddress,0);t->setAddress(s.diagnosticAddress,17);t->setAddress(globalProduced->gpuAddress(),18);t->setAddress(skyProduced[view]->gpuAddress(),19);t->setAddress(s.samples->gpuAddress(),20);
                s.counters=static_cast<MTL::Buffer*>(ctx.buffer(src.counters));
                t->setAddress(config.homogeneousFog?static_cast<MTL::Buffer*>(ctx.buffer(src.fogIntegrated))->gpuAddress():s.samples->gpuAddress(),21);
                t->setAddress(config.homogeneousFog?static_cast<MTL::Buffer*>(ctx.buffer(src.fogCells))->gpuAddress():s.samples->gpuAddress(),22);
                tex(t,ctx,src.transmittance,0);tex(t,ctx,src.multiple,1);tex(t,ctx,src.sky,2);tex(t,ctx,solarRef,3);lighting::dispatch(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),p,collector,t,params.sampleCount);s.pending=true;});}
    bool consume(u32 index){auto& s=slots.at(index);if(!s.pending)return true;if(c.frameEvent()->signaledValue()<=s.frame)throw std::logic_error("Volume oracle read before GPU completion");
        const auto* data=static_cast<const GPUVolumeNumericSample*>(s.samples->contents());const auto cases=evaluateVolumeOracle(s.expected,{data,s.expected.diagnostics.sampleCount});
        nlohmann::json output;output["schema"]="phosphor.volume-oracle.v1";output["kind"]="f14-volume";output["frame"]=s.frame;output["view"]=s.view;output["corruption_requested"]=config.corruption;output["corruption_armed"]=s.armed;
        output["provenance"]={{"shader_generation",s.generation},{"source_hash_method","fnv1a64-source-files-at-prepare"},{"source_hash_at_prepare",s.sourceHash},{"source_sha",nullptr},{"binary_sha",nullptr},{"manifest",nullptr},{"gpu_name",c.gpuName()}};
        if(!s.provenance.is_null()){output["provenance"]["supplied_manifest"]=s.provenance;for(const char* field:{"source_sha","binary_sha","manifest"})if(s.provenance.contains(field))output["provenance"][field]=s.provenance[field];}
        std::array<u32,sizeof(GPUAtmosphereParams)/4> expectedWords{},submittedWords{};std::array<u32,sizeof(GPUFogParams)/4> fogWords{};
        std::memcpy(expectedWords.data(),&s.expected.atmosphere,sizeof(GPUAtmosphereParams));std::memcpy(submittedWords.data(),&s.submitted,sizeof(GPUAtmosphereParams));std::memcpy(fogWords.data(),&s.expected.fog,sizeof(GPUFogParams));
        output["parameter_blocks"]={{"encoding","little-endian-u32-shared-scalar-ABI"},{"expected_atmosphere",expectedWords},{"submitted_atmosphere",submittedWords},{"fog",fogWords}};
        output["reference"]="adaptive-Simpson transport; independent Gauss-Legendre angular closure and CPU LUT-node memoization; homogeneous analytic exp; shared solar producer with independent CPU radiance/orientation";
        output["settings"]={{"homogeneous_fog",config.homogeneousFog},{"fixture_extinction_m_inv",s.expected.diagnostics.fixtureExtinction},{"fixture_source",{s.expected.diagnostics.fixtureSource[0],s.expected.diagnostics.fixtureSource[1],s.expected.diagnostics.fixtureSource[2]}},
            {"expected_physics_revision",s.expected.atmosphere.parameterRevision},{"expected_sky_revision",s.expected.atmosphere.skyRevision},{"planet_radius_m",s.expected.atmosphere.bottomRadius},{"march_steps",s.submitted.marchSteps},
            {"expected_sun_irradiance",{s.expected.atmosphere.sunIrradiance[0],s.expected.atmosphere.sunIrradiance[1],s.expected.atmosphere.sunIrradiance[2]}},{"submitted_sun_irradiance",{s.submitted.sunIrradiance[0],s.submitted.sunIrradiance[1],s.submitted.sunIrradiance[2]}}};
        bool passed=true;output["cases"]=nlohmann::json::array();for(const auto& item:cases){passed&=item.passed;output["cases"].push_back({{"kind",item.kind},{"pixel",{item.x,item.y}},{"expected",item.expected},{"actual",item.actual},{"expected_epoch",item.expectedEpoch},{"actual_epoch",item.actualEpoch},
            {"absolute_error",item.absoluteError},{"relative_error",item.relativeError},{"tolerance",{{"absolute",item.absoluteTolerance},{"relative",item.relativeTolerance}}},{"passed",item.passed}});}
        if(s.counters){const auto& counters=*static_cast<const GPUVolumeCounters*>(s.counters->contents());output["gpu_counters"]={{"nonfinite",counters.nonfinite},{"invalid_units",counters.invalidUnits},{"invalid_history",counters.invalidHistory},{"history_reused",counters.historyReused}};passed&=!counters.nonfinite&&!counters.invalidUnits&&!counters.invalidHistory;}
        output["passed"]=passed;output["certification"]=false;
        if(!config.path.empty()){std::error_code ec;std::filesystem::create_directories(config.path,ec);std::ostringstream filename;filename<<"frame-"<<std::setfill('0')<<std::setw(6)<<s.frame<<"-view-"<<s.view<<".json";const auto path=std::filesystem::path(config.path)/filename.str();
            if(ec||std::filesystem::exists(path))throw std::runtime_error("Volume oracle output directory failed or immutable snapshot already exists");std::ofstream file(path);file<<output.dump(2)<<'\n';if(!file)throw std::runtime_error("Volume oracle JSON write failed");}
        LOG_INFO("F14 sparse oracle frame %llu view %u: %s",static_cast<unsigned long long>(s.frame),s.view,passed?"PASS":"FAIL");s.pending=false;return passed;}
};
VolumeDiagnostics::VolumeDiagnostics(MetalContext& c,PipelineCache& p,Config config):impl_(std::make_unique<Impl>(c,p,std::move(config))){}
VolumeDiagnostics::~VolumeDiagnostics()=default;
bool VolumeDiagnostics::prepare(u32 s,u64 f,u32 v,const GPUAtmosphereParams& e,const GPUAtmosphereParams& a,const GPUFogParams& fog,bool armed){return impl_->prepare(s,f,v,e,a,fog,armed);}
void VolumeDiagnostics::beginGraph(rg::RenderGraph& g){impl_->begin(g);}
void VolumeDiagnostics::stamp(rg::RenderGraph& g,rg::TextureRef t,u32 kind){impl_->stampProducer(g,t,kind);}
void VolumeDiagnostics::homogeneous(rg::PassContext& c,MTL::Buffer* b,MTL::GPUAddress p){impl_->homogeneous(c,b,p);}
void VolumeDiagnostics::foreignHistory(rg::RenderGraph& g,rg::BufferRef& r,MTL::Buffer* p,u32 count,bool cloud){impl_->foreign(g,r,p,count,cloud);}
void VolumeDiagnostics::collect(rg::RenderGraph& g,const Sources& s){impl_->collect(g,s);}
void VolumeDiagnostics::bindFrame(MetalGraphExecutor& e){e.bindBuffer(impl_->globalRef,impl_->globalProduced);e.bindBuffer(impl_->skyRef,impl_->skyProduced[impl_->view]);if(impl_->samplesRef.valid())e.bindBuffer(impl_->samplesRef,impl_->slots[impl_->slot].samples);}
bool VolumeDiagnostics::consume(u32 s){return impl_->consume(s);}
bool VolumeDiagnostics::selected()const{return impl_->capture;}
bool VolumeDiagnostics::homogeneousFog()const{return impl_->config.homogeneousFog;}
u32 VolumeDiagnostics::corruption()const{return impl_->config.corruption;}
} // namespace phosphor
