#include "platform/metal/reference_snapshot.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/lighting_dispatch.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/metal_texture_manager.h"
#include "platform/metal/direct_lighting_passes.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "rendergraph/pass_context.h"
#include <limits>
namespace phosphor {
struct ReferenceSnapshot::Impl {
    MetalContext& c;SceneRenderer& renderer;AccelerationStructures* rt;std::string destination;u64 captureFrame=0;
    std::vector<GPUVertex> vertices;std::vector<u32> indices;std::vector<GPUMeshInfo> meshes;std::vector<ReferenceTexture> textures;
    struct Slot {MTL::Buffer *instances=nullptr,*materials=nullptr,*lights=nullptr,*emitters=nullptr,*vertices=nullptr;u32 instanceCount=0,materialCount=0,lightCount=0;bool recorded=false;u64 frame=0;ReferenceCamera camera{};std::vector<GPULight> punctual;};
    std::array<Slot,3> slots{};u32 slot=0;u64 graphVersion=1;bool done=false,requested=false;
    DirectLightingPasses* direct=nullptr;ShadowPasses::Frame frame{};
    rg::BufferRef instanceRef{},materialRef{},lightRef{},emitterRef{},vertexRef{};
    Impl(MetalContext& context,SceneRenderer& r,const std::string& d,u64 capture,AccelerationStructures* a):c(context),renderer(r),rt(a),destination(d),captureFrame(capture){}
    ~Impl(){c.waitIdle();for(auto& s:slots)for(auto* b:{s.instances,s.materials,s.lights,s.emitters,s.vertices})c.memory().release(b,MemoryCategory::RayTracing);}
    void load(const GpuScene& g,const SceneStore& store,const MetalTextureManager& t){
        if(done)return;vertices=g.vertices();indices=g.indices();meshes=g.meshInfos();textures=t.referenceTextures();
        for(auto& s:slots){for(auto* b:{s.instances,s.materials,s.lights,s.emitters,s.vertices})c.memory().release(b,MemoryCategory::RayTracing);s.instances=lighting::buffer(c,std::max<u64>(1,store.slotCapacity())*sizeof(GPUInstance),"Reference WORLD instances",true);s.materials=lighting::buffer(c,std::max<size_t>(1,store.materials().size())*sizeof(GPUMaterial),"Reference materials",true);s.vertices=lighting::buffer(c,std::max<size_t>(1,g.vertices().size())*sizeof(GPUVertex),"Reference current raster geometry",true);s.lights=s.emitters=nullptr;s.recorded=false;}
        ++graphVersion;
    }
    void prepare(const ShadowPasses::Frame& f,ReferenceCamera camera,std::span<const GPULight> lights,const SceneStore& store,DirectLightingPasses* di){
        const bool was=requested;requested=!done && f.index==captureFrame;slot=f.slot;frame=f;direct=di;
        if(was!=requested)++graphVersion;if(!requested)return;
        auto& s=slots[slot];s.instanceCount=store.slotCapacity();s.materialCount=store.materials().size();s.lightCount=di?di->lightCount():0;s.camera=camera;s.frame=f.index;s.punctual.assign(lights.begin(),lights.end());
        if(s.instances->length()<u64(s.instanceCount)*sizeof(GPUInstance)||s.materials->length()<u64(s.materialCount)*sizeof(GPUMaterial))throw std::logic_error("Reference snapshot capacity grew before capture");
        if(s.lightCount){s.lights=lighting::buffer(c,u64(s.lightCount)*sizeof(GPUSampledLight),"Reference world sampled lights",true);s.emitters=lighting::buffer(c,u64(s.lightCount)*sizeof(GPUEmissiveSurface),"Reference emitter metadata",true);}++graphVersion;
    }
    void add(rg::RenderGraph& g){using namespace rg;if(!requested)return;auto& s=slots[slot];instanceRef=g.importBuffer("Reference world instance snapshot",{s.instances->length()},ImportPerFrame|ImportOutput);materialRef=g.importBuffer("Reference material snapshot",{s.materials->length()},ImportPerFrame|ImportOutput);vertexRef=g.importBuffer("Reference raster geometry snapshot",{s.vertices->length()},ImportPerFrame|ImportOutput);
        if(s.lightCount){lightRef=g.importBuffer("Reference sampled lights snapshot",{s.lights->length()},ImportPerFrame|ImportOutput);emitterRef=g.importBuffer("Reference emitter records snapshot",{s.emitters->length()},ImportPerFrame|ImportOutput);}else{lightRef={};emitterRef={};}
        g.addPass("Offline reference same-frame snapshot",PassType::Blit,[&](PassBuilder& b){b.read(renderer.dataRef(),Usage::CopySrc,StageBlit);if(rt && rt->geometryBufferRef().valid())b.read(rt->geometryBufferRef(),Usage::CopySrc,StageBlit);instanceRef=b.write(instanceRef,Usage::CopyDst,StageBlit);materialRef=b.write(materialRef,Usage::CopyDst,StageBlit);vertexRef=b.write(vertexRef,Usage::CopyDst,StageBlit);if(s.lightCount){b.read(direct->lightsRef(),Usage::CopySrc,StageBlit);b.read(direct->emittersRef(),Usage::CopySrc,StageBlit);lightRef=b.write(lightRef,Usage::CopyDst,StageBlit);emitterRef=b.write(emitterRef,Usage::CopyDst,StageBlit);}},[this](PassContext& ctx){auto& s=slots[slot];auto* e=static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());if(s.instanceCount)e->copyFromBuffer(renderer.buffers().instances(),0,s.instances,0,u64(s.instanceCount)*sizeof(GPUInstance));if(!vertices.empty())e->copyFromBuffer(renderer.vertexBuffer(),0,s.vertices,0,vertices.size()*sizeof(GPUVertex));if(s.materialCount)e->copyFromBuffer(renderer.buffers().materials(),0,s.materials,0,u64(s.materialCount)*sizeof(GPUMaterial));if(s.lightCount){e->copyFromBuffer(direct->lightsBuffer(),0,s.lights,0,u64(s.lightCount)*sizeof(GPUSampledLight));e->copyFromBuffer(direct->emittersBuffer(),0,s.emitters,0,u64(s.lightCount)*sizeof(GPUEmissiveSurface));}s.recorded=true;});
    }
    void bind(MetalGraphExecutor& e){if(!requested)return;auto& s=slots[slot];e.bindBuffer(instanceRef,s.instances);e.bindBuffer(materialRef,s.materials);e.bindBuffer(vertexRef,s.vertices);if(s.lightCount){e.bindBuffer(lightRef,s.lights);e.bindBuffer(emitterRef,s.emitters);}}
    void consume(u32 i){auto& s=slots.at(i);if(!s.recorded || done)return;if(c.frameEvent()->signaledValue()<=s.frame)throw std::logic_error("Offline reference readback before completion");
        std::vector<ReferenceAreaLight> area;
        if(s.lightCount){auto* lights=static_cast<const GPUSampledLight*>(s.lights->contents());auto* records=static_cast<const GPUEmissiveSurface*>(s.emitters->contents());for(u32 k=0;k<s.lightCount;++k){const auto& l=lights[k];ReferenceAreaLight a;a.id=l.id;a.generation=l.generation;a.type=l.type;a.flags=l.flags;a.range=l.range;a.radius=l.radius;a.innerCone=l.innerCone;a.outerCone=l.outerCone;for(u32 v=0;v<3;++v){a.position[v]=l.position[v];a.axisU[v]=l.axisU[v];a.axisV[v]=l.axisV[v];a.emission[v]=l.emission[v];}if(records[k].valid){a.materialIndex=records[k].materialIndex;std::memcpy(a.uv0,records[k].uv0,8);std::memcpy(a.uv1,records[k].uv1,8);std::memcpy(a.uv2,records[k].uv2,8);}area.push_back(a);}}
        OfflineReferenceScene snap;snap.vertices=std::span(static_cast<const GPUVertex*>(s.vertices->contents()),vertices.size());snap.indices=indices;snap.meshes=meshes;snap.worldInstances=std::span(static_cast<const GPUInstance*>(s.instances->contents()),s.instanceCount);snap.materials=std::span(static_cast<const GPUMaterial*>(s.materials->contents()),s.materialCount);snap.lights=s.punctual;snap.sampledLights=area;snap.textures=textures;snap.camera=s.camera;snap.sunAngularRadius=0.00465f;snap.frame=s.frame;
        const auto result=exportOfflineReference(snap,destination);if(!result.ok)throw std::runtime_error("Offline reference export: "+result.error);done=true;s.recorded=false;++graphVersion;
    }
};
ReferenceSnapshot::ReferenceSnapshot(MetalContext& c,SceneRenderer& r,const std::string& d,u64 frame,AccelerationStructures* a):impl_(std::make_unique<Impl>(c,r,d,frame,a)){}
ReferenceSnapshot::~ReferenceSnapshot()=default;
void ReferenceSnapshot::loadScene(const GpuScene& g,const SceneStore& s,const MetalTextureManager& t){impl_->load(g,s,t);}
void ReferenceSnapshot::prepareFrame(const ShadowPasses::Frame& f,ReferenceCamera camera,std::span<const GPULight> l,const SceneStore& s,DirectLightingPasses* d){impl_->prepare(f,camera,l,s,d);}
void ReferenceSnapshot::addToGraph(rg::RenderGraph& g){impl_->add(g);}void ReferenceSnapshot::bindFrame(MetalGraphExecutor& e){impl_->bind(e);}void ReferenceSnapshot::consume(u32 s){impl_->consume(s);}u64 ReferenceSnapshot::version()const{return impl_->graphVersion;}
} // namespace phosphor
