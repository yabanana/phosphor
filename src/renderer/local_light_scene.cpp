#include "renderer/local_light_scene.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include <cstring>
#include <cmath>
namespace phosphor {
void LocalLightScene::rebuild(const GpuScene& scene,const SceneStore& store,std::span<const GPULight> punctual) {
    lights.clear();emitters.clear();weights_.clear();
    auto append=[&](GPUSampledLight l,GPUEmissiveSurface e={}) {l.id=lights.size();lights.push_back(l);emitters.push_back(e);};
    for(const auto& l:punctual)if(l.type==LIGHT_POINT || l.type==LIGHT_SPOT)append(di::fromPunctual(l,lights.size(),1));
    for(const auto& l:scene.sampledLights())append(l);
    const auto materials=store.materials();const auto instances=store.instances();
    for(u32 slot=0;slot<instances.size();++slot) {
        const auto& i=instances[slot];
        if(!(i.flags&INSTANCE_FLAG_VALID) || !(i.flags&1u) || i.meshIndex>=scene.meshInfos().size() || i.materialIndex>=materials.size())continue;
        const auto& material=materials[i.materialIndex];
        if(material.emissive[0]<=0 && material.emissive[1]<=0 && material.emissive[2]<=0)continue;
        const auto& mesh=scene.meshInfos()[i.meshIndex];
        for(u32 triangle=0;triangle<mesh.indexCount/3;++triangle) {
            const auto& a=scene.vertices()[mesh.vertexOffset+scene.indices()[mesh.indexOffset+triangle*3]];
            const auto& b=scene.vertices()[mesh.vertexOffset+scene.indices()[mesh.indexOffset+triangle*3+1]];
            const auto& c=scene.vertices()[mesh.vertexOffset+scene.indices()[mesh.indexOffset+triangle*3+2]];
            GPUSampledLight l{};l.generation=i.generation;l.type=DI_LIGHT_TRIANGLE;
            l.flags=(material.flags&MATERIAL_FLAG_DOUBLE_SIDED)?DI_LIGHT_TWO_SIDED:0;
            const float p0[]={a.px,a.py,a.pz},p1[]={b.px,b.py,b.pz},p2[]={c.px,c.py,c.pz};
            for(u32 k=0;k<3;++k){l.position[k]=p0[k];l.axisU[k]=p1[k]-p0[k];l.axisV[k]=p2[k]-p0[k];l.emission[k]=material.emissive[k];}
            GPUEmissiveSurface e{};std::memcpy(e.p0,p0,12);std::memcpy(e.p1,p1,12);std::memcpy(e.p2,p2,12);
            e.instanceSlot=slot;e.instanceGeneration=i.generation;e.materialIndex=i.materialIndex;e.valid=1;
            e.uv0[0]=a.u;e.uv0[1]=a.v;e.uv1[0]=b.u;e.uv1[1]=b.v;e.uv2[0]=c.u;e.uv2[1]=c.v;
            e.vertex0=mesh.vertexOffset+scene.indices()[mesh.indexOffset+triangle*3];
            e.vertex1=mesh.vertexOffset+scene.indices()[mesh.indexOffset+triangle*3+1];
            e.vertex2=mesh.vertexOffset+scene.indices()[mesh.indexOffset+triangle*3+2];e.geometryValid=1;
            append(l,e);
        }
    }
    u64 hash=1469598103934665603ull;
    auto hashBytes=[&](const void* data,size_t bytes){const auto* p=static_cast<const unsigned char*>(data);for(size_t k=0;k<bytes;++k){hash^=p[k];hash*=1099511628211ull;}};
    hashBytes(lights.data(),lights.size()*sizeof(GPUSampledLight));hashBytes(emitters.data(),emitters.size()*sizeof(GPUEmissiveSurface));
    // Proposal weights may be local-area based: they are sampling heuristics,
    // while exact transformed area/PDF is evaluated by the shader integrand.
    for(const auto& l:lights)weights_.push_back(di::powerWeight(l));
    if(hash!=radianceHash_){++radianceRevision;radianceHash_=hash;}
    // Rigid light movement keeps the discrete/area domain and can reuse DI.
    // The radiance cache still sees the FULL content revision above.
    hash=1469598103934665603ull;
    for(const auto& l:lights) {
        hashBytes(&l.id,16);hashBytes(l.emission,12);hashBytes(&l.range,4);
        hashBytes(&l.radius,4);hashBytes(&l.innerCone,4);hashBytes(&l.outerCone,4);
        const double a=di::area(l);hashBytes(&a,sizeof a);
    }
    hashBytes(emitters.data(),emitters.size()*sizeof(GPUEmissiveSurface));
    if(hash!=hash_){++revision;hash_=hash;}alias.rebuild(weights_,revision);
}
} // namespace phosphor
