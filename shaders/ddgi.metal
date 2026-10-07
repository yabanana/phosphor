#include "gi_common.h"

// ddgi_trace: AS0 params1 states2 rays3 IFT4 instances5 RT meshes6 vertices7
// RT indices8 materials9 textures10 GPULight11 candidates12 sampled lights13
// traceExtra14; previous irradiance texture0/moments texture1.
// Every consumer links rt_alpha_generic and owns an IFT of THIS pipeline.
kernel void ddgi_trace(instance_acceleration_structure as [[buffer(0)]],
                       constant GPUProbeGridParams& p [[buffer(1)]],
                       const device GPUProbeState* states [[buffer(2)]],
                       device GPUProbeRay* rays [[buffer(3)]],
                       intersection_function_table<triangle_data,instancing> ift [[buffer(4)]],
                       const device GPUInstance* instances [[buffer(5)]],
                       const device GPURtMesh* meshes [[buffer(6)]],
                       const device GPUVertex* vertices [[buffer(7)]],
                       const device uint* indices [[buffer(8)]],
                       const device GPUMaterial* materials [[buffer(9)]],
                       const device DITextureHandle* textures [[buffer(10)]],
                       const device GPULight* lights [[buffer(11)]],
                       device GPUGiReservoir* candidates [[buffer(12)]],
                       const device GPUSampledLight* sampled [[buffer(13)]],
                       constant GPUProbeTraceExtra& extra [[buffer(14)]],
                       const device GPUEmissiveSurface* emissive [[buffer(16)]],
                       texture2d<float> irradiance [[texture(0)]],texture2d<float> moments [[texture(1)]],
                       uint tid [[thread_position_in_grid]]) {
    uint count=p.countX*p.countY*p.countZ*p.raysPerProbe;
    if(tid>=count) return;
    uint probe=tid/p.raysPerProbe, index=tid%p.raysPerProbe;
    float3 origin=giProbePosition(probe,p,states),direction=giProbeDirection(index,p.raysPerProbe,p.frameIndex);
    GPURtRay ray{}; ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;
    ray.dx=direction.x;ray.dy=direction.y;ray.dz=direction.z;
    ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;
    RtPayload payload{};
    GPURtHit hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);
    GPUProbeRay output{};output.direction[0]=direction.x;output.direction[1]=direction.y;output.direction[2]=direction.z;
    output.distance=p.maxDistance;
    float3 L(0.0f); // no direct environment in the INDIRECT atlas
    GPUGiReservoir candidate{};
    float3 point,normal;GiMaterial material;
    if(hit.hit && rtSurface(hit,instances,meshes,vertices,indices,p.slotCount,p.meshCount,point,normal) &&
        giMaterial(hit,p,instances,meshes,vertices,indices,materials,textures,material)) {
        const bool backface=hit.frontFacing==0u && !material.doubleSided;
        output.backface=backface?1u:0u;output.distance=backface?-max(hit.t,1e-8f):hit.t;
        if(dot(normal,-direction)<0.0f) normal=-normal;
        uint seed=giHash(tid^giHash(p.frameIndex)^extra.frameSeed);
        L=backface?float3(0.0f):giSecondaryRadiance(point,normal,material,as,ift,instances,p,lights,materials,textures,sampled,emissive,extra,seed,states,irradiance,moments);
        if(!backface) {
            for(uint i=0;i<3u;++i) {candidate.position[i]=point[i];candidate.normal[i]=normal[i];
                candidate.radiance[i]=L[i];candidate.sourcePosition[i]=origin[i];candidate.sourceNormal[i]=direction[i];}
            candidate.proposalSolidAngle=1.0f/(4.0f*M_PI_F);
            candidate.proposalArea=candidate.proposalSolidAngle*max(0.0f,dot(normal,-direction))/max(hit.t*hit.t,1e-12f);
            candidate.flags=GI_SAMPLE_VALID;candidate.slot=hit.slot;candidate.instanceGeneration=hit.generation;
            candidate.geometryRevision=p.geometryRevision;candidate.lightRevision=p.lightRevision;
            candidate.materialRevision=p.materialRevision;candidate.viewRevision=p.viewRevision;
        }
    }
    if(!all(isfinite(L))) {L=0.0f;candidate.flags=0u;}
    // Cache candidate keeps full outgoing Le+reflected radiance. Atlas removes
    // Le at this first segment because F11 samples direct emissive separately.
    if(candidate.flags&GI_SAMPLE_VALID) L=max(L-material.emissive,0.0f);
    output.radiance[0]=L.x;output.radiance[1]=L.y;output.radiance[2]=L.z;
    rays[tid]=output;candidates[tid]=candidate;
}

// One lane per probe. Inactive probes remain scheduled for traces so moving
// walls/relocation recover; no inactive-probe trace starvation.
kernel void ddgi_classify(constant GPUProbeGridParams& p [[buffer(0)]],
                          device GPUProbeState* states [[buffer(1)]],
                          const device GPUProbeRay* rays [[buffer(2)]],uint probe [[thread_position_in_grid]]) {
    if(probe>=p.countX*p.countY*p.countZ) return;
    GPUProbeState s=states[probe];
    if(p.reset || s.generation!=p.generation) {s={};s.generation=p.generation;}
    float backDistance=p.maxDistance,frontDistance=p.maxDistance;float3 escape=0.0f,away=0.0f;
    uint backfaces=0u;
    for(uint i=0;i<p.raysPerProbe;++i) {
        const device GPUProbeRay& r=rays[probe*p.raysPerProbe+i];
        float3 d(r.direction[0],r.direction[1],r.direction[2]);
        if(r.backface) {++backfaces;if(-r.distance<backDistance) {backDistance=-r.distance;escape=d;}}
        else if(r.distance<frontDistance) {frontDistance=r.distance;away=-d;}
    }
    bool inside=float(backfaces)/float(p.raysPerProbe)>p.backfaceThreshold;
    float3 old(s.offset[0],s.offset[1],s.offset[2]),delta=0.0f;
    if(inside && backDistance<p.maxDistance) delta=escape*min(p.relocationStep,backDistance+p.minFrontDistance);
    else if(frontDistance<p.minFrontDistance) delta=away*min(p.relocationStep,p.minFrontDistance-frontDistance);
    float3 next=old+delta;
    float scaled=length(next/float3(p.spacing[0],p.spacing[1],p.spacing[2]));
    if(scaled>p.maxRelocation && scaled>0.0f) next*=p.maxRelocation/scaled;
    float moved=length(next-old);
    s.relocationTravel+=moved;s.offset[0]=next.x;s.offset[1]=next.y;s.offset[2]=next.z;
    s.state=(inside||moved>1e-6f)?GI_PROBE_INACTIVE:GI_PROBE_ACTIVE;
    s.age=s.state==GI_PROBE_ACTIVE?min(s.age,0xfffffffeu)+1u:0u;
    states[probe]=s;
}

inline uint2 giFoldTile(uint2 local,uint side) {
    int x=int(local.x)-1,y=int(local.y)-1,n=int(side);
    if(x<0) {x=-x-1;y=n-1-y;} if(x>=n) {x=2*n-1-x;y=n-1-y;}
    if(y<0) {y=-y-1;x=n-1-x;} if(y>=n) {y=2*n-1-y;x=n-1-x;}
    return uint2(clamp(x,0,n-1),clamp(y,0,n-1));
}
inline bool giTile(uint2 pixel,uint side,constant GPUProbeGridParams& p,thread uint& probe,thread float3& n) {
    uint stride=side+2u;
    if(pixel.x>=p.countX*stride || pixel.y>=p.countY*p.countZ*stride) return false;
    uint2 tile=pixel/stride;
    probe=tile.y*p.countX+tile.x;
    n=giOctDecode((float2(giFoldTile(pixel%stride,side))+0.5f)/float(side)*2.0f-1.0f);
    return true;
}
// Dispatch max(irradianceWidth,distanceWidth) x max(...height). Previous and
// next are DISTINCT graph resources. Each border is written by its own lane,
// avoiding races between interior and border-copy kernels.
kernel void ddgi_blend(constant GPUProbeGridParams& p [[buffer(0)]],
                       const device GPUProbeState* states [[buffer(1)]],
                       const device GPUProbeRay* rays [[buffer(2)]],
                       texture2d<float,access::read> previousIrradiance [[texture(0)]],
                       texture2d<float,access::read> previousMoments [[texture(1)]],
                       texture2d<float,access::write> nextIrradiance [[texture(2)]],
                       texture2d<float,access::write> nextMoments [[texture(3)]],uint2 pixel [[thread_position_in_grid]]) {
    uint probe;float3 n;
    if(giTile(pixel,p.irradianceTexels,p,probe,n)) {
        const device GPUProbeState& s=states[probe];
        bool active=s.state==GI_PROBE_ACTIVE && s.generation==p.generation;
        float3 sum=0.0f;
        if(active) for(uint i=0;i<p.raysPerProbe;++i) {
            const device GPUProbeRay& r=rays[probe*p.raysPerProbe+i];
            if(!r.backface) sum+=float3(r.radiance[0],r.radiance[1],r.radiance[2])*max(0.0f,dot(n,float3(r.direction[0],r.direction[1],r.direction[2])));
        }
        float history=active && !p.reset && s.age>1u?p.hysteresis:0.0f;
        float3 fresh=sum*(4.0f*M_PI_F/float(p.raysPerProbe));
        float3 E=history>0.0f?mix(fresh,previousIrradiance.read(pixel).rgb,history):fresh;
        nextIrradiance.write(float4(E,active?1.0f:0.0f),pixel);
    }
    if(giTile(pixel,p.distanceTexels,p,probe,n)) {
        const device GPUProbeState& s=states[probe];bool active=s.state==GI_PROBE_ACTIVE && s.generation==p.generation;
        float2 sum=0.0f;float weights=0.0f;
        if(active) for(uint i=0;i<p.raysPerProbe;++i) {
            const device GPUProbeRay& r=rays[probe*p.raysPerProbe+i];
            float w=pow(max(0.0f,dot(n,float3(r.direction[0],r.direction[1],r.direction[2]))),50.0f);
            float distance=max(0.0f,r.distance);sum+=float2(distance,distance*distance)*w;weights+=w;
        }
        float history=active && !p.reset && s.age>1u?p.hysteresis:0.0f;
        float2 fresh=weights>1e-20f?sum/weights:float2(0.0f);
        float2 m=history>0.0f?mix(fresh,previousMoments.read(pixel).xy,history):fresh;
        nextMoments.write(float4(m,0.0f,active?1.0f:0.0f),pixel);
    }
}

kernel void ddgi_resolve(constant GPUProbeGridParams& p [[buffer(0)]],
                         const device GPUProbeState* states [[buffer(1)]],
                         texture2d<float,access::read> positions [[texture(0)]],
                         texture2d<float,access::read> normals [[texture(1)]],
                         texture2d<float> irradiance [[texture(2)]],texture2d<float> moments [[texture(3)]],
                         texture2d<float,access::write> indirectIrradiance [[texture(4)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    float4 point=positions.read(pixel),normal=normals.read(pixel);
    float3 E=(point.w>0.0f && normal.w>0.0f)?giIrradiance(point.xyz,normal.xyz,p,states,irradiance,moments,true):float3(0.0f);
    indirectIrradiance.write(float4(E,1.0f),pixel);
}
