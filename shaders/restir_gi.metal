#include "gi_cache_common.h"

// Diffuse-only area-measure reconnection. Initial q_A = q_omega*cos_y/r^2.
// t(y) = luminance(L_y * cos_x*cos_y/(pi*r^2)); output irradiance=pi*f_A*W.
// Moving/transformed secondary geometry is rejected by global revisions and
// instance generations. Area->area reconnection J=1 for fixed world surfaces.
// Basic reuse uses bounded M normalization and is EXPLICITLY BIASED when source
// visibility supports differ. Fresh-only mode is independent cosine-path RIS.
inline float3 giReservoirPoint(GPUGiReservoir r) {return float3(r.position[0],r.position[1],r.position[2]);}
inline float3 giReservoirNormal(GPUGiReservoir r) {return float3(r.normal[0],r.normal[1],r.normal[2]);}
inline float3 giAreaIntegrand(float3 x,float3 n,GPUGiReservoir r) {
    float3 delta=giReservoirPoint(r)-x;float r2=dot(delta,delta);
    if(!(r.flags&GI_SAMPLE_VALID) || r2<1e-12f || !isfinite(r2)) return 0.0f;
    float3 wi=delta*rsqrt(r2),L(r.radiance[0],r.radiance[1],r.radiance[2]);
    float cosX=max(0.0f,dot(n,wi)),cosY=max(0.0f,dot(giReservoirNormal(r),-wi));
    return max(L,0.0f)*(cosX*cosY/(M_PI_F*r2));
}
inline void giReservoirFinalize(thread GPUGiReservoir& r) {
    r.W=r.M && r.target>0.0f?r.weightSum/(float(r.M)*r.target):0.0f;
    if(!isfinite(r.W) || r.W<=0.0f) {r.W=0.0f;r.flags=0u;}
}
inline bool giReusable(GPUGiReservoir r,constant GPUProbeGridParams& p,const device GPUInstance* instances) {
    return !p.reset && (r.flags&GI_SAMPLE_VALID) && r.slot<p.slotCount && r.age<32u &&
        r.geometryRevision==p.geometryRevision && r.lightRevision==p.lightRevision &&
        r.materialRevision==p.materialRevision && r.viewRevision==p.viewRevision &&
        instances[r.slot].generation==r.instanceGeneration && (instances[r.slot].flags&INSTANCE_FLAG_VALID);
}
inline bool giReconnectVisible(float3 x,float3 n,GPUGiReservoir r,instance_acceleration_structure as,
                                intersection_function_table<triangle_data,instancing> ift,
                                const device GPUInstance* instances,constant GPUProbeGridParams& p) {
    float3 origin=rtOffsetRay(x,n),delta=giReservoirPoint(r)-origin;
    float distance=length(delta);if(distance<1e-6f) return false;
    float3 wi=delta/distance;GPURtRay ray{};
    ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=wi.x;ray.dy=wi.y;ray.dz=wi.z;
    // Pull the endpoint toward its incoming hemisphere using the same W&B rule.
    float3 endpoint=rtOffsetRay(giReservoirPoint(r),giReservoirNormal(r));
    ray.tmax=max(0.0f,distance-length(endpoint-giReservoirPoint(r))-1e-5f);
    ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_SHADOW;RtPayload payload{};
    return rtTrace(ray,as,ift,instances,p.slotCount,payload).hit==0u;
}
inline void giReservoirMerge(thread GPUGiReservoir& dst,GPUGiReservoir source,float target,bool visible,float random) {
    if(!(source.flags&GI_SAMPLE_VALID) || !source.M || source.W<=0.0f) return;
    uint m=min(source.M,32u),total=dst.M+m;
    if(total<dst.M) return;
    dst.M=total;
    float weight=visible?target*source.W*float(m):0.0f,sum=dst.weightSum+weight;
    if(!isfinite(weight) || !isfinite(sum) || weight<=0.0f) return;
    if(random*sum<weight) {dst=source;dst.M=total;dst.target=target;dst.age=source.age+1u;}
    dst.weightSum=sum;
}

// gi_candidates mirrors ddgi_trace bindings except states2, candidates3, cache15.
// Textures position0/geometricNormal1/previous irradiance2/previous moments3.
// Cache mode executes this then gi_shade; ReSTIR mode inserts temporal/spatial.
kernel void gi_candidates(instance_acceleration_structure as [[buffer(0)]],
                           constant GPUProbeGridParams& p [[buffer(1)]],
                           const device GPUProbeState* states [[buffer(2)]],
                           device GPUGiReservoir* candidates [[buffer(3)]],
                           intersection_function_table<triangle_data,instancing> ift [[buffer(4)]],
                           const device GPUInstance* instances [[buffer(5)]],
                           const device GPURtMesh* meshes [[buffer(6)]],
                           const device GPUVertex* vertices [[buffer(7)]],const device uint* indices [[buffer(8)]],
                           const device GPUMaterial* materials [[buffer(9)]],const device DITextureHandle* textures [[buffer(10)]],
                           const device GPULight* lights [[buffer(11)]],
                           const device GPUSampledLight* sampled [[buffer(13)]],
                           constant GPUProbeTraceExtra& extra [[buffer(14)]],
                       const device GPUEmissiveSurface* emissive [[buffer(16)]],
                           const device GPURadianceCacheEntry* cache [[buffer(15)]],
                           texture2d<float,access::read> positions [[texture(0)]],
                           texture2d<float,access::read> normals [[texture(1)]],
                           texture2d<float> irradiance [[texture(2)]],texture2d<float> moments [[texture(3)]],
                           uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const uint index=pixel.y*p.width+pixel.x;GPUGiReservoir r{};
    float4 point=positions.read(pixel),normal=normals.read(pixel);
    if(point.w<=0.0f || normal.w<=0.0f || dot(normal.xyz,normal.xyz)<1e-20f) {candidates[index]=r;return;}
    float3 n=normalize(normal.xyz);
    uint seed=giHash(index^giHash(p.frameIndex)^extra.frameSeed);
    float2 uv(giRandom(seed),giRandom(seed));
    float radius=sqrt(uv.x),angle=2.0f*M_PI_F*uv.y;
    float3 t=normalize(cross(abs(n.z)<0.999f?float3(0,0,1):float3(0,1,0),n)),b=cross(n,t);
    float3 wi=t*(radius*cos(angle))+b*(radius*sin(angle))+n*sqrt(max(0.0f,1.0f-uv.x));
    float3 origin=rtOffsetRay(point.xyz,n);GPURtRay ray{};
    ray.ox=origin.x;ray.oy=origin.y;ray.oz=origin.z;ray.dx=wi.x;ray.dy=wi.y;ray.dz=wi.z;
    ray.tmax=p.maxDistance;ray.mask=RT_MASK_INDIRECT;ray.type=RT_PROBE_DIFFUSE;RtPayload payload{};
    GPURtHit hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);
    float3 y=point.xyz+wi*p.maxDistance,ny=-wi,L(0.0f); // direct environment excluded
    GiMaterial material;
    bool valid=true;
    if(hit.hit) {
        valid=rtSurface(hit,instances,meshes,vertices,indices,p.slotCount,p.meshCount,y,ny) &&
            giMaterial(hit,p,instances,meshes,vertices,indices,materials,textures,material);
        if(valid && (hit.frontFacing || material.doubleSided)) {
            if(dot(ny,-wi)<0.0f) ny=-ny;
            bool cached=(p.mode!=GI_MODE_DDGI) && giCacheLookup(y,ny,point.xyz-y,p,cache,L);
            if(!cached) L=giSecondaryRadiance(y,ny,material,as,ift,instances,p,lights,materials,textures,sampled,emissive,extra,seed,states,irradiance,moments);
            L=max(L-material.emissive,0.0f); // first-segment Le belongs to direct F11
        } else L=0.0f;
    }
    r.M=1u; // even zero contribution/backface/miss counts as a sampled path
    r.slot=hit.hit?hit.slot:~0u;r.instanceGeneration=hit.generation;
    for(uint k=0;k<3u;++k) {r.position[k]=y[k];r.normal[k]=ny[k];r.radiance[k]=L[k];r.sourcePosition[k]=point[k];r.sourceNormal[k]=n[k];}
    r.flags=valid?GI_SAMPLE_VALID:0u;
    float3 delta=y-point.xyz;float r2=dot(delta,delta),cosY=max(0.0f,dot(ny,-normalize(delta)));
    r.proposalSolidAngle=max(0.0f,dot(n,normalize(delta)))/M_PI_F;
    r.proposalArea=r.proposalSolidAngle*cosY/max(r2,1e-12f);r.sourceProposalArea=r.proposalArea;
    r.target=diLuminance(giAreaIntegrand(point.xyz,n,r));
    r.weightSum=r.proposalArea>0.0f?r.target/r.proposalArea:0.0f;
    r.geometryRevision=p.geometryRevision;r.lightRevision=p.lightRevision;r.materialRevision=p.materialRevision;r.viewRevision=p.viewRevision;
    giReservoirFinalize(r);candidates[index]=r;
}

// Reprojection uses current-to-previous motion in PIXELS (MetalFX contract).
// Buffers AS0/params1/fresh2/history3/output4/IFT5/instances6; textures
// positions0/normals1/motion2. Stored source point/normal are previous guides.
kernel void gi_temporal(instance_acceleration_structure as [[buffer(0)]],constant GPUProbeGridParams& p [[buffer(1)]],
                         const device GPUGiReservoir* fresh [[buffer(2)]],const device GPUGiReservoir* history [[buffer(3)]],
                         device GPUGiReservoir* output [[buffer(4)]],
                         intersection_function_table<triangle_data,instancing> ift [[buffer(5)]],
                         const device GPUInstance* instances [[buffer(6)]],
                         texture2d<float,access::read> positions [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
                         texture2d<float,access::read> motion [[texture(2)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    uint id=pixel.y*p.width+pixel.x;GPUGiReservoir r=fresh[id];
    float3 x=positions.read(pixel).xyz,n=normals.read(pixel).xyz;
    float2 previous=float2(pixel)+motion.read(pixel).xy;
    if(!p.reset && all(isfinite(previous)) && all(previous>=0.0f) && previous.x<float(p.width) && previous.y<float(p.height)) {
        uint2 q=uint2(floor(previous+0.5f));
        if(q.x<p.width && q.y<p.height) {
            GPUGiReservoir old=history[q.y*p.width+q.x];
            float3 source(old.sourcePosition[0],old.sourcePosition[1],old.sourcePosition[2]),sn(old.sourceNormal[0],old.sourceNormal[1],old.sourceNormal[2]);
            float tolerance=min(min(p.spacing[0],p.spacing[1]),p.spacing[2])*0.05f;
            if(giReusable(old,p,instances) && length(source-x)<tolerance && dot(sn,n)>0.95f) {
                float target=diLuminance(giAreaIntegrand(x,n,old));
                bool visible=target>0.0f && giReconnectVisible(x,n,old,as,ift,instances,p);
                uint seed=giHash(id^giHash(p.frameIndex)^0x6428317du);
                giReservoirMerge(r,old,target,visible,giRandom(seed));
            }
        }
    }
    for(uint k=0;k<3u;++k) {r.sourcePosition[k]=x[k];r.sourceNormal[k]=n[k];}
    giReservoirFinalize(r);output[id]=r;
}

// Separate input/output buffers: neighborhood reads never race writes.
kernel void gi_spatial(instance_acceleration_structure as [[buffer(0)]],constant GPUProbeGridParams& p [[buffer(1)]],
                        const device GPUGiReservoir* input [[buffer(2)]],device GPUGiReservoir* output [[buffer(3)]],
                        intersection_function_table<triangle_data,instancing> ift [[buffer(4)]],
                        const device GPUInstance* instances [[buffer(5)]],
                        texture2d<float,access::read> positions [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
                        uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    uint id=pixel.y*p.width+pixel.x;GPUGiReservoir r=input[id];
    float4 x4=positions.read(pixel),n4=normals.read(pixel);float3 x=x4.xyz,n=n4.xyz;
    if(x4.w>0.0f && n4.w>0.0f) for(uint k=0;k<4u;++k) {
        int2 offset=k==0u?int2(-1,0):k==1u?int2(1,0):k==2u?int2(0,-1):int2(0,1),q=int2(pixel)+offset;
        if(any(q<0) || q.x>=int(p.width) || q.y>=int(p.height)) continue;
        GPUGiReservoir other=input[uint(q.y)*p.width+uint(q.x)];
        float3 source(other.sourcePosition[0],other.sourcePosition[1],other.sourcePosition[2]),sn(other.sourceNormal[0],other.sourceNormal[1],other.sourceNormal[2]);
        float tolerance=min(min(p.spacing[0],p.spacing[1]),p.spacing[2])*0.25f;
        if(!giReusable(other,p,instances) || length(source-x)>tolerance || dot(sn,n)<0.95f) continue;
        float target=diLuminance(giAreaIntegrand(x,n,other));
        bool visible=target>0.0f && giReconnectVisible(x,n,other,as,ift,instances,p);
        uint seed=giHash(id^giHash(p.frameIndex)^giHash(k+0x17a52be7u));
        giReservoirMerge(r,other,target,visible,giRandom(seed));
    }
    for(uint k=0;k<3u;++k) {r.sourcePosition[k]=x[k];r.sourceNormal[k]=n[k];}
    giReservoirFinalize(r);output[id]=r;
}
// Irradiance output, NOT HDR and NOT final primary reflected radiance.
kernel void gi_shade(constant GPUProbeGridParams& p [[buffer(0)]],const device GPUGiReservoir* reservoirs [[buffer(1)]],
                      const device GPUProbeState* states [[buffer(2)]],
                      texture2d<float,access::read> positions [[texture(0)]],texture2d<float,access::read> normals [[texture(1)]],
                      texture2d<float> irradiance [[texture(2)]],texture2d<float> moments [[texture(3)]],
                      texture2d<float,access::write> indirectIrradiance [[texture(4)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    float4 x=positions.read(pixel),n=normals.read(pixel);float3 E=0.0f;
    if(x.w>0.0f && n.w>0.0f) {
        GPUGiReservoir r=reservoirs[pixel.y*p.width+pixel.x];
        if(r.M==0u) E=giIrradiance(x.xyz,n.xyz,p,states,irradiance,moments,true);
        else if(r.flags&GI_SAMPLE_VALID) E=giAreaIntegrand(x.xyz,n.xyz,r)*r.W*M_PI_F;
    }
    if(!all(isfinite(E))) E=0.0f;
    indirectIrradiance.write(float4(E,1.0f),pixel);
}
