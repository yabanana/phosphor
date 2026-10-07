// F10: pre-resolve geometric guides, conservative full-scene caster selection,
// four-map CSM/orthographic PCSS, one-ray solar visibility and temporal moments.
// Every buffer/texture index is documented in renderer/shadow_layout.h.
#define PHOSPHOR_RT_NO_ALPHA_FUNCTION 1
#include "rt_common.h"
#include "renderer/shadow_math.h"
#include "renderer/shadow_layout.h"
#include "renderer/shadow_dispatch.h"
#include "renderer/cull_math.h"
#include "renderer/visibility_math.h"
#include "renderer/meshlet_layout.h"

static float4x4 shadowMatrix(constant float* m) {
    return float4x4(float4(m[0],m[1],m[2],m[3]),float4(m[4],m[5],m[6],m[7]),
                    float4(m[8],m[9],m[10],m[11]),float4(m[12],m[13],m[14],m[15]));
}
static float3 shadowPoint(GPUShadowSurface s) { return float3(s.position[0],s.position[1],s.position[2]); }
static float3 shadowNormal(GPUShadowSurface s) { return float3(s.geometricNormal[0],s.geometricNormal[1],s.geometricNormal[2]); }
static uint shadowHash(uint x) { x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16); }
static float shadowUniform(uint x) { return float(shadowHash(x)>>8)*(1.0f/16777216.0f); }
static void shadowCount(device atomic_uint* counters,uint index,uint value) {
    if(value) atomic_fetch_add_explicit(counters+index,value,memory_order_relaxed);
}
kernel void shadow_clear_counters(device atomic_uint* counters [[buffer(15)]],uint tid [[thread_position_in_grid]]) {
    if(tid<8) atomic_store_explicit(counters+tid,0u,memory_order_relaxed);
}

kernel void shadow_surface_guides(
    constant GPUShadowParams& p [[buffer(1)]],device GPUShadowSurface* surfaces [[buffer(2)]],
    const device GPUInstance* instances [[buffer(3)]],const device GPUMeshInfo* meshes [[buffer(5)]],
    const device GPUVertex* vertices [[buffer(6)]],const device GPUMeshlet* meshlets [[buffer(7)]],
    const device uint* meshletVertices [[buffer(8)]],const device uchar* triangles [[buffer(9)]],
    const device GPUMeshletCandidate* a [[buffer(10)]],const device GPUMeshletCandidate* b [[buffer(11)]],
    device atomic_uint* counters [[buffer(15)]],texture2d<float,access::read> depth [[texture(0)]],
    texture2d<uint,access::read> visibility [[texture(1)]],
    texture2d<float,access::write> worldPosition [[texture(5)]],
    texture2d<float,access::write> geometricNormal [[texture(6)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const uint index=pixel.y*p.width+pixel.x,id=visibility.read(pixel).x;
    GPUShadowSurface s{}; s.slot=~0u; s.primitive=~0u;
    bool valid=id!=VISIBILITY_BACKGROUND && visibilityCluster(id)<2u*p.candidateCapacity;
    if(valid) {
        const uint cluster=visibilityCluster(id);
        const GPUMeshletCandidate candidate=cluster<p.candidateCapacity?a[cluster]:b[cluster-p.candidateCapacity];
        valid=candidate.slot<p.slotCount;
        if(valid) {
            const device GPUInstance& instance=instances[candidate.slot];
            valid=(instance.flags&INSTANCE_FLAG_VALID) && instance.meshIndex<p.meshCount;
            if(valid) {
                const device GPUMeshInfo& mesh=meshes[instance.meshIndex];
                valid=candidate.meshlet>=mesh.meshletOffset && candidate.meshlet-mesh.meshletOffset<mesh.meshletCount;
                if(valid) {
                    const GPUMeshlet m=meshlets[candidate.meshlet];
                    const uint triangle=visibilityTriangle(id);
                    valid=triangle<m.triangleCount;
                    if(valid) {
                        const uint base=m.triangleOffset+3u*triangle;
                        const uint i0=triangles[base],i1=triangles[base+1],i2=triangles[base+2];
                        valid=i0<m.vertexCount && i1<m.vertexCount && i2<m.vertexCount;
                        if(valid) {
                            const float3 w0=rtWorldPoint(instance,rtPosition(vertices[meshletVertices[m.vertexOffset+i0]]));
                            const float3 w1=rtWorldPoint(instance,rtPosition(vertices[meshletVertices[m.vertexOffset+i1]]));
                            const float3 w2=rtWorldPoint(instance,rtPosition(vertices[meshletVertices[m.vertexOffset+i2]]));
                            const float z=depth.read(pixel).x;
                            // Raster depth can lie behind the true primitive by more
                            // than W&B's rounding offset. Reconstruct the receiver from
                            // its V-buffer triangle, using the same perspective weights
                            // as material resolve, instead of launching rays from depth.
                            const float4x4 vp=shadowMatrix(p.viewProjection);
                            const float4 c0=vp*float4(w0,1),c1=vp*float4(w1,1),c2=vp*float4(w2,1);
                            const auto bary=visibilityBarycentrics(c0.x,c0.y,c0.w,c1.x,c1.y,c1.w,
                                c2.x,c2.y,c2.w,float(pixel.x)+0.5f,float(pixel.y)+0.5f,float(p.width),float(p.height));
                            const float3 crossNormal=cross(w1-w0,w2-w0);
                            valid=z>0 && z<=1 && bary.valid && dot(crossNormal,crossNormal)>1e-30f;
                            if(valid) {
                                const float3 world=w0*bary.value[0]+w1*bary.value[1]+w2*bary.value[2];
                                float3 normal=normalize(crossNormal);
                                const float3 eye=float3(p.cameraPosition[0],p.cameraPosition[1],p.cameraPosition[2]);
                                if(dot(normal,eye-world)<0) normal=-normal;
                                const float viewDepth=-(shadowMatrix(p.view)*float4(world,1)).z;
                                valid=all(isfinite(world)) && all(isfinite(normal)) && isfinite(viewDepth) && viewDepth>0;
                                if(valid) {
                                    s.position[0]=world.x; s.position[1]=world.y; s.position[2]=world.z; s.viewDepth=viewDepth;
                                    s.geometricNormal[0]=normal.x; s.geometricNormal[1]=normal.y; s.geometricNormal[2]=normal.z;
                                    s.reverseDepth=z; s.slot=candidate.slot; s.generation=instance.generation;
                                    s.primitive=triangle; s.valid=1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    surfaces[index]=s;
    worldPosition.write(float4(shadowPoint(s),float(s.valid)),pixel);
    geometricNormal.write(float4(shadowNormal(s),float(s.valid)),pixel);
    shadowCount(counters,0,s.valid);
    if(id!=VISIBILITY_BACKGROUND && !s.valid) shadowCount(counters,7,1u);
}

kernel void shadow_caster_flags(constant GPUShadowParams& p [[buffer(1)]],
    const device GPUInstance* instances [[buffer(3)]],const device GPUMeshInfo* meshes [[buffer(5)]],
    device uint* flags [[buffer(14)]],device atomic_uint* counters [[buffer(15)]],uint slot [[thread_position_in_grid]]) {
    if(slot>=p.slotCount) return;
    const device GPUInstance& i=instances[slot];
    uint bits=0;
    if((i.flags&INSTANCE_FLAG_VALID) && (i.flags&2u) && i.meshIndex<p.meshCount) {
        const GPUWorldSphere s=cullWorldSphere(i.modelMatrix,meshes[i.meshIndex].boundingSphere);
        for(uint c=0;c<4;++c) if(shadowCasterIntersects(p.cascades[c],s.x,s.y,s.z,s.r)) bits|=1u<<c;
    }
    if(p.corruption==SHADOW_CORRUPT_CASTER) bits=0u;
    flags[slot]=bits;
    shadowCount(counters,5,bits!=0);
}

struct ShadowDepthOut {
    float4 position [[position]];
    float2 uv;
    uint material [[flat]];
};
static ShadowDepthOut shadowDepthVertex(const device GPUVertex& v,const device GPUInstance& i,
                                        constant GPUShadowParams& p) {
    ShadowDepthOut o;
    o.position=shadowMatrix(p.cascades[p.cascadeIndex].viewProjection)*float4(rtWorldPoint(i,rtPosition(v)),1);
    o.uv=float2(v.u,v.v); o.material=i.materialIndex;
    return o;
}
vertex ShadowDepthOut shadow_depth_vertex(uint vertexId [[vertex_id]],uint instanceID [[instance_id]],
    constant GPUShadowParams& p [[buffer(1)]],const device GPUInstance* instances [[buffer(3)]],
    const device GPUVertex* vertices [[buffer(6)]],const device uint* casterFlags [[buffer(14)]]) {
    const uint slot=shadowDrawSlot(p.casterSlot,instanceID,p.slotCount);
    ShadowDepthOut invalid{}; invalid.position=float4(2,2,0,1);
    if(slot>=p.slotCount || p.cascadeIndex>=4) return invalid;
    const device GPUInstance& i=instances[slot];
    if(!(casterFlags[slot]&(1u<<p.cascadeIndex)) || i.materialIndex>=p.materialCount ||
        !(i.flags&INSTANCE_FLAG_VALID) || !(i.flags&2u) || i.meshIndex!=p.pad) return invalid;
    return shadowDepthVertex(vertices[vertexId],i,p); // host passes mesh vertexOffset as baseVertex
}

// Each direct draw covers only one bucket's meshlet/slot chunk. Coordinates
// are local to this dispatch; camera visibility never defines caster membership.
using ShadowDepthMesh=metal::mesh<ShadowDepthOut,void,MESHLET_MESH_GROUP,MESHLET_MESH_GROUP,topology::triangle>;
[[mesh]] void shadow_depth_mesh(ShadowDepthMesh out,uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],constant GPUShadowParams& p [[buffer(1)]],
    const device GPUInstance* instances [[buffer(3)]],const device GPUMeshInfo* meshes [[buffer(5)]],
    const device GPUVertex* vertices [[buffer(6)]],const device GPUMeshlet* meshlets [[buffer(7)]],
    const device uint* meshletVertices [[buffer(8)]],const device uchar* triangles [[buffer(9)]],
    const device uint* casterFlags [[buffer(14)]]) {
    const uint slot=shadowDrawSlot(p.casterSlot,group.y,p.slotCount);
    const uint meshlet=shadowDrawMeshlet(p.meshletFirst,group.x,p.meshletCount);
    bool valid=slot<p.slotCount && p.cascadeIndex<4;
    if(valid) {
        const device GPUInstance& i=instances[slot];
        valid=(i.flags&INSTANCE_FLAG_VALID) && (i.flags&2u) && i.meshIndex==p.pad &&
              i.meshIndex<p.meshCount && i.materialIndex<p.materialCount && (casterFlags[slot]&(1u<<p.cascadeIndex));
        if(valid) {
            const device GPUMeshInfo& mesh=meshes[i.meshIndex];
            valid=meshlet>=mesh.meshletOffset && meshlet-mesh.meshletOffset<mesh.meshletCount;
        }
    }
    if(!valid) { if(tid==0) out.set_primitive_count(0); return; }
    const GPUMeshlet m=meshlets[meshlet];
    if(m.vertexCount>MESHLET_MESH_GROUP || m.triangleCount>MESHLET_MESH_GROUP) {
        if(tid==0) out.set_primitive_count(0); return; // upload validation must reject this asset
    }
    if(tid==0) out.set_primitive_count(m.triangleCount);
    if(tid<m.vertexCount) out.set_vertex(tid,shadowDepthVertex(vertices[meshletVertices[m.vertexOffset+tid]],instances[slot],p));
    if(tid<m.triangleCount) {
        const uint base=m.triangleOffset+3u*tid;
        out.set_index(3u*tid,triangles[base]); out.set_index(3u*tid+1u,triangles[base+1u]); out.set_index(3u*tid+2u,triangles[base+2u]);
    }
}
fragment void shadow_depth_fragment(ShadowDepthOut in [[stage_in]],
    const device GPUMaterial* materials [[buffer(16)]],const device RtTextureHandle* textures [[buffer(17)]]) {
    const device GPUMaterial& m=materials[in.material];
    if(m.alphaCutoff>0) {
        float alpha=m.baseColor[3];
        if(m.baseColorTex!=INVALID_TEXTURE_INDEX) alpha*=float(half(textures[m.baseColorTex].tex.sample(kRtAlphaSampler,in.uv,level(0)).a));
        if(alpha<m.alphaCutoff) discard_fragment();
    }
}

static float2 shadowDisk(uint sample,uint count,float rotation) {
    const float r=sqrt((float(sample)+0.5f)/float(max(count,1u)));
    const float a=float(sample)*2.399963229728653f+rotation;
    return r*float2(cos(a),sin(a));
}
static float shadowMapRead(depth2d<float,access::sample> map0,depth2d<float,access::sample> map1,
                             depth2d<float,access::sample> map2,depth2d<float,access::sample> map3,
                             float2 uv,uint cascade) {
    constexpr sampler smp(coord::normalized,filter::nearest,address::clamp_to_edge);
    switch(cascade) {
        case 0: return map0.sample(smp,uv);
        case 1: return map1.sample(smp,uv);
        case 2: return map2.sample(smp,uv);
        default: return map3.sample(smp,uv);
    }
}
static float shadowPcss(float3 point,float3 normal,uint cascade,constant GPUShadowParams& p,
                         depth2d<float,access::sample> map0,depth2d<float,access::sample> map1,
                         depth2d<float,access::sample> map2,depth2d<float,access::sample> map3,uint seed) {
    const GPUShadowCascade c=p.cascades[cascade];
    const float3 light=normalize(float3(p.lightDirection[0],p.lightDirection[1],p.lightDirection[2]));
    if(dot(normal,light)<0) normal=-normal;
    const float3 receiver=point+normal*c.normalBiasWorld;
    const float4 clip=shadowMatrix(p.cascades[cascade].viewProjection)*float4(receiver,1);
    const float2 uv=clip.xy*float2(0.5f,-0.5f)+0.5f;
    if(any(uv<0)||any(uv>1)||clip.z<0||clip.z>1) return 1;
    const float slope=sqrt(max(0.0f,1-dot(normal,light)*dot(normal,light)))/max(abs(dot(normal,light)),0.2f);
    const float bias=(c.biasWorld+c.texelWorld*slope*0.25f)/max(c.depthRange,1e-6f);
    const float receiverDistance=(1-clip.z)*c.depthRange;
    const float searchWorld=receiverDistance*tan(p.lightDirection[3]);
    const float searchUv=min(searchWorld/(2*c.radius),0.25f);
    const float rotation=6.283185307f*shadowUniform(seed);
    const uint searchCount=clamp(p.pcssSearchSamples,1u,64u),filterCount=clamp(p.pcssFilterSamples,1u,64u);
    float blockers=0,blockerDistance=0;
    for(uint s=0;s<searchCount;++s) {
        const float2 at=uv+shadowDisk(s,searchCount,rotation)*searchUv;
        if(any(at<0)||any(at>1)) continue;
        const float d=shadowMapRead(map0,map1,map2,map3,at,cascade);
        if(!shadowDepthVisible(clip.z,d,bias)) { blockerDistance+=(1-d)*c.depthRange; blockers+=1; }
    }
    if(blockers==0) return 1;
    const float penumbra=shadowPenumbraWorld(receiverDistance,blockerDistance/blockers,p.lightDirection[3]);
    const float radiusUv=min(max(penumbra,c.texelWorld*0.5f)/(2*c.radius),0.25f);
    float visible=0;
    for(uint s=0;s<filterCount;++s) {
        const float2 at=uv+shadowDisk(s,filterCount,rotation)*radiusUv;
        visible+=any(at<0)||any(at>1)?1.0f:float(shadowDepthVisible(clip.z,shadowMapRead(map0,map1,map2,map3,at,cascade),bias));
    }
    return visible/float(filterCount);
}
kernel void shadow_csm_pcss(constant GPUShadowParams& p [[buffer(1)]],
    const device GPUShadowSurface* surfaces [[buffer(2)]],depth2d<float,access::sample> map0 [[texture(2)]],
    depth2d<float,access::sample> map1 [[texture(7)]],depth2d<float,access::sample> map2 [[texture(8)]],
    depth2d<float,access::sample> map3 [[texture(9)]],
    texture2d<float,access::write> mask [[texture(4)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const GPUShadowSurface s=surfaces[pixel.y*p.width+pixel.x];
    float visibility=1;
    if(s.valid && (p.flags&SHADOW_FLAG_LIGHT_VALID)) {
        for(uint c=0;c<4;++c) if(s.viewDepth>=p.cascades[c].splitNear && s.viewDepth<=p.cascades[c].splitFar) {
            visibility=shadowPcss(shadowPoint(s),shadowNormal(s),c,p,map0,map1,map2,map3,pixel.y*p.width+pixel.x);
            // Blend only the common receiver volume close to the cascade far
            // boundary. The next cascade must cover this overlap at kickoff.
            const float start=p.cascades[c].splitFar-(p.cascades[c].splitFar-p.cascades[c].splitNear)*0.1f;
            if(c<3 && s.viewDepth>start) visibility=mix(visibility,
                shadowPcss(shadowPoint(s),shadowNormal(s),c+1,p,map0,map1,map2,map3,pixel.y*p.width+pixel.x),
                saturate((s.viewDepth-start)/max(p.cascades[c].splitFar-start,1e-6f)));
            break;
        }
    }
    mask.write(float4(visibility,visibility*visibility,0,1),pixel);
}

static float3 shadowSolarSample(float3 light,float angularRadius,float u,float v) {
    const float3 helper=abs(light.y)<0.95f?float3(0,1,0):float3(1,0,0);
    const float3 right=normalize(cross(helper,light)),up=cross(light,right);
    const float z=1-u*(1-cos(angularRadius)),r=sqrt(max(0.0f,1-z*z)),a=6.283185307f*v;
    return normalize(light*z+(right*cos(a)+up*sin(a))*r);
}
kernel void shadow_sun_rt(instance_acceleration_structure as [[buffer(0)]],
    constant GPUShadowParams& p [[buffer(1)]],const device GPUShadowSurface* surfaces [[buffer(2)]],
    const device GPUInstance* instances [[buffer(3)]],
    intersection_function_table<triangle_data,instancing> ift [[buffer(4)]],
    device atomic_uint* counters [[buffer(15)]],texture2d<float,access::write> mask [[texture(4)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const uint index=pixel.y*p.width+pixel.x;
    const GPUShadowSurface s=surfaces[index];
    float visible=1;
    if(s.valid && (p.flags&SHADOW_FLAG_LIGHT_VALID)) {
        const uint seed=shadowHash(index^shadowHash(p.frameIndex)^shadowHash(p.viewID)^shadowHash(p.lightID));
        const float3 towardLight=normalize(float3(p.lightDirection[0],p.lightDirection[1],p.lightDirection[2]));
        const float3 direction=shadowSolarSample(towardLight,p.lightDirection[3],shadowUniform(seed),shadowUniform(seed^0x51633e2du));
        float3 normal=shadowNormal(s);
        if(dot(normal,direction)<0) normal=-normal;
        if(p.corruption==SHADOW_CORRUPT_BIAS) normal=-normal;
        const float3 origin=rtOffsetRay(shadowPoint(s),normal);
        GPURtRay ray{};
        ray.ox=origin.x; ray.oy=origin.y; ray.oz=origin.z;
        ray.dx=direction.x; ray.dy=direction.y; ray.dz=direction.z; ray.tmax=p.maxTraceDistance;
        ray.mask=RT_MASK_SHADOW; ray.type=RT_PROBE_SHADOW; ray.coneWidth=0;
        RtPayload payload{};
        const GPURtHit hit=rtTrace(ray,as,ift,instances,p.slotCount,payload);
        visible=hit.hit?0.0f:1.0f;
        shadowCount(counters,1,1); shadowCount(counters,2,hit.hit);
        if(hit.t==-2.0f) shadowCount(counters,7,1);
    }
    mask.write(float4(visible,visible*visible,0,1),pixel);
}

kernel void shadow_temporal(constant GPUShadowParams& p [[buffer(1)]],
    const device GPUShadowSurface* surfaces [[buffer(2)]],const device GPUShadowHistory* previous [[buffer(12)]],
    device GPUShadowHistory* next [[buffer(13)]],device atomic_uint* counters [[buffer(15)]],
    texture2d<float,access::read> raw [[texture(3)]],texture2d<float,access::write> mask [[texture(4)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const uint index=pixel.y*p.width+pixel.x;
    const GPUShadowSurface s=surfaces[index];
    const float value=saturate(raw.read(pixel).x);
    GPUShadowHistory h{}; h.visibility=value; h.secondMoment=value*value;
    h.slot=s.slot; h.generation=s.generation; h.lightID=p.lightID; h.lightRevision=p.lightRevision;
    h.sceneRevision=p.sceneRevision; h.viewID=p.viewID; h.samples=1; h.valid=s.valid && (p.flags&SHADOW_FLAG_LIGHT_VALID);
    for(uint k=0;k<3;++k) { h.position[k]=s.position[k]; h.geometricNormal[k]=s.geometricNormal[k]; }
    bool accepted=false;
    if(s.valid && (p.flags&SHADOW_FLAG_HISTORY_VALID)) {
        const float4 clip=shadowMatrix(p.previousViewProjection)*float4(shadowPoint(s),1);
        if(clip.w>0 && all(isfinite(clip))) {
            const float2 old=(clip.xy/clip.w*float2(0.5f,-0.5f)+0.5f)*float2(p.width,p.height);
            if(all(old>=0) && all(old<float2(p.width,p.height))) {
                const uint2 q=uint2(old);
                GPUShadowHistory prev=previous[q.y*p.width+q.x];
                if(p.corruption==SHADOW_CORRUPT_HISTORY) prev.generation^=1u;
                accepted=shadowHistoryMatches(p,s,prev) && isfinite(prev.visibility) && isfinite(prev.secondMoment);
                if(accepted) {
                    // Keep the Bernoulli expectation. Clipping a converged
                    // mean to one noisy raw neighborhood contracts penumbrae
                    // even with a static scene; stale history is rejected by
                    // shadowHistoryMatches rather than by sample extrema.
                    h.samples=min(prev.samples+1u,clamp(p.historyMaxSamples,1u,64u));
                    h.visibility=shadowTemporalMean(prev.visibility,value,h.samples);
                    h.secondMoment=shadowTemporalMean(prev.secondMoment,value*value,h.samples);
                }
            }
        }
        shadowCount(counters,accepted?3u:4u,1u);
    }
    next[index]=h;
    mask.write(float4(h.visibility,max(0.0f,h.secondMoment-h.visibility*h.visibility),float(h.samples),1),pixel);
}

// Bilateral 3x3 moment-aware mask filter; history remains the unfiltered
// temporal value, avoiding an unbounded spatial blur feedback loop.
kernel void shadow_filter(constant GPUShadowParams& p [[buffer(1)]],
    const device GPUShadowSurface* surfaces [[buffer(2)]],texture2d<float,access::read> input [[texture(3)]],
    texture2d<float,access::write> output [[texture(4)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const GPUShadowSurface s=surfaces[pixel.y*p.width+pixel.x];
    const float4 center=input.read(pixel);
    float sum=center.x,weight=1;
    if(s.valid) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        if(x==0&&y==0) continue;
        const int2 n=int2(pixel)+int2(x,y);
        if(any(n<0)||any(n>=int2(p.width,p.height))) continue;
        const GPUShadowSurface q=surfaces[uint(n.y)*p.width+uint(n.x)];
        if(!q.valid || q.slot!=s.slot || q.generation!=s.generation || dot(shadowNormal(q),shadowNormal(s))<0.95f) continue;
        const float distance=length(shadowPoint(q)-shadowPoint(s));
        const float varianceStrength=saturate(max(center.y,0.0f)*16.0f);
        const float w=exp(-distance/max(p.temporalPositionThreshold*4,1e-5f))*mix(0.125f,0.5f,varianceStrength);
        sum+=input.read(uint2(n)).x*w; weight+=w;
    }
    output.write(float4(sum/weight,center.yzw),pixel);
}

kernel void shadow_contact(constant GPUShadowParams& p [[buffer(1)]],
    const device GPUShadowSurface* surfaces [[buffer(2)]],device atomic_uint* counters [[buffer(15)]],
    texture2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> mainMask [[texture(3)]],
    texture2d<float,access::write> output [[texture(4)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.width || pixel.y>=p.height) return;
    const GPUShadowSurface s=surfaces[pixel.y*p.width+pixel.x];
    float4 value=mainMask.read(pixel);
    bool occluded=false;
    if(s.valid && (p.flags&SHADOW_FLAG_LIGHT_VALID) && (p.flags&SHADOW_FLAG_CONTACT)) {
        const float3 light=normalize(float3(p.lightDirection[0],p.lightDirection[1],p.lightDirection[2]));
        float3 normal=shadowNormal(s); if(dot(normal,light)<0) normal=-normal;
        const float3 origin=rtOffsetRay(shadowPoint(s),normal);
        const uint count=clamp(p.contactSteps,1u,64u);
        for(uint k=1;k<=count;++k) {
            const float distance=p.contactDistance*float(k)/float(count);
            const float3 point=origin+light*distance;
            const float4 clip=shadowMatrix(p.viewProjection)*float4(point,1);
            if(!(clip.w>0)||!all(isfinite(clip))) break;
            const float2 uv=clip.xy/clip.w*float2(0.5f,-0.5f)+0.5f;
            if(any(uv<0)||any(uv>=1)) break; // missing depth falls back to main signal
            const uint2 q=uint2(uv*float2(p.width,p.height));
            if(all(q==pixel)) continue;
            const float d=depth.read(q).x;
            if(!(d>0&&d<=1)) continue;
            const float4 h=shadowMatrix(p.inverseViewProjection)*float4((float2(q)+0.5f)*float2(2,-2)/float2(p.width,p.height)+float2(-1,1),d,1);
            if(abs(h.w)<1e-20f||!all(isfinite(h))) continue;
            const float3 screenPoint=h.xyz/h.w;
            const float rayViewDepth=-(shadowMatrix(p.view)*float4(point,1)).z;
            const float screenDepth=-(shadowMatrix(p.view)*float4(screenPoint,1)).z;
            const float delta=rayViewDepth-screenDepth;
            if(delta>0 && delta<p.contactThickness) { occluded=true; break; }
        }
        if(occluded) { value.x=min(value.x,1-saturate(p.contactStrength)); shadowCount(counters,6,1); }
    }
    output.write(value,pixel);
}
