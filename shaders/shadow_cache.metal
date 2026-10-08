#include <metal_stdlib>
#include "renderer/gpu_types.h"
#include "renderer/shadow_dispatch.h"
#include "renderer/meshlet_layout.h"
using namespace metal;
using namespace phosphor;

// F10.5 adds buffer18 GPUShadowCacheParams, 19 CPU conservative static class,
// 20 expected GPUShadowCacheTile[64], 21 persistent GPUShadowCacheTile[64].
// Raster bindings 1/3/5/6/7/8/9/14 match shadows.metal. Cache copy: texture0
// current static depth, texture1 persistent R32Float cache. Composite: tex0
// cache R32Float, tex1 current static depth, tex2 CURRENT dynamic depth.
static bool cacheBit(uint lo,uint hi,uint tile) { return tile<32?bool(lo&(1u<<tile)):bool(hi&(1u<<(tile-32))); }
static uint cacheTile(uint2 pixel,uint resolution) {
    return min(pixel.y*8u/resolution,7u)*8u+min(pixel.x*8u/resolution,7u);
}
static bool cacheSame(GPUShadowCacheTile a,GPUShadowCacheTile b) {
    return a.valid && b.valid && a.lightLo==b.lightLo && a.lightHi==b.lightHi &&
           a.casterLo==b.casterLo && a.casterHi==b.casterHi && a.materialLo==b.materialLo && a.materialHi==b.materialHi &&
           a.projectionLo==b.projectionLo && a.projectionHi==b.projectionHi;
}
static float4x4 cacheMatrix(constant float* m) {
    return float4x4(float4(m[0],m[1],m[2],m[3]),float4(m[4],m[5],m[6],m[7]),
                    float4(m[8],m[9],m[10],m[11]),float4(m[12],m[13],m[14],m[15]));
}
static float3 cacheWorld(const device GPUInstance& i,const device GPUVertex& v) {
    const device float* m=i.modelMatrix;
    return float3(m[0],m[1],m[2])*v.px+float3(m[4],m[5],m[6])*v.py+
           float3(m[8],m[9],m[10])*v.pz+float3(m[12],m[13],m[14]);
}
struct CacheDepthOut { float4 position [[position]]; float2 uv; uint material [[flat]]; };
static bool cacheCaster(uint slot,constant GPUShadowParams& p,constant GPUShadowCacheParams& cache,
                         const device uint* flags,const device uint* classifications) {
    if(slot>=p.slotCount || p.cascadeIndex>=4 || !(flags[slot]&(1u<<p.cascadeIndex))) return false;
    return cache.casterClass==2u || (cache.casterClass==0u?classifications[slot]!=0u:classifications[slot]==0u);
}
static CacheDepthOut cacheVertex(const device GPUVertex& v,const device GPUInstance& i,constant GPUShadowParams& p) {
    CacheDepthOut o; o.position=cacheMatrix(p.cascades[p.cascadeIndex].viewProjection)*float4(cacheWorld(i,v),1);
    o.uv=float2(v.u,v.v);o.material=i.materialIndex;return o;
}
vertex CacheDepthOut shadow_cache_depth_vertex(uint vertexId [[vertex_id]],uint instance [[instance_id]],
    constant GPUShadowParams& p [[buffer(1)]],const device GPUInstance* instances [[buffer(3)]],
    const device GPUVertex* vertices [[buffer(6)]],const device uint* flags [[buffer(14)]],
    constant GPUShadowCacheParams& cache [[buffer(18)]],const device uint* classifications [[buffer(19)]]) {
    CacheDepthOut invalid{};invalid.position=float4(2,2,0,1);
    const uint slot=shadowDrawSlot(p.casterSlot,instance,p.slotCount);
    if(!cacheCaster(slot,p,cache,flags,classifications)) return invalid;
    const device GPUInstance& i=instances[slot];
    if(!(i.flags&INSTANCE_FLAG_VALID) || !(i.flags&2u) || i.meshIndex!=p.pad || i.materialIndex>=p.materialCount) return invalid;
    return cacheVertex(vertices[vertexId],i,p);
}
using CacheMesh=metal::mesh<CacheDepthOut,void,MESHLET_MESH_GROUP,MESHLET_MESH_GROUP,topology::triangle>;
[[mesh]] void shadow_cache_depth_mesh(CacheMesh out,uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],constant GPUShadowParams& p [[buffer(1)]],
    const device GPUInstance* instances [[buffer(3)]],const device GPUMeshInfo* meshes [[buffer(5)]],
    const device GPUVertex* vertices [[buffer(6)]],const device GPUMeshlet* meshlets [[buffer(7)]],
    const device uint* vertexIndices [[buffer(8)]],const device uchar* triangles [[buffer(9)]],
    const device uint* flags [[buffer(14)]],constant GPUShadowCacheParams& cache [[buffer(18)]],
    const device uint* classifications [[buffer(19)]]) {
    const uint slot=shadowDrawSlot(p.casterSlot,group.y,p.slotCount);
    const uint meshlet=shadowDrawMeshlet(p.meshletFirst,group.x,p.meshletCount);
    bool valid=cacheCaster(slot,p,cache,flags,classifications);
    if(valid) {
        const device GPUInstance& i=instances[slot];
        valid=(i.flags&INSTANCE_FLAG_VALID) && (i.flags&2u) && i.meshIndex==p.pad &&
              i.meshIndex<p.meshCount && i.materialIndex<p.materialCount;
        if(valid) { const device GPUMeshInfo& m=meshes[i.meshIndex];valid=meshlet>=m.meshletOffset && meshlet-m.meshletOffset<m.meshletCount; }
    }
    if(!valid) { if(tid==0)out.set_primitive_count(0);return; }
    const GPUMeshlet m=meshlets[meshlet];
    if(m.vertexCount>MESHLET_MESH_GROUP || m.triangleCount>MESHLET_MESH_GROUP) {if(tid==0)out.set_primitive_count(0);return;}
    if(tid==0)out.set_primitive_count(m.triangleCount);
    if(tid<m.vertexCount)out.set_vertex(tid,cacheVertex(vertices[vertexIndices[m.vertexOffset+tid]],instances[slot],p));
    if(tid<m.triangleCount)for(uint k=0;k<3;++k)out.set_index(3u*tid+k,triangles[m.triangleOffset+3u*tid+k]);
}
struct CacheTexture { texture2d<float> tex; };
fragment void shadow_cache_depth_fragment(CacheDepthOut in [[stage_in]],
    const device GPUMaterial* materials [[buffer(16)]],const device CacheTexture* textures [[buffer(17)]],
    constant GPUShadowCacheParams& cache [[buffer(18)]]) {
    const uint tile=cacheTile(uint2(in.position.xy),cache.resolution);
    if(cache.casterClass==0 && !cacheBit(cache.currentLo,cache.currentHi,tile))discard_fragment();
    const device GPUMaterial& m=materials[in.material];
    if(m.alphaCutoff>0) {
        constexpr sampler alphaSampler(filter::linear,mip_filter::linear,address::repeat);
        float alpha=m.baseColor[3];
        if(m.baseColorTex!=INVALID_TEXTURE_INDEX)alpha*=float(half(textures[m.baseColorTex].tex.sample(alphaSampler,in.uv,level(0)).a));
        if(alpha<m.alphaCutoff)discard_fragment();
    }
}
kernel void shadow_cache_publish(constant GPUShadowCacheParams& p [[buffer(18)]],
    const device GPUShadowCacheTile* expected [[buffer(20)]],device GPUShadowCacheTile* stored [[buffer(21)]],
    depth2d<float,access::read> current [[texture(0)]],texture2d<float,access::write> cache [[texture(1)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=p.resolution || pixel.y>=p.resolution)return;
    const uint tile=cacheTile(pixel,p.resolution);
    if(!cacheBit(p.updateLo,p.updateHi,tile))return;
    cache.write(float4(current.read(pixel)),pixel);
    // ceil(tileCoord*N/8) is the first texel assigned to this tile, also for
    // map sizes that are not multiples of eight.
    const uint2 first=uint2((tile%8u*p.resolution+7u)/8u,(tile/8u*p.resolution+7u)/8u);
    if(all(pixel==first))stored[tile]=expected[tile];
}
kernel void shadow_cache_initialize(constant GPUShadowCacheParams& p [[buffer(18)]],
    device GPUShadowCacheTile* stored [[buffer(21)]],uint tile [[thread_position_in_grid]]) {
    if(tile<64 && p.pad!=0u)stored[tile]=GPUShadowCacheTile{};
}
struct CacheFullscreen { float4 position [[position]]; };
kernel void shadow_cache_validate(constant GPUShadowCacheParams& p [[buffer(18)]],
    const device GPUShadowCacheTile* expected [[buffer(20)]],const device GPUShadowCacheTile* stored [[buffer(21)]],
    device atomic_uint* counters [[buffer(15)]],uint tile [[thread_position_in_grid]]) {
    if(tile>=64)return;
    const bool ready=cacheBit(p.readyLo|p.updateLo,p.readyHi|p.updateHi,tile);
    if(ready && !cacheSame(stored[tile],expected[tile]))
        atomic_fetch_add_explicit(counters+7,1u,memory_order_relaxed);
}
vertex CacheFullscreen shadow_cache_composite_vertex(uint vertexId [[vertex_id]]) {
    CacheFullscreen out;out.position=float4(vertexId==1?3.0f:-1.0f,vertexId==2?3.0f:-1.0f,0,1);return out;
}
struct CacheComposedDepth { float depth [[depth(any)]]; };
fragment CacheComposedDepth shadow_cache_composite_fragment(CacheFullscreen in [[stage_in]],
    constant GPUShadowCacheParams& p [[buffer(18)]],const device GPUShadowCacheTile* expected [[buffer(20)]],
    const device GPUShadowCacheTile* stored [[buffer(21)]],
    texture2d<float,access::read> cached [[texture(0)]],depth2d<float,access::read> current [[texture(1)]],
    depth2d<float,access::read> dynamic [[texture(2)]]) {
    const uint2 pixel=uint2(in.position.xy);
    const uint tile=cacheTile(pixel,p.resolution);
    // Updates are graph-ordered before this read; they count as ready NOW.
    const bool ready=cacheBit(p.readyLo|p.updateLo,p.readyHi|p.updateHi,tile);
    const bool valid=ready && cacheSame(stored[tile],expected[tile]);
    const float staticDepth=valid?cached.read(pixel).x:current.read(pixel);
    return CacheComposedDepth{max(staticDepth,dynamic.read(pixel))}; // nearest reverse depth, never multiplied visibility
}
