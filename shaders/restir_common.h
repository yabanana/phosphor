#pragma once
#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;
using namespace phosphor;

// All light integrands are with respect to AREA endpoints plus punctual
// discrete endpoints. Solid-angle PDF is exposed for consumer/MIS diagnostics,
// never accidentally mixed into area-measure reservoir candidate weights.
struct DISample {
    float3 position, normal, wi, radiance;
    float distance, pdfArea, pdfSolidAngle, geometry;
    bool delta, valid;
};
struct DITextureHandle { texture2d<float> tex; };
static_assert(sizeof(DITextureHandle) == 8, "DI bindless texture handle ABI");
constexpr sampler kDIEmitterSampler(filter::linear, mip_filter::linear, address::repeat);
inline float3 diVec(const thread float* p) { return float3(p[0], p[1], p[2]); }
inline float diLuminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
inline float diArea(GPUSampledLight l) {
    const float3 u = diVec(l.axisU), v = diVec(l.axisV);
    float a = 0;
    if (l.type == DI_LIGHT_RECTANGLE) a = 4.0f * length(cross(u, v));
    if (l.type == DI_LIGHT_DISK && l.radius > 0) a = M_PI_F * l.radius * l.radius * length(cross(u, v));
    if (l.type == DI_LIGHT_TUBE && l.radius > 0) a = 4.0f * M_PI_F * l.radius * length(u);
    if (l.type == DI_LIGHT_TRIANGLE) a = 0.5f * length(cross(u, v));
    return isfinite(a) && a > 0 ? a : 0;
}

inline DISample diSampleLight(GPUSampledLight l, float2 uv, float3 receiver) {
    DISample s{};
    if (!all(isfinite(uv)) || any(uv < 0) || any(uv >= 1) || !all(isfinite(receiver))) return s;
    s.position = diVec(l.position);
    if (!all(isfinite(diVec(l.emission)))) return s;
    s.radiance = max(diVec(l.emission), 0.0f);
    const float3 u = diVec(l.axisU), v = diVec(l.axisV);
    s.delta = l.type == LIGHT_POINT || l.type == LIGHT_SPOT;
    if (!s.delta) {
        const float area = diArea(l);
        if (!(area > 0)) return s;
        s.pdfArea = 1.0f / area;
        if (l.type == DI_LIGHT_RECTANGLE) s.position += (2.0f * uv.x - 1.0f) * u + (2.0f * uv.y - 1.0f) * v;
        else if (l.type == DI_LIGHT_DISK) {
            const float r = l.radius * sqrt(uv.x), angle = 2.0f * M_PI_F * uv.y;
            s.position += r * (cos(angle) * u + sin(angle) * v);
        } else if (l.type == DI_LIGHT_TRIANGLE) {
            const float root = sqrt(uv.x);
            s.position += root * (1.0f - uv.y) * u + root * uv.y * v;
        } else if (l.type == DI_LIGHT_TUBE) {
            const float3 axis = normalize(u), radial = v - axis * dot(axis, v);
            if (!(dot(radial, radial) > 1e-20f)) return s;
            const float3 x = normalize(radial), y = cross(axis, x);
            const float angle = 2.0f * M_PI_F * uv.y;
            s.normal = cos(angle) * x + sin(angle) * y;
            s.position += (2.0f * uv.x - 1.0f) * u + l.radius * s.normal;
        } else return s;
        if (l.type != DI_LIGHT_TUBE) s.normal = normalize(cross(u, v));
        if (l.type == DI_LIGHT_TRIANGLE && (l.flags & DI_LIGHT_MIRRORED)) s.normal = -s.normal;
    }
    const float3 delta = s.position - receiver;
    s.distance = length(delta);
    if (!(s.distance > 1e-10f) || !isfinite(s.distance)) { s.valid = false; return s; }
    s.wi = delta / s.distance;
    if (s.delta) {
        s.pdfArea = 1.0f; // discrete endpoint measure, NOT m^-2
        const float ratio = s.distance / max(l.range, 1e-3f);
        const float window = saturate(1.0f - ratio * ratio * ratio * ratio);
        s.geometry = window * window / max(s.distance * s.distance, 1e-4f);
        if (l.type == LIGHT_SPOT) {
            if (!(dot(u, u) > 0)) return s;
            const float spot = saturate((dot(normalize(u), -s.wi) - cos(l.outerCone)) /
                                        max(cos(l.innerCone) - cos(l.outerCone), 1e-4f));
            s.geometry *= spot * spot * (3.0f - 2.0f * spot);
        }
    } else {
        const float facing = dot(s.normal, -s.wi);
        const float cosine = (l.flags & DI_LIGHT_TWO_SIDED) ? abs(facing) : max(facing, 0.0f);
        s.geometry = cosine / (s.distance * s.distance);
        s.pdfSolidAngle = cosine > 0 ? s.pdfArea * s.distance * s.distance / cosine : 0;
        if (l.range > 0) {
            const float ratio = s.distance / l.range;
            const float window = saturate(1.0f - ratio * ratio * ratio * ratio);
            s.geometry *= window * window;
        }
    }
    s.valid = all(isfinite(s.position)) && all(isfinite(s.wi)) && all(isfinite(s.radiance)) &&
              isfinite(s.geometry) && s.geometry >= 0;
    return s;
}

// The full-scene emissive extraction pass supplies transformed geometry, UVs
// and materialIndex. Constant/nontextured lights never read these buffers.
// Emissive RGB and MASK alpha use explicit LOD0, with the raster's half texel
// arithmetic. No camera derivatives exist on a sampled light surface.
inline DISample diSampleTexturedLight(GPUSampledLight l, uint lightIndex, float2 uv, float3 receiver,
                                      const device GPUEmissiveSurface* emitters,
                                      const device GPUMaterial* materials, const device DITextureHandle* textures) {
    DISample s = diSampleLight(l, uv, receiver);
    if (!s.valid || l.type != DI_LIGHT_TRIANGLE || !(l.flags & DI_LIGHT_TEXTURED_EMISSION)) return s;
    const GPUEmissiveSurface emitter = emitters[lightIndex];
    if (!emitter.valid) { s.valid = false; return s; }
    const GPUMaterial m = materials[emitter.materialIndex]; // host/extraction bounds-checked
    const float root = sqrt(uv.x);
    const float2 lightUV = (1.0f - root) * float2(emitter.uv0[0], emitter.uv0[1]) +
                           root * (1.0f - uv.y) * float2(emitter.uv1[0], emitter.uv1[1]) +
                           root * uv.y * float2(emitter.uv2[0], emitter.uv2[1]);
    const half3 texel = m.emissiveTex == INVALID_TEXTURE_INDEX ? half3(1) :
                        half3(textures[m.emissiveTex].tex.sample(kDIEmitterSampler, lightUV, level(0.0f)).rgb);
    s.radiance = float3(m.emissive[0], m.emissive[1], m.emissive[2]) * float3(texel);
    if (m.alphaCutoff > 0) {
        const half alphaTexel = m.baseColorTex == INVALID_TEXTURE_INDEX ? half(1) :
                                half(textures[m.baseColorTex].tex.sample(kDIEmitterSampler, lightUV, level(0.0f)).a);
        const float alpha = m.baseColor[3] * float(alphaTexel);
        if (!isfinite(alpha)) { s.valid = false; return s; }
        if (alpha < m.alphaCutoff) s.radiance = float3(0);
    }
    s.valid = all(isfinite(s.radiance)) && all(s.radiance >= 0);
    return s;
}

inline float3 diIncident(DISample s) { return s.valid ? s.radiance * s.geometry : float3(0); }
inline float3 diBRDF(GPUDISurface surface, DISample s) {
    if (!surface.valid || !s.valid) return float3(0);
    float3 n = diVec(surface.shadingNormal), v = diVec(surface.viewDirection);
    if (!all(isfinite(n)) || !all(isfinite(v)) || !(dot(n, n) > 0 && dot(v, v) > 0)) return float3(0);
    n = normalize(n); v = normalize(v);
    const float nl = saturate(dot(n, s.wi)), nv = max(dot(n, v), 1e-4f);
    if (!(nl > 0) || dot(v + s.wi, v + s.wi) <= 1e-20f) return float3(0);
    const float3 h = normalize(v + s.wi);
    const float nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
    const float metallic = saturate(surface.metallic);
    const float3 albedo = max(diVec(surface.albedo), 0.0f);
    const float a = max(surface.roughness * surface.roughness, 0.002f), a2 = a * a;
    const float d = nh * nh * (a2 - 1.0f) + 1.0f;
    const float distribution = a2 / (M_PI_F * d * d);
    const float gv = nl * sqrt(nv * nv * (1.0f - a2) + a2), gl = nv * sqrt(nl * nl * (1.0f - a2) + a2);
    const float smith = 0.5f / max(gv + gl, 1e-5f);
    const float3 f0 = mix(float3(0.04f), albedo, metallic);
    const float3 fresnel = f0 + (1.0f - f0) * pow(1.0f - vh, 5.0f);
    const float3 result = ((1.0f - fresnel) * (1.0f - metallic) * albedo / M_PI_F +
                           distribution * smith * fresnel) * diIncident(s) * nl;
    return result; // caller MUST record nonfinite BEFORE any safe-output sanitization
}
inline float diTarget(GPUDISurface surface, DISample sample, float positiveFloor) {
    if (!sample.valid || !isfinite(positiveFloor) || !(positiveFloor > 0)) return 0;
    const float3 value=diBRDF(surface,sample);
    if(!all(isfinite(value)))return as_type<float>(0x7fc00000u);
    return max(diLuminance(value),positiveFloor);
}

inline uint diHash(uint v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}
inline float diWhite(uint x, uint y, uint frame, uint dimension, uint seed) {
    const uint bits = diHash(seed ^ diHash(x) ^ diHash(y + 0x9e3779b9u) ^
                             diHash(frame + 0x632be5abu) ^ diHash(dimension + 0x85157af5u));
    return float(bits >> 8) * (1.0f / 16777216.0f);
}
inline float diRandom(uint2 pixel, uint dimension, constant GPUDIParams& p, const device uint* ranks) {
    if (!(p.flags & DI_USE_STBN) || !p.stbnWidth || !p.stbnHeight || !p.stbnFrames || !p.stbnDimensions)
        return diWhite(pixel.x, pixel.y, p.frameIndex, dimension, p.stbnSeed);
    const uint n = p.stbnWidth * p.stbnHeight * p.stbnFrames;
    const uint index = (((dimension % p.stbnDimensions) * p.stbnFrames + p.frameIndex % p.stbnFrames) *
                        p.stbnHeight + pixel.y % p.stbnHeight) * p.stbnWidth + pixel.x % p.stbnWidth;
    const float rank = (float(ranks[index]) + 0.5f) / float(n);
    return fract(rank + diWhite(0, 0, p.frameIndex / p.stbnFrames, dimension, p.stbnSeed ^ 0xa511e9b3u));
}

inline GPUDIReservoir diEmpty(constant GPUDIParams& p) {
    GPUDIReservoir r{};
    r.lightIndex = ~0u; r.viewID = p.viewID; r.historyEpoch = p.historyEpoch; r.lightRevision = p.lightRevision;
    return r;
}
inline bool diStream(thread GPUDIReservoir& r, GPUDIReservoir sample, float weight, uint m, float random) {
    if (!isfinite(weight) || weight < 0 || !isfinite(random) || random < 0 || random >= 1 ||
        m == 0 || m > ~0u - r.M) { r.pad[0] |= DI_ERROR_WEIGHT; return false; }
    const float sum = r.weightSum + weight;
    if (!isfinite(sum)) { r.pad[0] |= DI_ERROR_WEIGHT; return false; }
    r.M += m;r.pad[1]|=DI_PROPOSAL_VALID;
    if (weight > 0 && random * sum < weight) {
        r.lightIndex = sample.lightIndex; r.lightID = sample.lightID; r.lightGeneration = sample.lightGeneration;
        r.u = sample.u; r.v = sample.v; r.target = sample.target; r.age = sample.age; r.valid = 1;
    }
    r.weightSum = sum; r.normalization = 0;
    return true;
}
inline void diFinalize(thread GPUDIReservoir& r) {
    // Divide successively to avoid overflow of M*target even when W is finite.
    const float w = r.M > 0 && r.target > 0 ? (r.weightSum / float(r.M)) / r.target : 0;
    r.valid = r.valid && r.pad[0] == 0 && isfinite(w) && w > 0;
    r.normalization = r.valid ? w : 0;
}
inline bool diValid(GPUDIReservoir r, GPUSampledLight light, constant GPUDIParams& p) {
    return r.valid && r.pad[0] == 0 && r.M > 0 && r.viewID == p.viewID &&
           r.historyEpoch == p.historyEpoch && r.lightRevision == p.lightRevision &&
           r.lightID == light.id && r.lightGeneration == light.generation && isfinite(r.target) && r.target > 0 &&
           isfinite(r.normalization) && r.normalization > 0 && isfinite(r.u) && isfinite(r.v) &&
           r.u >= 0 && r.u < 1 && r.v >= 0 && r.v < 1;
}
inline bool diZeroProposal(GPUDIReservoir r,constant GPUDIParams& p) {
    return (r.pad[1]&DI_PROPOSAL_VALID) && !r.valid && r.M>0 && r.pad[0]==0 &&
        r.weightSum==0 && r.normalization==0 && r.age<p.maxHistoryAge &&
        r.viewID==p.viewID && r.historyEpoch==p.historyEpoch && r.lightRevision==p.lightRevision;
}
inline bool diReusable(GPUDIReservoir r, GPUSampledLight light, constant GPUDIParams& p) {
    return diZeroProposal(r,p) || (diValid(r, light, p) && r.age < p.maxHistoryAge);
}
inline bool diCompatible(GPUDISurface a, GPUDISurface b, constant GPUDIParams& p, bool temporal) {
    if (!a.valid || !b.valid || !isfinite(a.depth) || !isfinite(b.depth) || a.depth <= 0 || b.depth <= 0) return false;
    if (temporal && (a.instanceSlot != b.instanceSlot || a.instanceGeneration != b.instanceGeneration ||
                     a.materialRevision != b.materialRevision)) return false;
    const float3 an = diVec(a.geometricNormal), bn = diVec(b.geometricNormal);
    const float3 ap = diVec(a.position), bp = diVec(b.position);
    if (!all(isfinite(an)) || !all(isfinite(bn)) || !all(isfinite(ap)) || !all(isfinite(bp)) ||
        !(dot(an, an) > 0 && dot(bn, bn) > 0) || dot(normalize(an), normalize(bn)) < p.normalThreshold) return false;
    const float tolerance = p.depthRelativeThreshold * max(a.depth, b.depth);
    return abs(a.depth - b.depth) <= tolerance && abs(dot(bp - ap, normalize(an))) <= tolerance;
}
inline bool diMerge(thread GPUDIReservoir& r, GPUDIReservoir source, GPUDISurface destination,
                     GPUSampledLight light, constant GPUDIParams& p, float random, bool advanceAge,
                     const device GPUEmissiveSurface* emitters,
                     const device GPUMaterial* materials, const device DITextureHandle* textures) {
    if(diZeroProposal(source,p))return diStream(r,source,0,min(source.M,p.maxHistoryM),random);
    if (!diReusable(source, light, p)) return false;
    const uint m = min(source.M, p.maxHistoryM);
    const DISample sample = diSampleTexturedLight(light, source.lightIndex, float2(source.u, source.v),
                                                 diVec(destination.position), emitters, materials, textures);
    source.target = diTarget(destination, sample, p.targetFloor);
    if (advanceAge) ++source.age;
    return diStream(r, source, source.target * source.normalization * float(m), m, random);
}

// Persistent diagnostics survive sanitization; this is independent of output
// texture consistency and does not claim an energetic reference comparison.
inline void diRecordNonfinite(float3 value,device atomic_uint* errors,uint word) {
    if(!all(isfinite(value)))atomic_fetch_add_explicit(errors+word,1u,memory_order_relaxed);
}
inline bool diLightFinite(GPUSampledLight light) {
    return all(isfinite(diVec(light.emission))) && all(isfinite(diVec(light.position))) &&
           all(isfinite(diVec(light.axisU))) && all(isfinite(diVec(light.axisV))) &&
           isfinite(light.range) && isfinite(light.radius) && isfinite(light.innerCone) && isfinite(light.outerCone);
}
