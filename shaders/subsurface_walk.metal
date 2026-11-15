// F15.1 (ticket 17/18, RT tier): volumetric subsurface by a ray-traced random walk (--subsurface-mode rt-walk, ADR-0005, decision 10).
//
// The resolve of a scene with volumetric subsurface (SSS_OUTPUT == 2) keeps the ambient and the specular terms in the colour, strips only the
// lights' diffuse (weight x base colour x the analytic lights' irradiance) and writes the subsurface colour and the weight to the target (alpha =
// material index + 1 + 0.49 weight).  This kernel replaces the screen-space diffusion: for every subsurface pixel it adds weight x the radiance
// of random walks through the ACTUAL closed mesh, so thickness, concavity and back-lit light come out of the same code.
//
// One walk (a hero colour channel per walk: the three channels have their own mean free path and albedo; the estimate is scaled by 3 on that
// channel):
//   1. the camera ray hits the boundary; a GGX visible normal is sampled and the ray is refracted into the medium (weight
//      (1 - F) G1(inside direction) / eta^2: the importance-sampling weight of the Walter transmission lobe, radiance form);
//   2. free flights of exponential length (sigma_t = 1 / (radius x radiusScale)) with a closest-hit ray bounded by the flight: a scattering
//      multiplies the weight by the single-scatter albedo (boundary-aware table, subsurface_walk_math.h) and draws a Henyey-Greenstein direction;
//   3. at a boundary hit from inside: next-event estimation of every analytic light through the Walter transmission BTDF (n_o^2 (1 - F) G D ...,
//      subsurfaceBtdfCos) with an optional shadow ray from the exit point, then the walk continues on the internal REFLECTION only, weight
//      F G1 (the VNDF estimator of the reflection lobe; the transmitted continuation leaves the object and contributes through the next-event
//      estimation).  The material's single-scattering GGX loses energy at each reflection exactly as Mitsuba's roughdielectric does.
// Russian roulette on the weight; every loop is bounded (GPUSubsurfaceWalkParams::maxSteps) and a walk that exhausts it contributes nothing
// (counted by the unit tests of the bound, not in the image).  The medium parameters are those of the entry pixel (texture-modulated colour and
// radius scale: a locally homogeneous medium), the exits use the interpolated vertex normals of the mesh.
#include <metal_stdlib>
#include "rt_common.h"
#include "renderer/normal_transform.h"
#include "renderer/subsurface_walk_math.h"
using namespace metal;
using namespace raytracing;
using namespace phosphor;

constexpr sampler kWalkSampler(filter::linear, mip_filter::linear, address::repeat);

inline uint swHash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
inline float swRand(thread uint& seed) { seed = swHash(seed + 0x9e3779b9u); return float(seed >> 8) * (1.0f / 16777216.0f); }

inline void swBasis(float3 n, thread float3& t, thread float3& b) {
    const float s = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z), c = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Visible-normal sample (Heitz 2018) of the isotropic GGX; `v` in the local frame, z up, v.z > 0.
inline float3 swSampleVndf(float3 v, float alpha, float u1, float u2) {
    const float3 vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 = lensq > 1e-12f ? float3(-vh.y, vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    const float3 t2 = cross(vh, t1);
    const float r = sqrt(u1), phi = 2.0f * M_PI_F * u2;
    const float a = r * cos(phi);
    const float s = 0.5f * (1.0f + vh.z);
    const float b = (1.0f - s) * sqrt(max(1.0f - a * a, 0.0f)) + s * r * sin(phi);
    const float3 nh = a * t1 + b * t2 + sqrt(max(1.0f - a * a - b * b, 0.0f)) * vh;
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(nh.z, 0.0f)));
}

struct SwHit {
    bool valid;
    float3 point;
    float3 ng; // outward geometric normal
    float3 ns; // outward interpolated shading normal
    float2 uv;
};

// The surface of a hit seen from INSIDE (the ray travels along `direction`): the outward normals face along the ray.
inline SwHit swSurface(GPURtHit hit, float3 direction, bool fromOutside, const device GPUInstance* instances, const device GPURtMesh* meshes,
                       const device GPUVertex* vertices, const device uint* indices, constant GPUSubsurfaceWalkParams& p) {
    SwHit out{};
    float3 point, ng;
    if (!rtSurface(hit, instances, meshes, vertices, indices, p.slotCount, p.meshCount, point, ng)) return out;
    const device GPUInstance& instance = instances[hit.slot];
    const device GPURtMesh& mesh = meshes[instance.meshIndex];
    const uint base = mesh.indexOffset + 3u * hit.primitive;
    const GPUVertex a = vertices[mesh.vertexOffset + indices[base]], b = vertices[mesh.vertexOffset + indices[base + 1u]],
                    c = vertices[mesh.vertexOffset + indices[base + 2u]];
    const float3 bary(1.0f - hit.u - hit.v, hit.u, hit.v);
    float3 n = float3(a.nx, a.ny, a.nz) * bary.x + float3(b.nx, b.ny, b.nz) * bary.y + float3(c.nx, c.ny, c.nz) * bary.z;
    const device float* m = instance.modelMatrix;
    const float3 x(m[0], m[1], m[2]), y(m[4], m[5], m[6]), z(m[8], m[9], m[10]);
    const auto transformed = phosphor::transformSurfaceNormal(x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z, n.x, n.y, n.z);
    float3 ns = float3(transformed.x, transformed.y, transformed.z);
    if (!(dot(ns, ns) > 1e-20f)) ns = ng;
    ns = normalize(ns);
    // outward: against the ray when it comes from outside, along it from inside
    const float sign = fromOutside ? -1.0f : 1.0f;
    if (dot(ng, direction) * sign < 0.0f) ng = -ng;
    if (dot(ns, ng) < 0.0f) ns = -ns;
    out.valid = all(isfinite(point)) && all(isfinite(ng)) && all(isfinite(ns));
    out.point = point; out.ng = ng; out.ns = ns;
    out.uv = float2(a.u, a.v) * bary.x + float2(b.u, b.v) * bary.y + float2(c.u, c.v) * bary.z;
    return out;
}

inline float swAttenuation(float dist, float range) {
    const float r = dist / max(range, 1e-3f);
    const float window = saturate(1.0f - r * r * r * r);
    return window * window / max(dist * dist, 1e-4f);
}

// Light `light` seen from `point`: direction towards it (unit), irradiance at normal incidence, distance to it (a large value for the sun).
inline bool swLight(const device GPULight& light, float3 point, thread float3& toLight, thread float3& radiance, thread float& distance) {
    const float3 color = float3(light.color[0], light.color[1], light.color[2]) * light.intensity;
    if (light.type == LIGHT_DIRECTIONAL) {
        toLight = -normalize(float3(light.direction[0], light.direction[1], light.direction[2]));
        radiance = color;
        distance = 1e30f;
        return true;
    }
    const float3 delta = float3(light.position[0], light.position[1], light.position[2]) - point;
    distance = length(delta);
    toLight = delta / max(distance, 1e-4f);
    float attenuation = swAttenuation(distance, light.range);
    if (light.type == LIGHT_SPOT) {
        const float3 spotDir = normalize(float3(light.direction[0], light.direction[1], light.direction[2]));
        attenuation *= smoothstep(cos(light.outerCone), cos(light.innerCone), dot(-toLight, spotDir));
    }
    radiance = color * attenuation;
    return true;
}

// One walk of channel c: returns the radiance estimate (before the factor 3 of the channel choice).
template<typename Payload>
inline float swWalk(float3 entryPoint, float3 entryNormalGeometric, float3 entryDirection, float entryWeight, float sigmaT, float albedo, float g, float alpha, uint channel, thread float& transfer,
                    constant GPUSubsurfaceWalkParams& p, const device GPULight* lights, instance_acceleration_structure as,
                    intersection_function_table<triangle_data, instancing> ift, const device GPUInstance* instances, const device GPURtMesh* meshes,
                    const device GPUVertex* vertices, const device uint* indices, thread uint& seed) {
    float3 x = rtOffsetRay(entryPoint, -entryNormalGeometric);
    float3 d = entryDirection;
    float weight = entryWeight;
    float radiance = 0.0f;
    const float eta = kSubsurfaceIor;
    for (uint step = 0u; step < p.maxSteps; ++step) {
        const float flight = -log(max(1.0f - swRand(seed), 1e-12f)) / sigmaT;
        GPURtRay ray{};
        ray.ox = x.x; ray.oy = x.y; ray.oz = x.z; ray.dx = d.x; ray.dy = d.y; ray.dz = d.z;
        ray.tmin = 0.0f; ray.tmax = flight; ray.mask = RT_MASK_ALL; ray.type = RT_PROBE_DIFFUSE; ray.coneWidth = 0.0f;
        Payload payload{};
        const GPURtHit hit = rtTraceSurface(ray, as, ift, instances, p.slotCount, payload, meshes, vertices, indices, p.meshCount);
        if (!hit.hit) {
            if (hit.t == -2.0f) break; // invalid ray
            // free flight ends inside the medium: scatter
            x += d * flight;
            weight *= albedo;
            const float cosine = subsurfaceHgCos(g, swRand(seed));
            const float sine = sqrt(max(1.0f - cosine * cosine, 0.0f)), phi = 2.0f * M_PI_F * swRand(seed);
            float3 t, b;
            swBasis(d, t, b);
            d = normalize(t * (sine * cos(phi)) + b * (sine * sin(phi)) + d * cosine);
        } else {
            const SwHit s = swSurface(hit, d, false, instances, meshes, vertices, indices, p);
            if (!s.valid) break;
            const float3 o = -d; // the direction back into the medium
            const float cosOn = dot(o, -s.ns);
            if (cosOn <= 1e-4f) break;
            // 3a. next-event estimation through the transmission BTDF
            if ((p.flags & SSS_WALK_NO_LIGHTS) == 0u) {
                uint count = p.lightCount;
                const uint first = count > 4u ? min(uint(swRand(seed) * float(count)), count - 1u) : 0u;
                const uint last = count > 4u ? first + 1u : count;
                const float lightScale = count > 4u ? float(count) : 1.0f;
                for (uint i = first; i < last; ++i) {
                    const device GPULight& light = lights[i];
                    if (light.type != LIGHT_DIRECTIONAL && (p.flags & SSS_WALK_NO_LOCAL) != 0u) continue;
                    float3 toLight, irradiance;
                    float distance;
                    if (!swLight(light, s.point, toLight, irradiance, distance)) continue;
                    if (dot(toLight, s.ng) <= 0.0f || dot(toLight, s.ns) <= 0.0f) continue;
                    // half vector of the transmission o (inside, index eta) -> i (outside, index 1): h = -(n_i i + n_o o), oriented outward
                    float3 m = -(toLight + eta * o);
                    const float ml = length(m);
                    if (!(ml > 1e-6f)) continue;
                    m /= ml;
                    if (dot(m, s.ns) < 0.0f) m = -m;
                    const float cosIm = dot(toLight, m), cosOm = -dot(o, m);
                    if (cosIm <= 0.0f || cosOm <= 0.0f) continue;
                    const float cosM = dot(m, s.ns);
                    const float fresnel = subsurfaceFresnel(cosOm, 1.0f / eta);
                    const float g1i = subsurfaceGgxG1(dot(toLight, s.ns), dot(toLight, m), alpha);
                    const float g1o = subsurfaceGgxG1(dot(o, -s.ns), cosOm, alpha); // o is on the inner side: both cosines flip together
                    const float f = subsurfaceBtdfCos(cosIm, cosOm, cosOn, cosM, alpha, 1.0f, eta, fresnel, g1i, g1o);
                    if (!(f > 0.0f)) continue;
                    float visibility = 1.0f;
                    if ((p.flags & SSS_WALK_SHADOW_RAYS) != 0u) {
                        const float3 origin = rtOffsetRay(s.point, s.ng);
                        GPURtRay shadow{};
                        shadow.ox = origin.x; shadow.oy = origin.y; shadow.oz = origin.z; shadow.dx = toLight.x; shadow.dy = toLight.y; shadow.dz = toLight.z;
                        shadow.tmin = 0.0f; shadow.tmax = distance > 1e29f ? 1e30f : max(distance - 1e-3f, 0.0f); shadow.mask = RT_MASK_SHADOW; shadow.type = RT_PROBE_SHADOW;
                        shadow.coneWidth = 0.0f;
                        Payload shadowPayload{};
                        const GPURtHit blocked = rtTrace(shadow, as, ift, instances, p.slotCount, shadowPayload);
                        if (blocked.hit || blocked.t == -2.0f) visibility = 0.0f;
                    }
                    radiance += weight * f * irradiance[channel] * lightScale * visibility;
                }
            }
            // 3b. continue on the internal reflection: VNDF sample of the inside direction in the frame whose z points into the medium
            float3 t, b;
            const float3 nIn = -s.ns;
            swBasis(nIn, t, b);
            const float3 vLocal(dot(o, t), dot(o, b), dot(o, nIn));
            const float3 mLocal = swSampleVndf(vLocal, alpha, swRand(seed), swRand(seed));
            const float cosVm = dot(vLocal, mLocal);
            if (cosVm <= 0.0f) break;
            const float reflectance = subsurfaceFresnel(cosVm, 1.0f / eta);
            // ambient dome (the resolve-owned environment / GI irradiance as a locally uniform radiance): the transmitted sample of the same
            // visible normal, weight (1 - F) G1 eta^2 (the entry's 1 / eta^2 cancels), no continuation
            {   // unit dome: the walk's response to a uniform environment of radiance 1 (the caller scales it by the ambient radiance)
                const float sin2 = eta * eta * (1.0f - cosVm * cosVm);
                if (sin2 < 1.0f) {
                    const float cosT = sqrt(1.0f - sin2);
                    const float3 tLocal = -eta * vLocal + (eta * cosVm - cosT) * mLocal; // travels out of the medium: z < 0 in this frame
                    if (tLocal.z < -1e-4f)
                        transfer += weight * (1.0f - reflectance) * subsurfaceGgxG1(tLocal.z, dot(tLocal, mLocal), alpha) * eta * eta;
                }
            }
            const float3 wiLocal = 2.0f * cosVm * mLocal - vLocal;
            if (wiLocal.z <= 1e-4f) break;
            const float g1 = subsurfaceGgxG1(wiLocal.z, dot(wiLocal, mLocal), alpha);
            weight *= reflectance * g1;
            d = normalize(t * wiLocal.x + b * wiLocal.y + nIn * wiLocal.z);
            if (dot(d, s.ng) >= 0.0f) break; // left through the geometry: the walk is lost
            x = rtOffsetRay(s.point, -s.ng);
        }
        if (!(weight > 0.0f)) break;
        if (step >= 3u && weight < 0.2f) { // Russian roulette keeps the weight in [0.2, entry weight]
            if (swRand(seed) > weight / 0.2f) break;
            weight = 0.2f;
        }
    }
    return radiance;
}

kernel void subsurface_walk(constant GPUSubsurfaceWalkParams& p [[buffer(0)]], constant FrameConstants& frame [[buffer(1)]],
                            const device GPUMaterial* materials [[buffer(2)]], instance_acceleration_structure as [[buffer(3)]],
                            intersection_function_table<triangle_data, instancing> ift [[buffer(4)]], const device GPUInstance* instances [[buffer(5)]],
                            const device GPURtMesh* meshes [[buffer(6)]], const device GPUVertex* vertices [[buffer(7)]],
                            const device uint* indices [[buffer(8)]], const device GPULight* lights [[buffer(9)]],
                            const device float* table [[buffer(10)]], const device RtTextureHandle* textures [[buffer(11)]],
                            texture2d<float, access::read> color [[texture(0)]], texture2d<float, access::read> sss [[texture(1)]],
                            texture2d<float, access::read> normalRoughness [[texture(2)]], texture2d<float, access::write> result [[texture(3)]],
                            texture2d<float, access::write> gainTexture [[texture(4)]], texture2d<float, access::write> termTexture [[texture(5)]],
                            uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight) return;
    const float4 base = color.read(pixel);
    const float4 own = sss.read(pixel);
    const uint material = uint(own.w);
    gainTexture.write(float4(0.0f), pixel);
    termTexture.write(float4(0.0f), pixel);
    if (material == 0u || pixel.x >= p.width || pixel.y >= p.height || p.samples == 0u) {
        result.write(base, pixel);
        return;
    }
    const float weight = saturate((own.w - float(material)) / 0.49f);
    const device GPUMaterial& m = materials[material - 1u];
    const GPUMaterialExt2 e2 = *reinterpret_cast<const device GPUMaterialExt2*>(frame.materialExtensions + m.extensionIndex + 2u);
    // the camera ray of the pixel centre
    const float3x3 viewRotation = float3x3(float3(frame.view[0], frame.view[1], frame.view[2]), float3(frame.view[4], frame.view[5], frame.view[6]),
                                           float3(frame.view[8], frame.view[9], frame.view[10]));
    const float2 ndc = (float2(pixel) + 0.5f) / float2(float(p.width), float(p.height)) * 2.0f - 1.0f;
    const float3 direction = normalize(transpose(viewRotation) * normalize(float3(ndc.x * p.tanHalfFovX, -ndc.y * p.tanHalfFovY, -1.0f)));
    const float3 origin(frame.cameraPosition[0], frame.cameraPosition[1], frame.cameraPosition[2]);
    GPURtRay cameraRay{};
    cameraRay.ox = origin.x; cameraRay.oy = origin.y; cameraRay.oz = origin.z; cameraRay.dx = direction.x; cameraRay.dy = direction.y; cameraRay.dz = direction.z;
    cameraRay.tmin = 0.0f; cameraRay.tmax = 1e30f; cameraRay.mask = RT_MASK_ALL; cameraRay.type = RT_PROBE_PRIMARY; cameraRay.coneWidth = 0.0f;
    RtPayloadLite cameraPayload{};
    const GPURtHit cameraHit = rtTraceSurface(cameraRay, as, ift, instances, p.slotCount, cameraPayload, meshes, vertices, indices, p.meshCount);
    if (!cameraHit.hit || cameraHit.slot >= p.slotCount || instances[cameraHit.slot].materialIndex != material - 1u) {
        result.write(base, pixel);
        return;
    }
    const SwHit entry = swSurface(cameraHit, direction, true, instances, meshes, vertices, indices, p);
    const float4 nr = normalRoughness.read(pixel);
    float3 shading = nr.xyz;
    if (!entry.valid || !(dot(shading, shading) > 0.25f)) {
        result.write(base, pixel);
        return;
    }
    shading = normalize(shading);
    const float roughness = clamp(nr.w, 0.04f, 1.0f);
    const float alpha = max(roughness * roughness, 0.002f);
    // the medium of the pixel (a locally homogeneous medium): per channel mean free path and single-scatter albedo
    // subsurface colour: material colour x colour texture (the resolve's target carries the ambient dome radiance instead)
    float3 colour(e2.color[0], e2.color[1], e2.color[2]);
    if (e2.colorTex != INVALID_TEXTURE_INDEX) colour *= float3(textures[e2.colorTex].tex.sample(kWalkSampler, entry.uv, level(0.0f)).rgb);
    const float3 ambientDome = (p.flags & SSS_WALK_NO_LIGHTS) != 0u ? float3(0.0f) : max(own.rgb, float3(0.0f));
    float3 radiusScale(e2.radiusScale[0], e2.radiusScale[1], e2.radiusScale[2]);
    if (e2.radiusScaleTex != INVALID_TEXTURE_INDEX)
        radiusScale *= float3(textures[e2.radiusScaleTex].tex.sample(kWalkSampler, entry.uv, level(0.0f)).rgb);
    float sigmaT[3], albedo[3];
    for (uint c = 0u; c < 3u; ++c) {
        sigmaT[c] = 1.0f / max(e2.radius * radiusScale[c], 1e-6f);
        albedo[c] = subsurfaceSingleScatterAlbedo(table, roughness, e2.anisotropy, colour[c]);
    }
    const float3 wo = -direction;
    float3 tangent, bitangent;
    swBasis(shading, tangent, bitangent);
    const float3 voutLocal(dot(wo, tangent), dot(wo, bitangent), dot(wo, shading));
    uint seed = swHash(pixel.x + p.width * pixel.y + 0x1234567u) ^ swHash(p.frameIndex * 7919u + 17u);
    float3 sum = float3(0.0f), transferSum = float3(0.0f);
    const float eta = kSubsurfaceIor;
    for (uint s = 0u; s < p.samples; ++s) {
        const uint channel = min(uint(swRand(seed) * 3.0f), 2u);
        if (voutLocal.z <= 1e-4f) continue;
        const float3 mLocal = swSampleVndf(voutLocal, alpha, swRand(seed), swRand(seed));
        const float cosI = dot(voutLocal, mLocal);
        if (cosI <= 0.0f) continue;
        const float fresnel = subsurfaceFresnel(cosI, eta);
        const float cosT = sqrt(max(1.0f - (1.0f - cosI * cosI) / (eta * eta), 0.0f));
        const float3 refracted = -voutLocal / eta + (cosI / eta - cosT) * mLocal; // travel direction in the frame of the shading normal
        if (refracted.z >= -1e-4f) continue;
        const float entryWeight = (1.0f - fresnel) * subsurfaceGgxG1(refracted.z, dot(refracted, mLocal), alpha) / (eta * eta);
        const float3 d = normalize(tangent * refracted.x + bitangent * refracted.y + shading * refracted.z);
        if (dot(d, entry.ng) >= 0.0f) continue;
        float transfer = 0.0f;
        const float walk = swWalk<RtPayloadLite>(entry.point, entry.ng, d, entryWeight, sigmaT[channel], albedo[channel], e2.anisotropy, alpha, channel, transfer, p,
                                                 lights, as, ift, instances, meshes, vertices, indices, seed);
        sum[channel] += 3.0f * (walk + transfer * ambientDome[channel]);
        transferSum[channel] += 3.0f * transfer;
    }
    // the composite-owned ambient / GI diffuse: rgb = w (1 - metallic) x the transfer factor, a = w (the walk code carries w (1 - metallic): metallic cancels in the
    // composite through its own (1 - metallic) base factor, so a = the weight of the diffuse substrate it replaces)
    if ((p.flags & SSS_WALK_NO_LIGHTS) == 0u) gainTexture.write(float4(weight * transferSum / float(p.samples), weight), pixel);
    termTexture.write(float4(weight * sum / float(p.samples), 1.0f), pixel);
    result.write(float4(base.rgb + weight * sum / float(p.samples), base.a), pixel);
}
