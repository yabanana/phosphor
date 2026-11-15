#pragma once
#include "renderer/gpu_types.h"
#include "renderer/denoise_history_layout.h"
#include <glm/glm.hpp>
namespace phosphor {
struct DenoiseSettings {
    float temporalAlpha=.08f, momentsAlpha=.2f, depthThreshold=.02f, normalThreshold=.95f;
    float roughnessThreshold=.08f, hitDistanceThreshold=.1f, luminancePhi=4, normalPhi=64;
    float varianceFloor=1e-6f, clampSigma=2, planeThreshold=.02f;
    u32 maxHistory=32, atrousIterations=3;
    // Opt-in biased experiment. A noisy neighborhood's extrema do not bound
    // the expected radiance/visibility, even when every receiver is static.
    // The baseline rejects stale history by identity/geometry/revision only.
    bool clampHistory=false;
};
[[nodiscard]] bool validDenoiseSettings(const DenoiseSettings&);
[[nodiscard]] GPUDenoiseParams denoiseParameters(const DenoiseSettings&,u32 width,u32 height,u32 signal,u32 view,u32 epoch,u32 revision,bool reset);
// F15.1 (ticket 08) lobe contract of the specular denoiser (CPU mirror of the "_lobe" kernels in shaders/denoise.metal).
// The fields of the pixel's GPULobeGuide the rule reads; a null lobe pointer below selects the base rule bit for bit.
// volumeThickness: the guide's world thickness of the transmission volume (0 = thin-walled: the refract axis does not bend; the default keeps a
// bent axis for callers that only know the ior).
struct DenoiseLobeGuide { float ior=0, clearcoatRoughness=0, volumeThickness=1, transmissionFactor=0; };
// F15.1 (ticket 13, temporal clip): the key of a TRANSMISSIVE pixel (transmission factor >= 0.5: the refracted image dominates) is stable per
// pixel: the refract key (id 2, refract axis), whichever branch of the Fresnel reflect/refract one-sample estimator this frame drew.  The two
// branches are the two terms of one unbiased estimator of the pixel's radiance; keyed apart, the reflect branch (probability ~F, up to 1/32
// at the floor) rejected the history and the neighbours every frame it was drawn and stored a raw sample.  Other pixels keep the sampled key.
[[nodiscard]] u32 denoiseStableLobeKey(u32 sampledKey,const DenoiseLobeGuide&);
// A pixel as the lobe kernels see it: its sample flags (sampled lobe id and axis) and its guide.
struct DenoiseLobePixel { u32 sampleFlags=0; DenoiseLobeGuide guide{}; };
// Edge-stopping roughness of the pixel's lobe key.  Ticket 09: with a coat the key is STABLE per pixel (renderer/lobe_select.h: the coat
// when its albedo weight is significant, else the base lobe; the transport mixes the two lobes per sample, so the sampled lobe
// cannot be the key), and a significant coat takes the SHARPER of the coat and base roughnesses.
// Id 0 the base roughness, id 1 min(base roughness, the clearcoat roughness of the guide),
// id 2 (transmission) the base roughness scaled by clamp(2*ior-2, 0, 1) (ticket 13 refines it), id 3 reserved = base.
[[nodiscard]] float denoiseLobeRoughness(float baseRoughness,u32 lobeKey,const DenoiseLobeGuide&);
// F15.1 (ticket 11): edge-stopping roughness of an anisotropic pixel = the roughness of its SHARPER axis: sqrt(min(alpha_t, alpha_b)) with
// alpha_b = max(r^2, 0.002) (perpendicular to the lobe direction) and alpha_t = mix(alpha_b, 1, strength^2) >= alpha_b, i.e. the base roughness
// (up to the alpha floor): the highlight keeps the width of the narrow axis across the bitangent, so the lobe bound of the neighbours and
// the history roughness test must not widen to the stretched axis or to the effective environment roughness.  The key of an anisotropic
// pixel is the base lobe (id 0), whose rule already returns this value: the "_lobe" denoiser kernels need no anisotropy code.
[[nodiscard]] float denoiseAnisotropicRoughness(float baseRoughness,float anisotropyStrength);
// Spatial guide axis of the sampled lobe: the reflection axis, or for the refract axis the direction refracted at the
// shading normal with eta = 1/ior (ior < 1 means the default 1.5; total internal reflection falls back to the reflection axis).
[[nodiscard]] glm::vec3 denoiseLobeAxis(const GPUDISurface&,u32 lobeKey,const DenoiseLobeGuide&);
// surfaceChecks=false (F15.1 ticket 13, endpoint-matched history): the previous pixel was chosen by the refracted endpoint, so the surface normal,
// depth and plane of the receivers are not compared (identity, revision, roughness, lobe key and stochastic flag still are).
[[nodiscard]] bool denoiseCompatible(const GPUDISurface&,const DenoiseHistory&,const GPUDenoiseParams&,const GPUSpecularSample& specular={},const DenoiseLobeGuide* lobe=nullptr,bool surfaceChecks=true);
// F15.1 (ticket 13, temporal clip): CPU mirror of the endpoint window search of denoise_temporal_lobe (renderer/denoise_endpoint.h).
// The endpoint a sample carries (SPECULAR_SAMPLE_ENDPOINT): its key and point; false when it carries none.
[[nodiscard]] bool denoiseSampleEndpoint(const GPUSpecularSample&,u32& key,float point[3]);
// Footprint of the current endpoint field: the largest endpoint distance between the sample and its same-key neighbour samples (0 = none).
[[nodiscard]] float denoiseEndpointFootprint(const GPUSpecularSample& center,const GPUSpecularSample* neighbours,u32 count);
// The index of the candidate (a history specular extension per window pixel, scan order) whose endpoint is nearest the sample's and is accepted
// by the footprint rule, or -1.  Ties go to the lowest index (the shader scans the window in the same order).
[[nodiscard]] int denoiseEndpointSelect(const GPUSpecularSample& center,float footprint,const GPUDenoiseEndpointHistory* candidates,u32 count);
// neighborhoodMean/Sigma: the current 3x3 raw statistics, used by DENOISE_SOFT_CHANGE.
[[nodiscard]] DenoiseHistory denoiseTemporal(const GPUDISurface&,glm::vec3 current,const DenoiseHistory&,
    const GPUDenoiseParams&,glm::vec3 neighborhoodLow,glm::vec3 neighborhoodHigh,const GPUSpecularSample& specular={},float neighborhoodVariance=-1,
    glm::vec3 neighborhoodMean=glm::vec3(0),glm::vec3 neighborhoodSigma=glm::vec3(-1),const DenoiseLobeGuide* lobe=nullptr,
    bool endpointMatched=false,bool endpointRough=false); // endpointRough: matched by the endpoint of a ROUGH interface: soft test kept, blend floor kDenoiseEndpointRoughFloor / maxHistory.
    // F15.1 ticket 13: the history was matched by the endpoint of a SMOOTH interface: no soft test, the blend floor is 1 / maxHistory
[[nodiscard]] float denoiseSpatialWeight(const GPUDISurface&,const GPUDISurface&,float centerLuminance,float neighborLuminance,
    float variance,const GPUDenoiseParams&,const DenoiseLobePixel* lobeA=nullptr,const DenoiseLobePixel* lobeB=nullptr);
} // namespace phosphor
