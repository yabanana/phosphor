#pragma once

#include "renderer/gpu_types.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>

namespace phosphor {
struct GiReceiver { glm::vec3 position{}, normal{0,1,0}, diffuse{1}; };
struct GiConnection {
    glm::vec3 contribution{}; // integrand in secondary AREA measure
    float target = 0, proposalArea = 0, geometry = 0;
    bool valid = false;
};
inline glm::vec3 giVector(const float p[3]) { return {p[0],p[1],p[2]}; }
inline float giLuminance(glm::vec3 L) { return glm::dot(L,glm::vec3(0.2126f,0.7152f,0.0722f)); }
inline bool giFinite(glm::vec3 x) { return std::isfinite(x.x)&&std::isfinite(x.y)&&std::isfinite(x.z); }

// Diffuse-only reconnection shift. Secondary vertex y remains fixed in WORLD
// space with fixed area; Jacobian of the area->area shift is exactly 1. q_omega
// transforms ONCE to q_area=q_omega*abs(n_y dot -wi)/r^2 at initial sampling.
// A source/receiver BRDF-cosine is included in target, not in L_y twice.
inline GiConnection giConnection(const GiReceiver& x, const GPUGiReservoir& y, bool visible) {
    GiConnection c;
    if (!visible || !(y.flags&GI_SAMPLE_VALID) || !giFinite(x.position)||!giFinite(x.normal)||
        !giFinite(x.diffuse)||glm::dot(x.normal,x.normal)<1e-12f) return c;
    const glm::vec3 delta=giVector(y.position)-x.position;
    const float r2=glm::dot(delta,delta);
    const glm::vec3 ny=giVector(y.normal), L=giVector(y.radiance);
    if (!(r2>1e-12f) || !giFinite(ny)||!giFinite(L)||glm::any(glm::lessThan(L,glm::vec3(0)))||
        glm::dot(ny,ny)<1e-12f) return c;
    const auto wi=delta/std::sqrt(r2);
    const float cosX=std::max(0.f,glm::dot(glm::normalize(x.normal),wi));
    const float cosY=std::max(0.f,glm::dot(glm::normalize(ny),-wi));
    c.geometry=cosX*cosY/r2;
    c.contribution=x.diffuse*L*(c.geometry/3.14159265358979323846f);
    c.target=giLuminance(c.contribution);
    c.proposalArea=(cosX/3.14159265358979323846f)*cosY/r2;
    c.valid=std::isfinite(c.target)&&c.target>0&&c.proposalArea>0;
    return c;
}
inline void giFinalize(GPUGiReservoir& r) {
    r.W=(r.M&&r.target>0&&std::isfinite(r.weightSum)) ? r.weightSum/(float(r.M)*r.target) : 0;
    if (!(r.W>0&&std::isfinite(r.W))) { r.W=0; r.flags=0; }
}
// Adds a fresh independent path. Zero-contribution/blocked paths count in M;
// skipping them would condition the estimator on visibility and add energy.
inline void giAddCandidate(GPUGiReservoir& reservoir, GPUGiReservoir candidate,
                           float target, float proposalArea, float uniformRandom) {
    if(reservoir.M==~0u) return;
    ++reservoir.M;
    if (!(std::isfinite(target)&&target>0&&std::isfinite(proposalArea)&&proposalArea>0)) return;
    const float weight=target/proposalArea;
    if (!std::isfinite(weight)) return;
    const float sum=reservoir.weightSum+weight;
    if (!std::isfinite(sum)) return;
    if (std::clamp(uniformRandom,0.f,0.99999994f)*sum<weight) {
        const u32 M=reservoir.M;
        candidate.target=target; candidate.proposalArea=proposalArea;
        candidate.sourceProposalArea=proposalArea; candidate.flags|=GI_SAMPLE_VALID;
        reservoir=candidate; reservoir.M=M;
    }
    reservoir.weightSum=sum;
}
// Basic (biased) ReSTIR GI reuse estimator: w=t_receiver(y)*W_source*M_source,
// W_final=sum(w)/(M_total*t_selected). Temporal/spatial correlation and differing
// source visibility support are NOT claimed unbiased. This explicit bounded
// version remains experimental and must be judged against independent reference.
// Fresh-only RIS is the reference path. Visibility is re-traced for every shift;
// a blocked shift contributes zero but STILL contributes M_source to normalizer.
inline bool giMerge(GPUGiReservoir& dst, const GPUGiReservoir& src, float receiverTarget,
                    bool validShift, float uniformRandom, u32 maxHistoryM=32) {
    if (!(src.flags&GI_SAMPLE_VALID) || !(src.W>0) || !src.M || !maxHistoryM) return false;
    const u32 M=std::min(src.M,maxHistoryM);
    const u32 total=dst.M+M;
    if (total<dst.M) return false;
    dst.M=total;
    if (!validShift || !(std::isfinite(receiverTarget)&&receiverTarget>0)) return false;
    const float weight=receiverTarget*src.W*float(M), sum=dst.weightSum+weight;
    if (!(std::isfinite(weight)&&weight>0&&std::isfinite(sum))) return false;
    const bool selected=std::clamp(uniformRandom,0.f,0.99999994f)*sum<weight;
    if (selected) { dst=src; dst.M=total; dst.target=receiverTarget; dst.age=src.age+1; }
    dst.weightSum=sum;
    return selected;
}
inline bool giHistoryCompatible(const GPUGiReservoir& r, const GPUProbeGridParams& p,
                                const GiReceiver& receiver, float positionTolerance, float normalCos=0.95f) {
    return (r.flags&GI_SAMPLE_VALID) && r.age<32 && r.geometryRevision==p.geometryRevision &&
        r.lightRevision==p.lightRevision && r.materialRevision==p.materialRevision &&
        r.viewRevision==p.viewRevision && giFinite(receiver.position)&&giFinite(receiver.normal)&&
        glm::length(giVector(r.sourcePosition)-receiver.position)<=positionTolerance &&
        glm::dot(giVector(r.sourceNormal),receiver.normal)>=normalCos;
}
inline glm::vec3 giShade(const GPUGiReservoir& r, const GiConnection& connection) {
    return connection.valid && std::isfinite(r.W) && r.W>0 ? connection.contribution*r.W : glm::vec3(0);
}
} // namespace phosphor
