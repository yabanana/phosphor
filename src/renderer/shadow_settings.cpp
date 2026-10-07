#include "renderer/shadow_settings.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <glm/gtc/type_ptr.hpp>

namespace phosphor {
namespace {
bool finite(glm::vec3 p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); }
glm::vec3 unit(glm::vec3 p) {
    if (!finite(p) || !(glm::dot(p,p) > 1e-20f)) throw std::invalid_argument("invalid shadow direction");
    return glm::normalize(p);
}
void lightBasis(glm::vec3 light, glm::vec3& right, glm::vec3& up) {
    const glm::vec3 helper = std::abs(light.y) < 0.95f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    right = glm::normalize(glm::cross(helper, light));
    up = glm::cross(light, right);
}
}
void validateShadowSettings(const ShadowSettings& s) {
    if (s.mapResolution < 16 || s.mapResolution > 8192 || s.blockerSamples < 1 || s.blockerSamples > 64 ||
        s.filterSamples < 1 || s.filterSamples > 64 || s.historySamples < 1 || s.historySamples > 64 ||
        s.contactSteps < 1 || s.contactSteps > 64 || s.cacheCapacity < 1 || s.cacheCapacity > 4096 ||
        !(s.distance > 0 && std::isfinite(s.distance)) || !(s.casterReach >= 0 && std::isfinite(s.casterReach)) ||
        !(s.splitLambda >= 0 && s.splitLambda <= 1) || !(s.sunAngularRadius >= 0 && s.sunAngularRadius < 0.25f) ||
        !(s.depthBiasWorld >= 0 && std::isfinite(s.depthBiasWorld)) ||
        !(s.normalBiasWorld >= 0 && std::isfinite(s.normalBiasWorld)) ||
        !(s.temporalPositionThreshold >= 0 && std::isfinite(s.temporalPositionThreshold)) ||
        !(s.temporalNormalThreshold >= -1 && s.temporalNormalThreshold <= 1) ||
        !(s.contactDistance >= 0 && std::isfinite(s.contactDistance)) ||
        !(s.contactThickness > 0 && std::isfinite(s.contactThickness)) ||
        !(s.contactStrength >= 0 && s.contactStrength <= 1))
        throw std::invalid_argument("invalid experimental shadow settings");
}
std::array<float,5> shadowCascadeSplits(float nearPlane, float farPlane, float lambda) {
    if (!(nearPlane > 0 && farPlane > nearPlane && std::isfinite(farPlane) && lambda >= 0 && lambda <= 1))
        throw std::invalid_argument("invalid cascade range");
    std::array<float,5> result{nearPlane,0,0,0,farPlane};
    for (u32 i=1;i<4;++i) {
        const float t=float(i)/4.0f;
        const float logarithmic=nearPlane*std::pow(farPlane/nearPlane,t);
        const float uniform=nearPlane+(farPlane-nearPlane)*t;
        result[i]=lambda*logarithmic+(1-lambda)*uniform;
    }
    return result;
}
std::array<GPUShadowCascade,4> makeShadowCascades(const ShadowCamera& camera, glm::vec3 towardLight,
                                                const ShadowSettings& settings,
                                                std::span<const ShadowBounds> casterBounds) {
    validateShadowSettings(settings);
    const auto splits=shadowCascadeSplits(camera.nearPlane,settings.distance,settings.splitLambda);
    const glm::vec3 forward=unit(camera.forward), light=unit(towardLight);
    if (!finite(camera.position)) throw std::invalid_argument("invalid shadow camera");
    glm::vec3 right,up;
    lightBasis(light,right,up);
    std::array<glm::vec3,4> rays{};
    for (u32 i=0;i<4;++i) {
        const glm::vec4 h=camera.inverseViewProjection*glm::vec4(i&1u?1.0f:-1.0f,i&2u?1.0f:-1.0f,1,1);
        if (!std::isfinite(h.w) || std::abs(h.w)<1e-20f) throw std::invalid_argument("invalid inverse projection");
        const glm::vec3 ray=glm::vec3(h)/h.w-camera.position;
        const float d=glm::dot(ray,forward);
        if (!finite(ray) || !(d>1e-20f)) throw std::invalid_argument("shadow camera forward disagrees with projection");
        rays[i]=ray/d; // scale by positive VIEW distance, not ray length
    }
    std::array<GPUShadowCascade,4> result{};
    for (u32 ci=0;ci<4;++ci) {
        std::array<glm::vec3,8> corners{};
        glm::vec3 center(0);
        for(u32 i=0;i<8;++i) { corners[i]=camera.position+rays[i&3u]*(i>=4?splits[ci+1]:(ci==0?splits[ci]:splits[ci]-(splits[ci]-splits[ci-1])*0.1f)); center+=corners[i]/8.0f; }
        float radius=0;
        for(auto p:corners) radius=std::max(radius,glm::length(p-center));
        // Quantized bounding sphere keeps rotational footprint stable. Margin
        // covers half-texel snapping in both axes, including the map border.
        radius=std::ceil(radius*16.0f)/16.0f;
        radius*=float(settings.mapResolution)/float(settings.mapResolution-2);
        const float texel=2*radius/float(settings.mapResolution);
        const float x=std::floor(glm::dot(center,right)/texel+0.5f)*texel;
        const float y=std::floor(glm::dot(center,up)/texel+0.5f)*texel;
        center+=right*(x-glm::dot(center,right))+up*(y-glm::dot(center,up));
        float zMin=std::numeric_limits<float>::max(), zMax=-zMin;
        for(auto p:corners) { zMin=std::min(zMin,glm::dot(light,p)); zMax=std::max(zMax,glm::dot(light,p)); }
        // Off-camera upstream casters remain in the light volume. If actual
        // scene bounds are provided, use them in addition to the finite preset.
        zMax+=settings.casterReach;
        for(const auto& b:casterBounds) {
            if (!finite(b.minimum)||!finite(b.maximum)||glm::any(glm::greaterThan(b.minimum,b.maximum)))
                throw std::invalid_argument("invalid caster bounds");
            for(u32 i=0;i<8;++i) {
                const glm::vec3 p(i&1u?b.maximum.x:b.minimum.x,i&2u?b.maximum.y:b.minimum.y,i&4u?b.maximum.z:b.minimum.z);
                zMin=std::min(zMin,glm::dot(light,p)); zMax=std::max(zMax,glm::dot(light,p));
            }
        }
        zMin-=std::max(texel,settings.depthBiasWorld+settings.normalBiasWorld);
        zMax+=texel;
        const float range=std::max(zMax-zMin,1e-4f);
        glm::mat4 m(1);
        for(u32 axis=0;axis<3;++axis) { m[axis][0]=right[axis]/radius; m[axis][1]=up[axis]/radius; m[axis][2]=light[axis]/range; }
        m[3][0]=-x/radius; m[3][1]=-y/radius; m[3][2]=-zMin/range;
        auto& c=result[ci];
        std::copy_n(glm::value_ptr(m),16,c.viewProjection);
        c.splitNear=splits[ci]; c.splitFar=splits[ci+1]; c.texelWorld=texel; c.depthRange=range;
        c.center[0]=center.x; c.center[1]=center.y; c.center[2]=center.z; c.radius=radius;
        c.depthMin=zMin; c.depthMax=zMax; c.biasWorld=settings.depthBiasWorld; c.normalBiasWorld=settings.normalBiasWorld;
    }
    return result;
}
u32 shadowCascadeIndex(float depth,const std::array<GPUShadowCascade,4>& cascades) {
    for(u32 i=0;i<4;++i) if(depth>=cascades[i].splitNear && depth<=cascades[i].splitFar) return i;
    return ~0u;
}
std::array<float,3> shadowSolarDirection(glm::vec3 towardLight,float radius,float u,float v) {
    if(!(radius>=0 && radius<0.25f && u>=0 && u<1 && v>=0 && v<1)) throw std::invalid_argument("invalid solar sample");
    const glm::vec3 light=unit(towardLight);
    glm::vec3 right,up; lightBasis(light,right,up);
    // Uniform in solid angle over the solar disk: p(omega)=1/[2pi(1-cos a)].
    const float cosTheta=1-u*(1-std::cos(radius));
    const float sinTheta=std::sqrt(std::max(0.0f,1-cosTheta*cosTheta));
    const float phi=6.283185307179586f*v;
    const glm::vec3 d=light*cosTheta+(right*std::cos(phi)+up*std::sin(phi))*sinTheta;
    return {d.x,d.y,d.z};
}
u64 shadowDirtyTiles(const GPUShadowCascade& c,const ShadowBounds& b) {
    if(!finite(b.minimum)||!finite(b.maximum)||glm::any(glm::greaterThan(b.minimum,b.maximum))) return ~u64(0);
    glm::vec2 lo(std::numeric_limits<float>::max()),hi(-std::numeric_limits<float>::max());
    const glm::mat4 m=glm::make_mat4(c.viewProjection);
    for(u32 i=0;i<8;++i) {
        const glm::vec3 p(i&1u?b.maximum.x:b.minimum.x,i&2u?b.maximum.y:b.minimum.y,i&4u?b.maximum.z:b.minimum.z);
        const glm::vec4 clip=m*glm::vec4(p,1);
        if(!std::isfinite(clip.x)||!std::isfinite(clip.y)) return ~u64(0);
        const glm::vec2 uv=glm::vec2(clip.x,-clip.y)*0.5f+0.5f;
        lo=glm::min(lo,uv); hi=glm::max(hi,uv);
    }
    if(hi.x<0 || hi.y<0 || lo.x>1 || lo.y>1) return 0;
    const float guard=c.texelWorld/std::max(2*c.radius,1e-6f);
    lo-=glm::vec2(guard); hi+=glm::vec2(guard);
    // One texel guard accounts for PCSS blockers across a tile boundary. The
    // consumer must additionally expand bounds by its maximum filter radius.
    const int x0=std::clamp(int(std::floor(lo.x*8)),0,7),x1=std::clamp(int(std::floor(hi.x*8)),0,7);
    const int y0=std::clamp(int(std::floor(lo.y*8)),0,7),y1=std::clamp(int(std::floor(hi.y*8)),0,7);
    u64 mask=0;
    for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) mask|=u64(1)<<(y*8+x);
    return mask;
}

ShadowStaticCache::ShadowStaticCache(u32 capacity):entries_(capacity) {
    if(capacity==0 || capacity>4096) throw std::invalid_argument("invalid shadow cache capacity");
}
void ShadowStaticCache::beginFrame(u64 frame,u64 completed,u32 budget) {
    if(frame<frame_ || completed<completed_) throw std::invalid_argument("shadow cache timeline regressed");
    frame_=frame; completed_=completed; budget_=budget; updates_=0;
    for(auto& e:entries_) e.reserved=false;
}
ShadowCacheDecision ShadowStaticCache::request(ShadowCacheKey key,ShadowCacheRevision revision) {
    if(key.cascade>=4 || key.tile>=64) throw std::out_of_range("shadow cache key");
    u32 found=~0u;
    for(u32 i=0;i<entries_.size();++i) if(entries_[i].occupied && entries_[i].key==key) { found=i; break; }
    if(found!=~0u) {
        auto& e=entries_[found]; e.touched=frame_;
        if(e.valid && e.stored==revision) return {ShadowCacheAction::Cached,found,e.writer};
        if(e.reserved) return {ShadowCacheAction::DynamicFallback,found,0};
    }
    if(updates_>=budget_) return {};
    if(found==~0u) {
        u64 oldest=std::numeric_limits<u64>::max();
        for(u32 i=0;i<entries_.size();++i) {
            const auto& e=entries_[i];
            if(e.reserved || (e.occupied && std::max(e.reader,e.writer)>completed_)) continue;
            if(!e.occupied) { found=i; break; }
            if(e.touched<oldest) { found=i; oldest=e.touched; }
        }
        if(found==~0u) return {};
    }
    auto& e=entries_[found];
    const u64 wait=std::max(e.writer,e.reader);
    e.key=key; e.occupied=true; e.valid=false; e.reserved=true; e.requested=revision; e.touched=frame_;
    ++updates_;
    return {ShadowCacheAction::Update,found,wait};
}
void ShadowStaticCache::publish(u32 index,ShadowCacheRevision revision,u64 submission) {
    auto& e=entries_.at(index);
    if(!e.reserved || e.requested!=revision || submission<std::max(e.writer,e.reader))
        throw std::logic_error("obsolete or unordered shadow cache publication");
    e.stored=revision; e.writer=submission; e.valid=true; e.reserved=false;
}
void ShadowStaticCache::read(u32 index,u64 submission) {
    auto& e=entries_.at(index);
    if(!e.valid || submission<e.writer || submission<e.reader) throw std::logic_error("unordered shadow cache read");
    e.reader=submission;
}
void ShadowStaticCache::invalidate(ShadowCacheKey key) { for(auto& e:entries_) if(e.occupied && e.key==key) { e.valid=false; e.reserved=false; } }
void ShadowStaticCache::invalidateLight(u32 light) { for(auto& e:entries_) if(e.occupied && e.key.light==light) { e.valid=false; e.reserved=false; } }
void ShadowStaticCache::clear() {
    for(const auto& e:entries_) if(std::max(e.reader,e.writer)>completed_) throw std::logic_error("shadow cache still in flight");
    std::fill(entries_.begin(),entries_.end(),Entry{});
}
} // namespace phosphor
