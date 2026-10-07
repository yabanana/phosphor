#include "renderer/reflection_settings.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace phosphor {
namespace {
constexpr double pi=3.14159265358979323846;
glm::dvec3 v3(const float* a){return {a[0],a[1],a[2]};}
bool finite(glm::dvec3 a){return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z);}
glm::dvec3 tangent(glm::dvec3 n){return glm::normalize(glm::cross(std::abs(n.z)<0.999?glm::dvec3(0,0,1):glm::dvec3(0,1,0),n));}
}
bool validReflectionSettings(const ReflectionSettings& s){return std::isfinite(s.maxDistance)&&s.maxDistance>0&&
    std::isfinite(s.ssrThickness)&&s.ssrThickness>0&&std::isfinite(s.rtRoughnessLow)&&std::isfinite(s.rtRoughnessHigh)&&
    s.rtRoughnessLow>=0&&s.rtRoughnessHigh>s.rtRoughnessLow&&s.rtRoughnessHigh<=1&&s.ssrSteps>=2&&s.ssrSteps<=512&&s.ssrBinarySteps<=16;}
float reflectionRTWeight(float r,const ReflectionSettings& s){if(!(s.flags&REFLECTION_ENABLE_RT)||!std::isfinite(r))return 0;
    const float x=std::clamp((r-s.rtRoughnessLow)/(s.rtRoughnessHigh-s.rtRoughnessLow),0.f,1.f);return 1-x*x*(3-2*x);}
glm::dvec3 specularBRDF(const GPUDISurface& s,glm::dvec3 wi){
    auto n=v3(s.shadingNormal),v=v3(s.viewDirection);if(!s.valid||!finite(n)||!finite(v)||!finite(wi)||glm::dot(n,n)<=0||glm::dot(v,v)<=0||glm::dot(wi,wi)<=0)return {};
    n=glm::normalize(n);v=glm::normalize(v);wi=glm::normalize(wi);const double nl=glm::dot(n,wi),nv=glm::dot(n,v);
    if(nl<=0||nv<=0||glm::dot(v+wi,v+wi)<=1e-20)return {};
    const auto h=glm::normalize(v+wi);const double nh=std::max(0.0,glm::dot(n,h)),vh=std::max(0.0,glm::dot(v,h));
    const double a=std::max(double(s.roughness)*s.roughness,0.002),a2=a*a,d=nh*nh*(a2-1)+1;
    const double D=a2/(pi*d*d),gv=nl*std::sqrt(nv*nv*(1-a2)+a2),gl=nv*std::sqrt(nl*nl*(1-a2)+a2);
    const auto f0=glm::mix(glm::dvec3(0.04),v3(s.albedo),std::clamp(double(s.metallic),0.0,1.0));
    const auto F=f0+(glm::dvec3(1)-f0)*std::pow(1-vh,5.0);
    return D*(0.5/std::max(gv+gl,1e-5))*F;
}
SpecularDirection sampleSpecular(const GPUDISurface& s,glm::dvec2 u){
    SpecularDirection out;if(!s.valid||!std::isfinite(u.x)||!std::isfinite(u.y)||u.x<0||u.x>=1||u.y<0||u.y>=1)return out;
    auto n=v3(s.shadingNormal),v=v3(s.viewDirection);if(!finite(n)||!finite(v)||glm::dot(n,n)<=0||glm::dot(v,v)<=0)return out;
    n=glm::normalize(n);v=glm::normalize(v);const auto t=tangent(n),b=glm::cross(n,t);
    const double a=std::max(double(s.roughness)*s.roughness,0.002),a2=a*a,c=std::sqrt((1-u.x)/(1+(a2-1)*u.x));
    const double sinH=std::sqrt(std::max(0.0,1-c*c)),phi=2*pi*u.y;
    const auto h=t*(sinH*std::cos(phi))+b*(sinH*std::sin(phi))+n*c;
    const double vh=glm::dot(v,h);if(vh<=0)return out;
    out.direction=glm::reflect(-v,h);const double nl=glm::dot(n,out.direction),d=c*c*(a2-1)+1;
    out.pdf=(a2/(pi*d*d))*c/(4*vh);out.valid=finite(out.direction)&&std::isfinite(out.pdf)&&out.pdf>0;
    if(out.valid&&nl>0)out.weight=specularBRDF(s,out.direction)*nl/out.pdf;
    // A valid NDF proposal reflecting below the surface has zero weight.
    return out;
}
glm::dvec3 reflectionReference(const GPUDISurface& s,u32 phiSteps,u32 thetaSteps,IncidentRadiance light,void* user){
    if(!s.valid||!phiSteps||!thetaSteps||!light)return {};auto n=v3(s.shadingNormal);if(!finite(n)||glm::dot(n,n)<=0)return {};
    n=glm::normalize(n);const auto t=tangent(n),b=glm::cross(n,t);glm::dvec3 sum(0);
    // Independent uniform solid-angle hemisphere quadrature, not GGX sampling.
    for(u32 y=0;y<thetaSteps;++y)for(u32 x=0;x<phiSteps;++x){const double z=(y+0.5)/thetaSteps,r=std::sqrt(1-z*z),a=2*pi*(x+0.5)/phiSteps;
        const auto wi=t*(r*std::cos(a))+b*(r*std::sin(a))+n*z;sum+=specularBRDF(s,wi)*light(wi,user)*z;}
    return sum*(2*pi/(double(phiSteps)*thetaSteps));
}
std::optional<SSRHit> reflectionSSR(glm::vec3 origin,glm::vec3 direction,const glm::mat4& vp,const glm::mat4& inv,const glm::mat4& view,u32 w,u32 h,std::span<const float> depth,const ReflectionSettings& s){
    if(!validReflectionSettings(s)||!w||!h||u64(w)*h>depth.size()||!finite(glm::dvec3(origin))||!finite(glm::dvec3(direction))||glm::dot(direction,direction)<=0)return std::nullopt;
    direction=glm::normalize(direction);float previous=0;bool previousFront=false;
    auto evaluate=[&](float distance,SSRHit& hit,float& delta){const glm::vec3 point=origin+direction*distance;const glm::vec4 clip=vp*glm::vec4(point,1);
        if(!finite(glm::dvec3(clip))||!(clip.w>0))return false;const glm::vec2 uv=glm::vec2(clip)/clip.w*glm::vec2(.5f,-.5f)+.5f;
        if(glm::any(glm::lessThan(uv,glm::vec2(0)))||glm::any(glm::greaterThanEqual(uv,glm::vec2(1))))return false;
        const u32 x=u32(uv.x*w),y=u32(uv.y*h);const float z=depth[size_t(y)*w+x];if(!std::isfinite(z)||z<=0||z>1)return false;
        const glm::vec2 ndc=(glm::vec2(x,y)+.5f)*glm::vec2(2,-2)/glm::vec2(w,h)+glm::vec2(-1,1);const auto p=inv*glm::vec4(ndc,z,1);
        if(!std::isfinite(p.w)||std::abs(p.w)<1e-20)return false;hit.position=glm::vec3(p)/p.w;hit.pixel=y*w+x;hit.distance=distance;
        delta=-(view*glm::vec4(point,1)).z+(view*glm::vec4(hit.position,1)).z;return std::isfinite(delta);};
    for(u32 step=1;step<=s.ssrSteps;++step){const float fraction=float(step)/s.ssrSteps;const float distance=s.maxDistance*fraction*fraction;SSRHit hit;float d=0;
        if(!evaluate(distance,hit,d)){previous=distance;previousFront=false;continue;}
        if(previousFront&&d>=0){float low=previous,high=distance;for(u32 j=0;j<s.ssrBinarySteps;++j){SSRHit mid;float md=0;const float m=(low+high)*.5f;if(!evaluate(m,mid,md)||md<0)low=m;else{high=m;hit=mid;}}
            float last=0;if(evaluate(high,hit,last)&&last>=0&&last<=s.ssrThickness)return hit;}
        previous=distance;previousFront=d<0;
    }return std::nullopt;
}
bool validAOSettings(const AOSettings& s){return std::isfinite(s.radius)&&s.radius>0&&std::isfinite(s.originBias)&&s.originBias>=0&&
    std::isfinite(s.thickness)&&s.thickness>0&&s.rays>=1&&s.rays<=64&&s.slices>=1&&s.slices<=16&&s.steps>=1&&s.steps<=32;}
glm::vec3 aoDirection(glm::vec3 normal,glm::vec2 u){if(!finite(glm::dvec3(normal))||glm::dot(normal,normal)<=0||u.x<0||u.x>=1||u.y<0||u.y>=1)return {};
    const auto n=glm::normalize(normal);const auto t=glm::vec3(tangent(glm::dvec3(n))),b=glm::cross(n,t);const float r=std::sqrt(u.x),a=float(2*pi)*u.y;
    return t*(r*std::cos(a))+b*(r*std::sin(a))+n*std::sqrt(1-u.x);}
float aoReference(glm::vec3 n,float radius,u32 side,Occluded occluded,void* user){if(!side||!occluded||!std::isfinite(radius)||radius<=0)return 1;
    double sum=0;for(u32 y=0;y<side;++y)for(u32 x=0;x<side;++x)sum+=occluded(aoDirection(n,{(x+.5f)/side,(y+.5f)/side}),radius,user)?0:1;return float(sum/(double(side)*side));}
float aoHorizonVisibility(glm::vec3 n,glm::vec3 v,glm::vec3 t,float hp,float hn,u32 count){
    if(!count||!finite(glm::dvec3(n))||!finite(glm::dvec3(v))||!finite(glm::dvec3(t))||glm::dot(n,n)<=0||glm::dot(v,v)<=0||glm::dot(t,t)<=0||!std::isfinite(hp)||!std::isfinite(hn))return 1;
    n=glm::normalize(n);v=glm::normalize(v);t-=v*glm::dot(v,t);if(glm::dot(t,t)<=1e-20)return 1;t=glm::normalize(t);
    double visible=0,total=0;for(u32 i=0;i<count;++i){const double theta=-pi+2*pi*(i+.5)/count;
        const auto wi=glm::dvec3(v)*std::cos(theta)+glm::dvec3(t)*std::sin(theta);const double w=std::max(0.0,glm::dot(glm::dvec3(n),wi))*std::abs(std::sin(theta));
        total+=w;if(std::abs(theta)<(theta>=0?hp:hn))visible+=w;}
    return total>0?float(visible/total):1;
}
glm::vec3 composeSignalLighting(glm::vec3 direct,glm::vec3 gi,glm::vec3 spec,glm::vec3 emission,glm::vec3 ambient,float ao,bool giEnabled){
    return direct+gi+spec+emission+(giEnabled?glm::vec3(0):ambient*std::clamp(ao,0.f,1.f));}
} // namespace phosphor
