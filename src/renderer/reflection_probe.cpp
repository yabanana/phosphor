#include "renderer/reflection_probe.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace phosphor {
namespace {
glm::vec3 v3(const float* p){return {p[0],p[1],p[2]};}bool finite(glm::vec3 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
float radicalInverse(u32 v){v=(v<<16)|(v>>16);v=((v&0x55555555u)<<1)|((v&0xaaaaaaaau)>>1);v=((v&0x33333333u)<<2)|((v&0xccccccccu)>>2);v=((v&0x0f0f0f0fu)<<4)|((v&0xf0f0f0f0u)>>4);v=((v&0x00ff00ffu)<<8)|((v&0xff00ff00u)>>8);return float(v)*2.3283064365386963e-10f;}
}
bool validReflectionProbe(const GPUReflectionProbe& p){return p.enabled&&finite(v3(p.boxMin))&&finite(v3(p.boxMax))&&finite(v3(p.capturePosition))&&
    glm::all(glm::greaterThan(v3(p.boxMax),v3(p.boxMin)))&&glm::all(glm::greaterThanEqual(v3(p.capturePosition),v3(p.boxMin)))&&glm::all(glm::lessThanEqual(v3(p.capturePosition),v3(p.boxMax)))&&
    std::isfinite(p.blendDistance)&&p.blendDistance>0&&std::isfinite(p.mipCount)&&p.mipCount>=1&&p.mipCount<=16&&std::floor(p.mipCount)==p.mipCount;}
float reflectionProbeWeight(const GPUReflectionProbe& p,glm::vec3 x){if(!validReflectionProbe(p)||!finite(x)||glm::any(glm::lessThan(x,v3(p.boxMin)))||glm::any(glm::greaterThan(x,v3(p.boxMax))))return 0;
    const auto d=glm::min(x-v3(p.boxMin),v3(p.boxMax)-x);return std::clamp(std::min({d.x,d.y,d.z})/p.blendDistance,0.f,1.f);}
std::optional<glm::vec3> reflectionProbeParallax(const GPUReflectionProbe& p,glm::vec3 x,glm::vec3 d){
    if(!validReflectionProbe(p)||!finite(x)||!finite(d)||glm::dot(d,d)<=0||glm::any(glm::lessThan(x,v3(p.boxMin)))||glm::any(glm::greaterThan(x,v3(p.boxMax))))return std::nullopt;
    d=glm::normalize(d);float exit=std::numeric_limits<float>::infinity();for(u32 i=0;i<3;++i)if(std::abs(d[i])>1e-8f)exit=std::min(exit,((d[i]>0?p.boxMax[i]:p.boxMin[i])-x[i])/d[i]);
    const auto corrected=x+d*exit-v3(p.capturePosition);return std::isfinite(exit)&&glm::dot(corrected,corrected)>1e-20f?std::optional(glm::normalize(corrected)):std::nullopt;
}
glm::vec3 reflectionCubeDirection(u32 face,glm::vec2 uv){const auto p=uv*2.f-1.f;switch(face){case 0:return glm::normalize(glm::vec3(1,-p.y,-p.x));case 1:return glm::normalize(glm::vec3(-1,-p.y,p.x));case 2:return glm::normalize(glm::vec3(p.x,1,p.y));case 3:return glm::normalize(glm::vec3(p.x,-1,-p.y));case 4:return glm::normalize(glm::vec3(p.x,-p.y,1));case 5:return glm::normalize(glm::vec3(-p.x,-p.y,-1));default:return {};}}
std::optional<CubeCoordinate> reflectionCubeCoordinate(glm::vec3 d){if(!finite(d)||glm::dot(d,d)<=0)return std::nullopt;const auto a=glm::abs(d);u32 face;glm::vec2 uv;
    if(a.x>=a.y&&a.x>=a.z){face=d.x>=0?0:1;uv={d.x>=0?-d.z:d.z,-d.y};uv/=a.x;}
    else if(a.y>=a.z){face=d.y>=0?2:3;uv={d.x,d.y>=0?d.z:-d.z};uv/=a.y;}
    else{face=d.z>=0?4:5;uv={d.z>=0?d.x:-d.x,-d.y};uv/=a.z;}return CubeCoordinate{face,uv*.5f+.5f};}
glm::mat4 reflectionProbeViewProjection(const GPUReflectionProbe& p,u32 face,float near,float far){if(!validReflectionProbe(p)||face>5||near<=0||far<=near)throw std::invalid_argument("invalid probe capture");
    const auto center=v3(p.capturePosition),forward=reflectionCubeDirection(face,{.5f,.5f});const glm::vec3 up=face==2?glm::vec3(0,0,1):face==3?glm::vec3(0,0,-1):glm::vec3(0,-1,0);
    // Reverse-Z zero-to-one, matching the engine. Independent of GLM depth mode.
    glm::mat4 projection(0);projection[0][0]=1;projection[1][1]=-1;projection[2][2]=near/(far-near);projection[2][3]=-1;projection[3][2]=far*near/(far-near);
    return projection*glm::lookAtRH(center,center+forward,up);
}
glm::vec3 reflectionProbePrefilter(glm::vec3 n,float roughness,u32 count,ProbeRadiance source,void* user){if(!source||!count||!finite(n)||glm::dot(n,n)<=0||!std::isfinite(roughness)||roughness<0||roughness>1)return {};
    n=glm::normalize(n);const auto t=glm::normalize(glm::cross(std::abs(n.z)<.999f?glm::vec3(0,0,1):glm::vec3(0,1,0),n)),b=glm::cross(n,t);
    if(roughness==0)return source(n,user);const float a=std::max(roughness*roughness,.002f),a2=a*a;glm::vec3 sum(0);float weights=0;
    for(u32 i=0;i<count;++i){const float u=(float(i)+.5f)/count,v=radicalInverse(i),c=std::sqrt((1-u)/(1+(a2-1)*u)),r=std::sqrt(std::max(0.f,1-c*c)),phi=6.28318530718f*v;
        const auto h=t*(r*std::cos(phi))+b*(r*std::sin(phi))+n*c,l=glm::reflect(-n,h);const float nl=std::max(0.f,glm::dot(n,l));sum+=source(l,user)*nl;weights+=nl;}
    return weights>0?sum/weights:glm::vec3(0);
}
glm::vec3 reflectionProbeContribution(const GPUDISurface& s,glm::vec3 L){
    if(!s.valid||!finite(L)||!finite(v3(s.shadingNormal))||!finite(v3(s.viewDirection))||glm::dot(v3(s.shadingNormal),v3(s.shadingNormal))<=0||glm::dot(v3(s.viewDirection),v3(s.viewDirection))<=0||!std::isfinite(s.roughness))return {};
    const float nv=std::max(0.f,glm::dot(glm::normalize(v3(s.shadingNormal)),glm::normalize(v3(s.viewDirection))));
    const glm::vec4 fit=std::clamp(s.roughness,0.f,1.f)*glm::vec4(-1,-.0275f,-.572f,.022f)+glm::vec4(1,.0425f,1.04f,-.04f);
    const float a=std::min(fit.x*fit.x,std::exp2(-9.28f*nv))*fit.x+fit.y;const glm::vec2 ab=glm::vec2(-1.04f,1.04f)*a+glm::vec2(fit.z,fit.w);
    const auto f0=glm::mix(glm::vec3(.04f),v3(s.albedo),std::clamp(s.metallic,0.f,1.f));return glm::max(L,glm::vec3(0))*(f0*ab.x+ab.y);
}
} // namespace phosphor
