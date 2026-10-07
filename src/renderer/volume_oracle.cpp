#include "renderer/volume_oracle.h"
#include "renderer/fog_settings.h"
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <limits>
#include <stdexcept>
namespace phosphor {
namespace {
constexpr double pi=3.14159265358979323846;
glm::dvec3 vec(const float* p){return {p[0],p[1],p[2]};}
AtmosphereSettings physics(const GPUAtmosphereParams& p){AtmosphereSettings a;a.planetCenter=vec(p.planetCenter);a.bottomRadius=p.bottomRadius;a.topRadius=p.topRadius;
    a.rayleighScaleHeight=p.rayleighScaleHeight;a.mieScaleHeight=p.mieScaleHeight;a.mieG=p.mieG;a.rayleighScattering=vec(p.rayleighScattering);a.mieScattering=vec(p.mieScattering);
    a.mieAbsorption=p.mieAbsorption;a.ozoneAbsorption=vec(p.ozoneAbsorption);a.ozoneCenterHeight=p.ozoneCenterHeight;a.ozoneHalfWidth=p.ozoneHalfWidth;a.groundAlbedo=vec(p.groundAlbedo);return a;}
glm::dvec3 quadrature(const std::function<glm::dvec3(double)>& f,double a,double b){
    constexpr double x[4]={0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363};
    constexpr double w[4]={0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763};
    glm::dvec3 result(0);const double middle=(a+b)*0.5,half=(b-a)*0.5;
    for(u32 i=0;i<4;++i)result+=(f(middle-half*x[i])+f(middle+half*x[i]))*(w[i]*half);return result;
}
glm::dvec3 skyDirection(const GPUAtmosphereParams& p,u32 x,u32 y){const glm::dvec3 relative=vec(p.cameraPosition)-vec(p.planetCenter),up=glm::normalize(relative);
    const double r=glm::length(relative),horizon=std::acos(-std::sqrt(std::max(0.0,(r-p.bottomRadius)*(r+p.bottomRadius)))/r);
    const double u=(double(x)+0.5)/p.skyWidth,v=(double(y)+0.5)/p.skyHeight;
    const double theta=v<0.5?horizon*(1-std::pow(1-2*v,2)):horizon+(pi-horizon)*std::pow(2*v-1,2),phi=u*2*pi;
    const auto east=glm::normalize(glm::cross(std::abs(up.z)<0.9?glm::dvec3(0,0,1):glm::dvec3(1,0,0),up)),north=glm::cross(up,east);
    return up*std::cos(theta)+(east*std::cos(phi)+north*std::sin(phi))*std::sin(theta);
}
glm::dvec3 skyReference(const GPUAtmosphereParams& p,u32 x,u32 y){const auto a=physics(p);const auto camera=vec(p.cameraPosition),ray=skyDirection(p,x,y),sun=vec(p.sunDirection),moon=vec(p.moonDirection);
    const auto path=atmosphereSegment(a,camera,ray);if(!path.valid)return glm::dvec3(0);
    glm::dvec3 L=atmosphereSingleScatteringReference(a,camera,ray,sun,vec(p.sunIrradiance),1e-5,8).radiance;
    L+=atmosphereSingleScatteringReference(a,camera,ray,moon,vec(p.moonIrradiance),1e-5,8).radiance;
    // Memoized independent angular quadrature at actual LUT grid nodes; CPU
    // never consumes GPU multiple-scattering values to manufacture its answer.
    std::map<std::pair<u32,u32>,glm::dvec3> nodes;
    const auto multiple=[&](glm::dvec3 point,glm::dvec3 light){const auto relative=point-a.planetCenter;const double r=glm::length(relative);
        const double px=std::clamp(glm::dot(relative,light)/r*0.5+0.5,0.0,1.0)*(p.multiWidth-1),py=std::clamp((r-a.bottomRadius)/(a.topRadius-a.bottomRadius),0.0,1.0)*(p.multiHeight-1);
        const u32 bx=std::min(u32(px),p.multiWidth-2),by=std::min(u32(py),p.multiHeight-2);const double fx=px-bx,fy=py-by;glm::dvec3 sum(0);
        for(u32 c=0;c<4;++c){const u32 xx=bx+(c&1u),yy=by+((c>>1u)&1u);const auto id=std::pair(xx,yy);auto found=nodes.find(id);
            if(found==nodes.end()){const double radius=a.bottomRadius+0.5+double(yy)/(p.multiHeight-1)*(a.topRadius-a.bottomRadius-1);
                found=nodes.emplace(id,atmosphereMultipleScatteringReference(a,radius,double(xx)/(p.multiWidth-1)*2-1,4,8)).first;}
            sum+=found->second*((c&1u)?fx:1-fx)*((c&2u)?fy:1-fy);}return sum;};
    L+=quadrature([&](double t){const auto point=camera+ray*t;const auto m=atmosphereMedium(a,point);
        const auto tau=quadrature([&](double d){return atmosphereMedium(a,camera+ray*d).extinction;},path.begin,t);
        return glm::exp(-tau)*(m.rayleigh+m.mie)*(vec(p.sunIrradiance)*multiple(point,sun)+vec(p.moonIrradiance)*multiple(point,moon));},path.begin,path.end);
    return L;
}
double fogDistance(const GPUFogParams& p,u32 index){const u32 x=index%p.gridX,y=(index/p.gridX)%p.gridY,z=index/(p.gridX*p.gridY);
    const glm::vec2 ndc((float(x)+0.5f)*2/float(p.gridX)-1,1-(float(y)+0.5f)*2/float(p.gridY));
    const glm::vec4 h=glm::make_mat4(p.inverseViewProjection)*glm::vec4(ndc,1,1);const glm::vec3 camera(p.cameraPosition[0],p.cameraPosition[1],p.cameraPosition[2]);
    const glm::vec3 ray=glm::normalize(glm::vec3(h)/h.w-camera);const double cosine=-(glm::make_mat4(p.view)*glm::vec4(ray,0)).z;
    const double far=p.nearDistance*std::pow(double(p.farDistance)/p.nearDistance,double(z+1)/p.gridZ);return (far-p.nearDistance)/cosine;
}
}
std::vector<VolumeOracleCase> evaluateVolumeOracle(const VolumeOracleInput& in,std::span<const GPUVolumeNumericSample> records,const VolumeOracleSettings& tolerances){
    std::vector<VolumeOracleCase> result;result.reserve(records.size());const auto& p=in.atmosphere;const auto a=physics(p);
    for(const auto& r:records){VolumeOracleCase c;c.x=r.x;c.y=r.y;c.index=r.x;c.actualEpoch=r.producedRevision;for(u32 i=0;i<4;++i)c.actual[i]=r.value[i];
        glm::dvec3 expected(0);double T=1;
        switch(r.kind){
        case VOLUME_NUMERIC_TRANS:{c.kind="transmittance";c.expectedEpoch=p.parameterRevision;c.absoluteTolerance=tolerances.transAbsolute;c.relativeTolerance=tolerances.transRelative;
            const auto ray=atmosphereTransmittanceRay(a,{double(r.x)/(p.transmittanceWidth-1),double(r.y)/(p.transmittanceHeight-1)});
            expected=atmosphereTransmittanceReference(a,a.planetCenter+glm::dvec3(0,ray.x,0),{std::sqrt(std::max(0.0,1-ray.y*ray.y)),ray.y,0},1e-8,14);break;}
        case VOLUME_NUMERIC_MULTI:{c.kind="multiscattering";c.expectedEpoch=p.parameterRevision;c.absoluteTolerance=tolerances.multipleAbsolute;c.relativeTolerance=tolerances.multipleRelative;
            const double radius=a.bottomRadius+0.5+double(r.y)/(p.multiHeight-1)*(a.topRadius-a.bottomRadius-1);expected=atmosphereMultipleScatteringReference(a,radius,double(r.x)/(p.multiWidth-1)*2-1,4,8);break;}
        case VOLUME_NUMERIC_SKY:c.kind="sky_view";c.expectedEpoch=p.skyRevision;c.absoluteTolerance=tolerances.skyAbsolute;c.relativeTolerance=tolerances.skyRelative;expected=skyReference(p,r.x,r.y);break;
        case VOLUME_NUMERIC_FOG:{if(!in.homogeneousFog)continue;c.kind="homogeneous_fog";c.expectedEpoch=in.fog.generation;c.absoluteTolerance=tolerances.fogAbsolute;c.relativeTolerance=tolerances.fogRelative;
            const auto answer=fogHomogeneous(in.diagnostics.fixtureExtinction,vec(in.diagnostics.fixtureSource),fogDistance(in.fog,r.x));expected=answer.radiance;T=answer.transmittance;break;}
        case VOLUME_NUMERIC_SOLAR:c.kind="solar_disk";c.expectedEpoch=p.skyRevision;c.absoluteTolerance=tolerances.solarAbsolute;c.relativeTolerance=tolerances.solarRelative;
            expected=r.x==0?vec(p.sunIrradiance)/(pi*std::pow(std::sin(double(p.sunAngularRadius)),2)):glm::dvec3(0);break;
        default:throw std::invalid_argument("unknown volume oracle record");}
        for(u32 i=0;i<3;++i)c.expected[i]=expected[i];c.expected[3]=T;c.passed=c.actualEpoch==c.expectedEpoch;
        for(u32 i=0;i<4;++i){const double error=std::abs(c.actual[i]-c.expected[i]),relative=error/std::max(std::abs(c.expected[i]),1e-8);
            if(!std::isfinite(c.actual[i])){c.absoluteError=std::numeric_limits<double>::infinity();c.relativeError=c.absoluteError;c.passed=false;continue;}
            c.absoluteError=std::max(c.absoluteError,error);c.relativeError=std::max(c.relativeError,relative);
            c.passed&=std::isfinite(c.actual[i])&&error<=c.absoluteTolerance+c.relativeTolerance*std::abs(c.expected[i]);}result.push_back(c);
    }return result;
}
} // namespace phosphor
