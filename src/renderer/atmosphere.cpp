#include "renderer/atmosphere.h"
#include "renderer/volume_noise.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

namespace phosphor {
namespace {
constexpr double pi=3.14159265358979323846;
bool finite(glm::dvec3 v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);}
glm::dvec3 direction(glm::dvec3 d){if(!finite(d)||!(glm::dot(d,d)>1e-24))throw std::invalid_argument("invalid atmosphere ray direction");return glm::normalize(d);}
std::array<double,2> sphere(glm::dvec3 p,glm::dvec3 d,double radius) {
    const double r=glm::length(p),b=glm::dot(p,d),c=(r-radius)*(r+radius),disc=b*b-c;
    if(disc<0)return {-1,-1};
    const double root=std::sqrt(std::max(0.0,disc)),q=-b-std::copysign(root,b);
    if(std::abs(q)<1e-30)return {-b,-b};
    const double a=q,z=c/q;return {std::min(a,z),std::max(a,z)};
}
glm::dvec3 integralFactor(glm::dvec3 sigma,double ds){glm::dvec3 r;for(u32 c=0;c<3;++c)r[c]=sigma[c]>1e-12?-std::expm1(-sigma[c]*ds)/sigma[c]:ds;return r;}
glm::dvec3 integrateSimpson(const std::function<glm::dvec3(double)>& fn,double a,double b,double tolerance,u32 depth) {
    const auto fa=fn(a),fb=fn(b),fm=fn((a+b)*0.5),whole=(b-a)/6*(fa+4.0*fm+fb);
    std::function<glm::dvec3(double,double,glm::dvec3,glm::dvec3,glm::dvec3,glm::dvec3,double,u32)> split;
    split=[&](double x,double y,glm::dvec3 left,glm::dvec3 mid,glm::dvec3 right,glm::dvec3 old,double eps,u32 remaining) {
        const double center=(x+y)*0.5;const auto l=fn((x+center)*0.5),r=fn((center+y)*0.5);
        const auto sl=(center-x)/6*(left+4.0*l+mid),sr=(y-center)/6*(mid+4.0*r+right),error=sl+sr-old;
        const double e=std::max({std::abs(error.x),std::abs(error.y),std::abs(error.z)});
        if(!remaining||e<15*eps)return sl+sr+error/15.0;
        return split(x,center,left,l,mid,sl,eps*0.5,remaining-1)+split(center,y,mid,r,right,sr,eps*0.5,remaining-1);
    };
    return split(a,b,fa,fm,fb,whole,tolerance,depth);
}
void bits(std::vector<u64>& key,double d){key.push_back(std::bit_cast<u64>(d));}
void bits(std::vector<u64>& key,glm::dvec3 d){bits(key,d.x);bits(key,d.y);bits(key,d.z);}
glm::dvec3 celestial(double latitude,double declination,double hour){return direction({-std::cos(declination)*std::sin(hour),
    std::sin(latitude)*std::sin(declination)+std::cos(latitude)*std::cos(declination)*std::cos(hour),
    std::cos(latitude)*std::sin(declination)-std::sin(latitude)*std::cos(declination)*std::cos(hour)});}
double fraction(double x){return x-std::floor(x);}
double luminance(glm::dvec3 x){return glm::dot(x,glm::dvec3(0.2126,0.7152,0.0722));}
void copy3(float* p,glm::dvec3 v){p[0]=float(v.x);p[1]=float(v.y);p[2]=float(v.z);}
}
void validateAtmosphere(const AtmosphereSettings& s) {
    if(!finite(s.planetCenter)||!(s.bottomRadius>1&&s.topRadius>s.bottomRadius&&s.topRadius<=1e9)||
       !(s.rayleighScaleHeight>0&&std::isfinite(s.rayleighScaleHeight))||!(s.mieScaleHeight>0&&std::isfinite(s.mieScaleHeight))||
       !(std::abs(s.mieG)<0.99)||!(s.mieAbsorption>=0&&std::isfinite(s.mieAbsorption))||
       !(s.ozoneCenterHeight>=0&&std::isfinite(s.ozoneCenterHeight))||!(s.ozoneHalfWidth>0&&std::isfinite(s.ozoneHalfWidth))||
       !s.marchSteps||s.marchSteps>512||!s.multiDirections||s.multiDirections>128)
        throw std::invalid_argument("invalid metre-based atmosphere preset");
    for(auto v:{s.rayleighScattering,s.mieScattering,s.ozoneAbsorption,s.groundAlbedo})
        if(!finite(v)||glm::any(glm::lessThan(v,glm::dvec3(0))))throw std::invalid_argument("invalid atmosphere coefficient");
    if(glm::any(glm::greaterThan(s.groundAlbedo,glm::dvec3(1))))throw std::invalid_argument("atmosphere albedo exceeds one");
    for(u32 d:{s.transmittanceWidth,s.transmittanceHeight,s.multiWidth,s.multiHeight,s.skyWidth,s.skyHeight})
        if(d<2||d>2048)throw std::invalid_argument("invalid atmosphere LUT dimension");
}
AtmosphereMedium atmosphereMedium(const AtmosphereSettings& s,glm::dvec3 world) {
    if(!finite(world))throw std::invalid_argument("nonfinite atmospheric world point");
    const double height=std::max(0.0,glm::length(world-s.planetCenter)-s.bottomRadius);
    if(height>s.topRadius-s.bottomRadius)return {};
    const double rayleigh=std::exp(-height/s.rayleighScaleHeight),mie=std::exp(-height/s.mieScaleHeight);
    const double ozone=std::clamp(1.0-std::abs(height-s.ozoneCenterHeight)/s.ozoneHalfWidth,0.0,1.0);
    AtmosphereMedium m;m.rayleigh=s.rayleighScattering*rayleigh;m.mie=s.mieScattering*mie;
    m.extinction=m.rayleigh+m.mie+glm::dvec3(s.mieAbsorption*mie)+s.ozoneAbsorption*ozone;return m;
}
AtmosphereSegment atmosphereSegment(const AtmosphereSettings& s,glm::dvec3 world,glm::dvec3 ray,double limit) {
    if(!finite(world)||!(limit>=0&&std::isfinite(limit)))throw std::invalid_argument("invalid atmosphere ray");
    const auto d=direction(ray);const auto p=world-s.planetCenter;const double radius=glm::length(p);
    if(radius<s.bottomRadius-1e-3)return {0,0,true,false};
    if(radius<=s.bottomRadius+1e-3&&glm::dot(p,d)<0)return {0,0,true,true};
    const auto top=sphere(p,d,s.topRadius);if(top[1]<=0)return {};
    AtmosphereSegment out;out.begin=std::max(0.0,top[0]);out.end=std::min(limit,top[1]);
    const auto bottom=sphere(p,d,s.bottomRadius);
    const double ground=bottom[0]>1e-3?bottom[0]:bottom[1]>1e-3&&radius<s.bottomRadius?bottom[1]:-1;
    if(ground>=out.begin&&ground<=out.end){out.end=ground;out.ground=true;}
    out.valid=out.end>=out.begin;return out;
}
glm::dvec3 atmosphereTransmittance(const AtmosphereSettings& s,glm::dvec3 p,glm::dvec3 ray,u32 steps) {
    if(!steps||steps>65536)throw std::invalid_argument("invalid atmosphere quadrature count");
    const auto d=direction(ray);const auto segment=atmosphereSegment(s,p,d);
    if(segment.ground)return glm::dvec3(0);if(!segment.valid)return glm::dvec3(1);
    const double ds=(segment.end-segment.begin)/steps;glm::dvec3 tau(0);
    for(u32 i=0;i<steps;++i)tau+=atmosphereMedium(s,p+d*(segment.begin+(i+0.5)*ds)).extinction*ds;
    return glm::exp(-tau);
}
glm::dvec3 atmosphereTransmittanceReference(const AtmosphereSettings& s,glm::dvec3 p,glm::dvec3 ray,double tolerance,u32 depth) {
    if(!(tolerance>0&&std::isfinite(tolerance))||depth>24)throw std::invalid_argument("invalid reference quadrature bound");
    const auto d=direction(ray);const auto segment=atmosphereSegment(s,p,d);
    if(segment.ground)return glm::dvec3(0);if(!segment.valid)return glm::dvec3(1);
    const auto tau=integrateSimpson([&](double t){return atmosphereMedium(s,p+d*t).extinction;},segment.begin,segment.end,tolerance,depth);
    return glm::exp(-tau);
}
double atmosphereRayleighPhase(double mu){return 3.0/(16*pi)*(1+mu*mu);}
double atmosphereMiePhase(double mu,double g){return (1-g*g)/(4*pi*std::pow(std::max(1e-12,1+g*g-2*g*mu),1.5));}
AtmosphereIntegral atmosphereSingleScattering(const AtmosphereSettings& s,glm::dvec3 p,glm::dvec3 ray,
                                              glm::dvec3 sun,glm::dvec3 irradiance,u32 steps,double limit,bool includeGroundBoundary) {
    if(!steps||steps>512||!finite(irradiance)||glm::any(glm::lessThan(irradiance,glm::dvec3(0))))throw std::invalid_argument("invalid atmosphere integration");
    const auto d=direction(ray),light=direction(sun);const auto segment=atmosphereSegment(s,p,d,limit);
    AtmosphereIntegral out;out.distance=segment.end;out.ground=segment.ground;if(!segment.valid)return out;
    const double span=segment.end-segment.begin,mu=glm::dot(d,light);
    for(u32 i=0;i<steps;++i) {
        const double a=double(i)/steps,b=double(i+1)/steps,t0=segment.begin+span*a*a,t1=segment.begin+span*b*b,ds=t1-t0;
        const auto point=p+d*((t0+t1)*0.5);const auto m=atmosphereMedium(s,point);const auto factor=integralFactor(m.extinction,ds);
        const auto source=irradiance*atmosphereTransmittance(s,point,light,128)*
                          (m.rayleigh*atmosphereRayleighPhase(mu)+m.mie*atmosphereMiePhase(mu,s.mieG));
        out.radiance+=out.transmittance*source*factor;out.scatteringFactor+=out.transmittance*(m.rayleigh+m.mie)*factor;
        out.transmittance*=glm::exp(-m.extinction*ds);
    }
    if(segment.ground&&includeGroundBoundary) {
        const auto point=p+d*segment.end,normal=direction(point-s.planetCenter);
        out.radiance+=out.transmittance*s.groundAlbedo/pi*irradiance*atmosphereTransmittance(s,point+normal*0.01,light,256)*std::max(0.0,glm::dot(normal,light));
    }
    return out;
}
namespace {
AtmosphereIntegral referenceIntegral(const AtmosphereSettings& s,glm::dvec3 p,glm::dvec3 ray,glm::dvec3 sun,
                                     glm::dvec3 E,double tolerance,u32 depth,double limit,bool isotropic,bool includeGroundBoundary) {
    if(!(tolerance>0&&std::isfinite(tolerance))||depth>18)throw std::invalid_argument("invalid radiance reference bound");
    const auto d=direction(ray),light=direction(sun);const auto segment=atmosphereSegment(s,p,d,limit);
    AtmosphereIntegral out;out.ground=segment.ground;out.distance=segment.end;if(!segment.valid)return out;
    const auto optical=[&](double t){return integrateSimpson([&](double distance){return atmosphereMedium(s,p+d*distance).extinction;},segment.begin,t,tolerance*0.25,depth);};
    const double mu=glm::dot(d,light);
    out.radiance=integrateSimpson([&](double t){const auto point=p+d*t;const auto m=atmosphereMedium(s,point);
        const auto phase=isotropic?(m.rayleigh+m.mie)/(4*pi):m.rayleigh*atmosphereRayleighPhase(mu)+m.mie*atmosphereMiePhase(mu,s.mieG);
        return glm::exp(-optical(t))*phase*E*atmosphereTransmittanceReference(s,point,light,tolerance*0.25,depth);},segment.begin,segment.end,tolerance,depth);
    out.transmittance=glm::exp(-optical(segment.end));
    out.scatteringFactor=integrateSimpson([&](double t){const auto m=atmosphereMedium(s,p+d*t);return glm::exp(-optical(t))*(m.rayleigh+m.mie);},segment.begin,segment.end,tolerance,depth);
    if(segment.ground&&includeGroundBoundary){const auto point=p+d*segment.end,normal=direction(point-s.planetCenter);
        out.radiance+=out.transmittance*s.groundAlbedo/pi*E*atmosphereTransmittanceReference(s,point+normal*0.01,light,tolerance*0.25,depth)*std::max(0.0,glm::dot(normal,light));
        if(isotropic)out.scatteringFactor+=out.transmittance*s.groundAlbedo;}
    return out;
}
}
AtmosphereIntegral atmosphereSingleScatteringReference(const AtmosphereSettings& s,glm::dvec3 p,glm::dvec3 ray,
                                                       glm::dvec3 sun,glm::dvec3 E,double tolerance,u32 depth,double limit,bool includeGroundBoundary) {
    if(!finite(E)||glm::any(glm::lessThan(E,glm::dvec3(0))))throw std::invalid_argument("invalid reference irradiance");
    return referenceIntegral(s,p,ray,sun,E,tolerance,depth,limit,false,includeGroundBoundary);
}
glm::dvec3 atmosphereMultipleScatteringReference(const AtmosphereSettings& s,double radius,double mu,u32 polar,u32 azimuth) {
    if(polar<2||polar>32||azimuth<2||azimuth>64||!(radius>=s.bottomRadius&&radius<=s.topRadius)||!(mu>=-1&&mu<=1))
        throw std::invalid_argument("invalid multiple-scattering reference quadrature");
    const glm::dvec3 point=s.planetCenter+glm::dvec3(0,radius,0),sun(std::sqrt(std::max(0.0,1-mu*mu)),mu,0);
    glm::dvec3 first(0),returned(0);
    for(u32 node=0;node<polar;++node) {
        double z=std::cos(pi*(double(node)+0.75)/(double(polar)+0.5)),derivative=0;
        for(u32 iteration=0;iteration<16;++iteration){double p0=1,p1=z;for(u32 l=2;l<=polar;++l){const double p2=((2*double(l)-1)*z*p1-(double(l)-1)*p0)/double(l);p0=p1;p1=p2;}
            derivative=double(polar)*(z*p1-p0)/(z*z-1);const double delta=p1/derivative;z-=delta;if(std::abs(delta)<1e-14)break;}
        const double weight=2/((1-z*z)*derivative*derivative),radial=std::sqrt(std::max(0.0,1-z*z));
        for(u32 j=0;j<azimuth;++j){const double phi=2*pi*(double(j)+0.5)/azimuth;const auto ray=glm::dvec3(radial*std::cos(phi),z,radial*std::sin(phi));
            const auto result=referenceIntegral(s,point,ray,sun,glm::dvec3(1),1e-5,8,1e12,true,true);
            first+=result.radiance*(weight/(2*azimuth));returned+=result.scatteringFactor*(weight/(2*azimuth));}
    }
    return first/glm::max(glm::dvec3(1e-3),glm::dvec3(1)-returned);
}
glm::dvec2 atmosphereTransmittanceUv(const AtmosphereSettings& s,double radius,double mu) {
    radius=std::clamp(radius,s.bottomRadius,s.topRadius);mu=std::clamp(mu,-1.0,1.0);
    const double H=std::sqrt((s.topRadius-s.bottomRadius)*(s.topRadius+s.bottomRadius)),rho=std::sqrt(std::max(0.0,(radius-s.bottomRadius)*(radius+s.bottomRadius)));
    const double d=-radius*mu+std::sqrt(std::max(0.0,radius*radius*(mu*mu-1)+s.topRadius*s.topRadius));
    const double dmin=s.topRadius-radius,dmax=rho+H;
    return {std::clamp((d-dmin)/std::max(dmax-dmin,1e-12),0.0,1.0),rho/H};
}
glm::dvec2 atmosphereTransmittanceRay(const AtmosphereSettings& s,glm::dvec2 uv) {
    uv=glm::clamp(uv,glm::dvec2(0),glm::dvec2(1));
    const double H=std::sqrt((s.topRadius-s.bottomRadius)*(s.topRadius+s.bottomRadius)),rho=H*uv.y,r=std::sqrt(rho*rho+s.bottomRadius*s.bottomRadius);
    const double dmin=s.topRadius-r,d=dmin+uv.x*(rho+H-dmin);
    return {r,d>1e-10?std::clamp((s.topRadius*s.topRadius-r*r-d*d)/(2*r*d),-1.0,1.0):1.0};
}

DayNightClock::DayNightClock(DayNightSettings settings):settings_(settings) {
    if(!(settings.dayLengthSeconds>0&&std::isfinite(settings.dayLengthSeconds))||!std::isfinite(settings.startDayFraction)||
       !(std::abs(settings.latitudeRadians)<=pi/2)||!(std::abs(settings.solarDeclinationRadians)<=pi/2)||
       !(settings.lunarOrbitDays>0&&std::isfinite(settings.lunarOrbitDays))||!std::isfinite(settings.lunarPhaseOffset)||
       !(std::abs(settings.lunarInclinationRadians)<=pi/2)||!(settings.jumpThresholdSeconds>0&&std::isfinite(settings.jumpThresholdSeconds))||
       !(settings.sunAngularRadius>0&&settings.sunAngularRadius<0.25)||!(settings.moonAngularRadius>0&&settings.moonAngularRadius<0.25)||
       !(settings.starIntensity>=0&&std::isfinite(settings.starIntensity))||!finite(settings.sunIrradiance)||!finite(settings.moonFullIrradiance)||
       glm::any(glm::lessThan(settings.sunIrradiance,glm::dvec3(0)))||glm::any(glm::lessThan(settings.moonFullIrradiance,glm::dvec3(0))))
        throw std::invalid_argument("invalid single celestial clock settings");
}
DayNightState DayNightClock::sample(double seconds,bool explicitJump) {
    if(!std::isfinite(seconds))throw std::invalid_argument("nonfinite celestial clock");
    DayNightState state;state.seconds=seconds;state.deltaSeconds=valid_?seconds-previous_:0;
    state.reset=!valid_||explicitJump||state.deltaSeconds<0||std::abs(state.deltaSeconds)>settings_.jumpThresholdSeconds;
    if(state.reset)++epoch_;state.epoch=epoch_;valid_=true;previous_=seconds;
    const double day=seconds/settings_.dayLengthSeconds+settings_.startDayFraction;
    state.dayFraction=fraction(day);const double hour=2*pi*(state.dayFraction-0.5);
    state.sunDirection=celestial(settings_.latitudeRadians,settings_.solarDeclinationRadians,hour);
    const double lunar=fraction(day/settings_.lunarOrbitDays+settings_.lunarPhaseOffset),lunarHour=hour-2*pi*lunar;
    state.moonDirection=celestial(settings_.latitudeRadians,settings_.solarDeclinationRadians+settings_.lunarInclinationRadians*std::sin(2*pi*lunar),lunarHour);
    const double phaseAngle=std::acos(std::clamp(-glm::dot(state.sunDirection,state.moonDirection),-1.0,1.0));
    state.moonPhase=(std::sin(phaseAngle)+(pi-phaseAngle)*std::cos(phaseAngle))/pi;
    state.sunIrradiance=settings_.sunIrradiance;state.moonIrradiance=settings_.moonFullIrradiance*state.moonPhase;
    state.starRotation=2*pi*fraction(day*1.00273790935);
    const auto E=state.sunIrradiance*std::max(0.0,state.sunDirection.y)+state.moonIrradiance*std::max(0.0,state.moonDirection.y);
    const double meterL=std::max(1e-8,luminance(E)*0.18/pi+settings_.starIntensity);
    state.exposureEv100=std::log2(meterL*100/12.5);return state;
}
std::array<GPULight,2> atmosphereDirectionalLights(const AtmosphereSettings& a,const DayNightState& state,glm::dvec3 point) {
    std::array<GPULight,2> lights{};
    for(u32 i=0;i<2;++i) {
        const auto d=i?state.moonDirection:state.sunDirection,E=i?state.moonIrradiance:state.sunIrradiance;
        const auto value=E*atmosphereTransmittance(a,point,d,256);
        lights[i].type=LIGHT_DIRECTIONAL;copy3(lights[i].direction,-d);copy3(lights[i].color,value);lights[i].intensity=1;
        lights[i].shadowMapIndex=i?~0u:0;
    }
    return lights;
}
double proceduralStarRadiance(glm::dvec3 ray,double rotation,double footprint,u32 seed) {
    const auto d=direction(ray);const double u=fraction(std::atan2(d.z,d.x)/(2*pi)+rotation/(2*pi))*256,v=std::acos(std::clamp(d.y,-1.0,1.0))/pi*128;
    const int ix=int(std::floor(u)),iy=int(std::floor(v));double result=0;
    for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
        const int cx=ix+x,cy=iy+y;if(cy<0||cy>=128)continue;
        const u32 h=volumeHash(u32(cx)&255u,u32(cy),0,seed);if((h&127u)!=0)continue;
        const double px=cx+volumeRandom(volumeHash(h,1,0,seed)),py=cy+volumeRandom(volumeHash(h,2,0,seed));
        const double dx=(u-px)*2*pi/256*std::max(0.02,std::sin(v*pi/128)),dy=(v-py)*pi/128;
        const double radius=std::max(footprint,0.00035),brightness=0.25+volumeRandom(h)*2;
        result+=brightness*std::exp(-(dx*dx+dy*dy)/(2*radius*radius));
    }
    return result;
}
AtmosphereUpdate AtmosphereVersions::update(const AtmosphereSettings& a,const DayNightState& state,glm::dvec3 camera,bool force) {
    validateAtmosphere(a);if(!finite(camera))throw std::invalid_argument("invalid sky-view camera");
    std::vector<u64> physics;physics.reserve(48);bits(physics,a.planetCenter);bits(physics,a.bottomRadius);bits(physics,a.topRadius);
    bits(physics,a.rayleighScaleHeight);bits(physics,a.mieScaleHeight);bits(physics,a.mieG);bits(physics,a.rayleighScattering);bits(physics,a.mieScattering);
    bits(physics,a.mieAbsorption);bits(physics,a.ozoneAbsorption);bits(physics,a.ozoneCenterHeight);bits(physics,a.ozoneHalfWidth);bits(physics,a.groundAlbedo);
    for(u32 x:{a.transmittanceWidth,a.transmittanceHeight,a.multiWidth,a.multiHeight,a.skyWidth,a.skyHeight,a.marchSteps,a.multiDirections})physics.push_back(x);
    std::vector<u64> sky=physics;bits(sky,camera);bits(sky,state.sunDirection);bits(sky,state.sunIrradiance);bits(sky,state.moonDirection);bits(sky,state.moonIrradiance);
    sky.push_back(state.epoch);AtmosphereUpdate out;out.transmittance=force||physics_!=physics;out.multiscattering=out.transmittance;
    out.skyView=force||out.transmittance||sky_!=sky;
    if(out.transmittance)++parameterRevision_;if(out.skyView)++skyRevision_;physics_=std::move(physics);sky_=std::move(sky);
    out.parameterRevision=parameterRevision_;out.skyRevision=skyRevision_;return out;
}
GPUAtmosphereParams makeAtmosphereParams(const AtmosphereSettings& a,const DayNightSettings& settings,const DayNightState& state,
                                         const AtmosphereUpdate& revision,glm::dvec3 camera,const float* inverse,const float* vp,u32 w,u32 h,u32 frame,u32 view) {
    validateAtmosphere(a);if(!inverse||!vp||!w||!h||!finite(camera))throw std::invalid_argument("invalid atmosphere frame");
    GPUAtmosphereParams p{};copy3(p.planetCenter,a.planetCenter);p.bottomRadius=float(a.bottomRadius);p.topRadius=float(a.topRadius);
    p.rayleighScaleHeight=float(a.rayleighScaleHeight);p.mieScaleHeight=float(a.mieScaleHeight);p.mieG=float(a.mieG);
    copy3(p.rayleighScattering,a.rayleighScattering);copy3(p.mieScattering,a.mieScattering);p.mieAbsorption=float(a.mieAbsorption);
    copy3(p.ozoneAbsorption,a.ozoneAbsorption);p.ozoneCenterHeight=float(a.ozoneCenterHeight);p.ozoneHalfWidth=float(a.ozoneHalfWidth);copy3(p.groundAlbedo,a.groundAlbedo);
    copy3(p.sunDirection,state.sunDirection);copy3(p.sunIrradiance,state.sunIrradiance);p.sunAngularRadius=float(settings.sunAngularRadius);
    copy3(p.moonDirection,state.moonDirection);copy3(p.moonIrradiance,state.moonIrradiance);p.moonAngularRadius=float(settings.moonAngularRadius);p.moonPhase=float(state.moonPhase);
    copy3(p.cameraPosition,camera);p.timeSeconds=float(state.seconds);p.timeDelta=float(state.deltaSeconds);
    std::memcpy(p.inverseViewProjection,inverse,64);std::memcpy(p.viewProjection,vp,64);
    p.transmittanceWidth=a.transmittanceWidth;p.transmittanceHeight=a.transmittanceHeight;p.multiWidth=a.multiWidth;p.multiHeight=a.multiHeight;
    p.skyWidth=a.skyWidth;p.skyHeight=a.skyHeight;p.marchSteps=a.marchSteps;p.multiDirections=a.multiDirections;p.outputWidth=w;p.outputHeight=h;
    p.parameterRevision=revision.parameterRevision;p.skyRevision=revision.skyRevision;p.frameIndex=frame;p.viewID=view;
    p.flags=ATMOSPHERE_ENABLE_MOON|ATMOSPHERE_ENABLE_STARS;p.starIntensity=float(settings.starIntensity);p.starRotation=float(state.starRotation);p.exposureEv100=float(state.exposureEv100);
    return p;
}
} // namespace phosphor
