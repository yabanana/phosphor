#include "renderer/emissive_domain.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace phosphor::di {
std::array<double,7> emitterLinearMetric(const float* m) {
    std::array<double,7> out{};u32 k=0;
    for(u32 a=0;a<3;++a)for(u32 b=a;b<3;++b) {
        double sum=0;for(u32 row=0;row<3;++row)sum+=double(m[4*a+row])*m[4*b+row];out[k++]=sum;
    }
    const double det=double(m[0])*(double(m[5])*m[10]-double(m[6])*m[9])-
                     double(m[4])*(double(m[1])*m[10]-double(m[2])*m[9])+
                     double(m[8])*(double(m[1])*m[6]-double(m[2])*m[5]);
    out[6]=std::isfinite(det)&&det!=0?(det>0?1.0:-1.0):0;
    return out;
}
namespace {
bool same(const std::array<double,7>& a,const std::array<double,7>& b,double tolerance) {
    if(a[6]==0 || a[6]!=b[6])return false;
    const double scale=std::max({a[0],a[3],a[5],b[0],b[3],b[5],1e-30});
    for(u32 k=0;k<6;++k)if(!std::isfinite(a[k])||!std::isfinite(b[k])||std::abs(a[k]-b[k])>tolerance*scale)return false;
    return true;
}
}
bool sameEmitterAreaDomain(const float* previous,const float* current,double tolerance){
    return std::isfinite(tolerance)&&tolerance>=0 && same(emitterLinearMetric(previous),emitterLinearMetric(current),tolerance);
}
double emitterWorldArea(const GPUEmissiveSurface& e,const float* m) {
    double u[3]{},v[3]{};
    for(u32 row=0;row<3;++row)for(u32 axis=0;axis<3;++axis){
        u[row]+=double(m[axis*4+row])*(double(e.p1[axis])-e.p0[axis]);
        v[row]+=double(m[axis*4+row])*(double(e.p2[axis])-e.p0[axis]);
    }
    const double x=u[1]*v[2]-u[2]*v[1],y=u[2]*v[0]-u[0]*v[2],z=u[0]*v[1]-u[1]*v[0];
    return 0.5*std::sqrt(x*x+y*y+z*z);
}
bool EmissiveDomainTracker::update(std::span<const GPUEmissiveSurface> emitters,std::span<const float> worlds) {
    bool changed=entries_.size()!=emitters.size();entries_.resize(emitters.size());
    for(u32 k=0;k<emitters.size();++k) {
        const auto& e=emitters[k];auto& old=entries_[k];Entry next;
        next.valid=e.valid && u64(e.instanceSlot)*16+16<=worlds.size();
        if(next.valid){next.slot=e.instanceSlot;next.generation=e.instanceGeneration;next.metric=emitterLinearMetric(worlds.data()+size_t(e.instanceSlot)*16);}
        if(old.valid!=next.valid || (next.valid && (old.slot!=next.slot || old.generation!=next.generation || !same(old.metric,next.metric,2e-5))))changed=true;
        old=next;
    }
    return changed;
}
}
