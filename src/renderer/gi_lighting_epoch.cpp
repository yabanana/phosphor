#include "renderer/gi_lighting_epoch.h"
#include <bit>
namespace phosphor {
u64 GiLightingEpoch::update(std::span<const GPULight> lights,const GiEnvironment& environment,u32 sampledRevision) {
    std::vector<u32> tuple;tuple.reserve(lights.size()*14+5);
    tuple.push_back(lights.size());tuple.push_back(sampledRevision);
    auto number=[&](float x){tuple.push_back(std::bit_cast<u32>(x));};
    for(const auto& light:lights) {
        tuple.push_back(light.type);for(float x:light.position)number(x);
        for(float x:light.direction)number(x);for(float x:light.color)number(x);
        number(light.intensity);number(light.range);number(light.innerCone);number(light.outerCone);
    }
    for(float x:environment.skyRadiance)number(x);number(environment.sunAngularRadius);
    if(tuple!=previous_ || external_!=environment.externalRevision){++epoch_;previous_=std::move(tuple);external_=environment.externalRevision;}
    return epoch_;
}
}
