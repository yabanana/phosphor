#include <doctest/doctest.h>
#include "renderer/emission_pack.h"
#include <array>
#include <cmath>
#include <limits>
using namespace phosphor;
namespace {
std::array<float,3> roundTrip(float r,float g,float b){const auto p=packEmission(r,g,b);
    return {unpackEmissionChannel(p,0),unpackEmissionChannel(p,1),unpackEmissionChannel(p,2)};}
}
TEST_CASE("F13.1 emission pack: exact for fixture emissions, bounded error, never above the input") {
    CHECK(packEmission(0,0,0)==0u);
    const auto white=roundTrip(2.5f,2.5f,2.5f);for(float c:white)CHECK(c==2.5f);
    const auto wide=roundTrip(368640,128,64);CHECK(wide[0]==368640);CHECK(wide[1]==0);CHECK(wide[2]==0);
    const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
    const auto bad=roundTrip(nan,-1,inf);CHECK(bad[0]==0);CHECK(bad[1]==0);CHECK(bad[2]==0);
    for(float scale:{1e-5f,0.01f,0.37f,1.0f,8.0f,333.3f,65504.0f,500000.0f}){
        const std::array<float,3> in{scale,scale*0.4f,scale*0.025f};const auto out=roundTrip(in[0],in[1],in[2]);
        const float m=in[0];
        // Half a step: 2^-9 of the largest channel, 2^-22 at the smallest exponent.
        for(int c=0;c<3;++c){CHECK(out[c]>=0);CHECK(std::fabs(out[c]-in[c])<=std::fmax(m*0x1p-9f,0x1p-22f));}
    }
    // Rounding up to the next exponent (511.5 / 512 of a power of two).
    const auto edge=roundTrip(0.9995f,0,0);CHECK(std::fabs(edge[0]-0.9995f)<=0x1p-9f);
    // Saturation: never more than the clamped maximum.
    const auto big=roundTrip(1e30f,1e30f,0);CHECK(big[0]==EmissionPackMax);CHECK(big[1]==EmissionPackMax);CHECK(big[2]==0);
}
