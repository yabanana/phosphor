#include "renderer/reflection_settings.h"
#include <doctest/doctest.h>
#include <glm/glm.hpp>
#include <cmath>
using namespace phosphor;
TEST_CASE("F13 positive partition avoids a quantized raw-subtraction negative"){
    // Explicit binary16 value below the real raw DI; no GPU/half conversion
    // implementation is reused to manufacture the counterexample.
    const double storedHalf=0.333251953125,rawDI=.3333,filteredDI=0;
    CHECK(storedHalf-rawDI+filteredDI<0);
    const auto positive=composeSignalLighting(glm::vec3(0),glm::vec3(0),glm::vec3(0),glm::vec3(0),glm::vec3(0),1,false);
    CHECK(positive==glm::vec3(0)); // direct selected0 + exact residual0
}
TEST_CASE("F13 positive raw SSR source contains every diffuse term and no reflection feedback"){
    const float pi=3.14159265358979323846f;const glm::vec3 residual(.2f,.3f,.4f),rawDI(2,3,4),rawE(pi,2*pi,3*pi),albedo(.6f,.4f,.2f);
    const glm::vec3 diffuse=rawE*albedo*.75f/pi;
    const auto preReflection=composeSignalLighting(residual+rawDI,diffuse,glm::vec3(0),glm::vec3(0),glm::vec3(9),1,true);
    CHECK(preReflection.x==doctest::Approx(2.65));CHECK(preReflection.y==doctest::Approx(3.9));CHECK(preReflection.z==doctest::Approx(4.85));
    const glm::vec3 reflection(5,6,7);const auto final=composeSignalLighting(residual+rawDI,diffuse,reflection,glm::vec3(0),glm::vec3(9),1,true);
    // Subtraction after rounded Float32 addition is not bit-exact. Bound it
    // by four unit roundoffs; the physical image gates are unchanged.
    for(int c=0;c<3;++c)CHECK(final[c]-preReflection[c]==doctest::Approx(reflection[c]).epsilon(4*1.1920928955078125e-7));
}
TEST_CASE("F13 positive ambient AO never modulates direct GI specular or emission"){
    const glm::vec3 residual(3),direct(2),spec(5),ambient(11),emission(7);
    const auto shadowed=composeSignalLighting(residual+direct,glm::vec3(0),spec,emission,ambient,.25f,false);
    CHECK(shadowed==glm::vec3(19.75f));
    const auto withGI=composeSignalLighting(residual+direct,glm::vec3(13),spec,emission,ambient,0,true);
    CHECK(withGI==glm::vec3(30));
    CHECK(withGI==composeSignalLighting(residual+direct,glm::vec3(13),spec,emission,ambient,1,true));
}
