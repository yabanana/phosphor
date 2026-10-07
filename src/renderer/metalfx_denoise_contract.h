#pragma once
#include "renderer/metalfx_denoise_layout.h"
#include "rendergraph/render_graph.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

namespace phosphor::metalfx_denoise {
// No production scene-linear radiometric domain has passed native acceptance.
// ControlledFixtureDiagnostic permits measurements; it is not certification.
enum class RadiometricDomain : u8 { UnqualifiedSceneLinear,ControlledFixtureDiagnostic };
[[nodiscard]] constexpr bool permitsNativeEncoding(RadiometricDomain domain) {
    return domain==RadiometricDomain::ControlledFixtureDiagnostic;
}
[[nodiscard]] constexpr const char* radiometricDomainName(RadiometricDomain domain) {
    return domain==RadiometricDomain::ControlledFixtureDiagnostic?"controlled-fixture-diagnostic":"unqualified-scene-linear";
}
inline constexpr const char* UnqualifiedRadiometryReason=
    "UnqualifiedRadiometricDomain: native scene-linear/HDR denoising is not qualified; custom Float32 selected before graph encoding. Native SDK execution is limited to controlled diagnostic fixtures.";

enum class NormalSpace : u8 { World, View, Tangent };
enum class NormalEncoding : u8 { SignedUnit, Unorm };
enum class MotionUnits : u8 { InputPixels, NormalizedUV, ClipNdc };
enum class MotionDirection : u8 { CurrentToPrevious, PreviousToCurrent };
enum class Roughness : u8 { LinearPerceptual, SquaredAlpha, GammaEncoded };
enum class Color : u8 { LinearCompositeRadiance, IsolatedSignal, DisplayEncoded };
struct Semantics {
    NormalSpace normalSpace=NormalSpace::World;
    NormalEncoding normalEncoding=NormalEncoding::SignedUnit;
    MotionUnits motionUnits=MotionUnits::InputPixels;
    MotionDirection motionDirection=MotionDirection::CurrentToPrevious;
    Roughness roughness=Roughness::LinearPerceptual;
    Color color=Color::LinearCompositeRadiance;
    bool specularAlbedoIncludesFresnel=true, reverseZ=true, motionIncludesJitter=false;
};
[[nodiscard]] inline bool validSemantics(const Semantics& s) {
    return s.normalSpace==NormalSpace::World && s.normalEncoding==NormalEncoding::SignedUnit &&
        s.motionUnits==MotionUnits::InputPixels && s.motionDirection==MotionDirection::CurrentToPrevious &&
        s.roughness==Roughness::LinearPerceptual && s.color==Color::LinearCompositeRadiance &&
        s.specularAlbedoIncludesFresnel && s.reverseZ && !s.motionIncludesJitter;
}
struct Extent {
    u32 inputWidth=0,inputHeight=0,outputWidth=0,outputHeight=0;
    bool operator==(const Extent&)const=default;
};
[[nodiscard]] inline bool validExtent(Extent e,float minimumScale=1,float maximumScale=3) {
    if(!e.inputWidth||!e.inputHeight||!e.outputWidth||!e.outputHeight ||
       e.inputWidth>16384||e.inputHeight>16384||e.outputWidth>16384||e.outputHeight>16384 ||
       !std::isfinite(minimumScale)||!std::isfinite(maximumScale)||minimumScale<=0||maximumScale<minimumScale)
        return false;
    const float x=float(e.outputWidth)/float(e.inputWidth),y=float(e.outputHeight)/float(e.inputHeight);
    return x>=minimumScale && x<=maximumScale && y>=minimumScale && y<=maximumScale;
}
enum class Channel : u8 { Color,Depth,Motion,DiffuseAlbedo,SpecularAlbedo,Normal,Roughness,HitDistance,Reactive,Strength,Exposure,Output,RestoredOutput,Count };
inline constexpr std::array<rg::Format,size_t(Channel::Count)> Formats{
    rg::Format::RGBA16Float,rg::Format::Depth32Float,rg::Format::RG32Float,
    rg::Format::RGBA16Float,rg::Format::RGBA16Float,rg::Format::RGBA16Float,
    rg::Format::R16Float,rg::Format::R32Float,rg::Format::R8Unorm,rg::Format::R8Unorm,
    rg::Format::R16Float,rg::Format::RGBA16Float,rg::Format::RGBA32Float};
inline constexpr std::array<std::string_view,size_t(Channel::Count)> Names{
    "MetalFX noisy linear color","MetalFX reverse-Z depth","MetalFX input-pixel motion",
    "MetalFX diffuse albedo","MetalFX Fresnel specular albedo","MetalFX signed world normal",
    "MetalFX linear roughness","MetalFX world specular distance","MetalFX reactive mask",
    "MetalFX skip-denoise mask","MetalFX unit exposure","MetalFX denoised HDR output","MetalFX restored physical radiance"};
[[nodiscard]] inline rg::Format format(Channel c){return Formats[size_t(c)];}
[[nodiscard]] inline std::string_view name(Channel c){return Names[size_t(c)];}
inline constexpr float MaximumHalf=65504.f;
struct GuideSample {
    glm::vec3 color{},normal{0,0,1},diffuseAlbedo{},specularAlbedo{};
    glm::vec2 motion{};
    float roughness=0.5f,hitDistance=0,depth=1,reactive=0,strength=0;
};
enum Error : u32 { ColorError=1,NormalError=2,AlbedoError=4,RoughnessError=8,MotionError=16,HitError=32,MaskError=64,DepthError=128 };
[[nodiscard]] inline bool finite(glm::vec3 v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);}
[[nodiscard]] inline u32 validateSample(const GuideSample& s,float normalTolerance=0.002f,float colorScale=1) {
    u32 result=0;
    const auto packed=s.color*colorScale;
    if(!std::isfinite(colorScale)||colorScale<=0||!finite(packed)||glm::any(glm::lessThan(packed,glm::vec3(0)))||
       glm::any(glm::greaterThan(packed,glm::vec3(MaximumHalf))))result|=ColorError;
    if(!std::isfinite(s.depth)||s.depth<0||s.depth>1)result|=DepthError;
    if(s.depth>0 && (!finite(s.normal)||std::abs(glm::dot(s.normal,s.normal)-1)>normalTolerance))result|=NormalError;
    if(!finite(s.diffuseAlbedo)||!finite(s.specularAlbedo)||
       glm::any(glm::lessThan(s.diffuseAlbedo,glm::vec3(0)))||glm::any(glm::greaterThan(s.diffuseAlbedo,glm::vec3(1)))||
       glm::any(glm::lessThan(s.specularAlbedo,glm::vec3(0)))||glm::any(glm::greaterThan(s.specularAlbedo,glm::vec3(1))))result|=AlbedoError;
    if(!std::isfinite(s.roughness)||s.roughness<0||s.roughness>1)result|=RoughnessError;
    if(!std::isfinite(s.motion.x)||!std::isfinite(s.motion.y))result|=MotionError;
    if(!std::isfinite(s.hitDistance)||s.hitDistance<0)result|=HitError;
    if(!std::isfinite(s.reactive)||!std::isfinite(s.strength)||s.reactive<0||s.reactive>1||s.strength<0||s.strength>1)result|=MaskError;
    return result;
}
// F8's raster displacement and motion are already INPUT pixels, +Y down.
// Keep the existing validated F8 adapter convention: scale=(1,1), same jitter.
[[nodiscard]] inline glm::vec2 sdkMotionScale(){return {1,1};}
[[nodiscard]] inline glm::vec2 sdkJitter(glm::vec2 rasterDisplacement){return rasterDisplacement;}
[[nodiscard]] inline glm::vec3 scaleInputRadiance(glm::vec3 physical,float preExposure){return physical*preExposure;}
// This mapping applies ONLY if the tester confirms SDK output preserves the
// input pre-exposure scale. It is not inferred from the SDK's input-division API.
[[nodiscard]] inline glm::vec3 restorePreExposedRadiance(glm::vec3 sdkOutput,float preExposure){return sdkOutput/preExposure;}

// A portable selection model for request supersession/lifecycle tests; an old
// future result is never installed merely because its view ID matches.
struct RequestKey {
    Extent extent{};
    u32 pipelineGeneration=0,flags=0;
    bool operator==(const RequestKey&)const=default;
};
[[nodiscard]] inline bool acceptsResult(const RequestKey& requested,const RequestKey& desired){return requested==desired;}
} // namespace phosphor::metalfx_denoise
