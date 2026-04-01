#ifndef BRDF_GLSL
#define BRDF_GLSL

// Depends on math.glsl for PI and EPSILON
#ifndef MATH_GLSL
#include "math.glsl"
#endif

// ---------------------------------------------------------------------------
// Cook-Torrance GGX Microfacet BRDF
// ---------------------------------------------------------------------------

// GGX/Trowbridge-Reitz normal distribution function.
// Approximates the fraction of microfacets aligned with the half-vector H.
float distributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (single direction).
float geometrySchlickGGX(float cosTheta, float k) {
    return cosTheta / (cosTheta * (1.0 - k) + k);
}

// Smith's method: combines geometry obstruction (view) and shadowing (light).
// Uses the direct lighting remapping: k = (roughness + 1)^2 / 8
float geometrySmithGGX(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return geometrySchlickGGX(NdotV, k) * geometrySchlickGGX(NdotL, k);
}

// Fresnel-Schlick approximation.
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float t = 1.0 - cosTheta;
    // Clamp to avoid negative values from precision issues
    t = clamp(t, 0.0, 1.0);
    float t2 = t * t;
    return F0 + (1.0 - F0) * (t2 * t2 * t); // t^5
}

// Fresnel-Schlick with roughness for IBL (environment reflections).
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    float t = 1.0 - cosTheta;
    t = clamp(t, 0.0, 1.0);
    float t2 = t * t;
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * (t2 * t2 * t);
}

// ---------------------------------------------------------------------------
// Full PBR evaluation for a single analytical light
//
// N          - surface normal (unit)
// V          - view direction (unit, surface to camera)
// L          - light direction (unit, surface to light)
// albedo     - base color (linear)
// metallic   - metallic factor [0,1]
// roughness  - roughness factor [0,1], clamped to avoid singularity
// lightColor - pre-multiplied light color * intensity
//
// Returns outgoing radiance towards V.
// ---------------------------------------------------------------------------

vec3 evaluatePBR(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness, vec3 lightColor) {
    // Clamp roughness to avoid division by zero in GGX
    roughness = max(roughness, 0.04);

    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), EPSILON);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Dielectric F0 = 0.04 (typical), metals use albedo as F0
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance specular BRDF: D * G * F / (4 * NdotV * NdotL)
    float D = distributionGGX(NdotH, roughness);
    float G = geometrySmithGGX(NdotV, NdotL, roughness);
    vec3  F = fresnelSchlick(HdotV, F0);

    vec3 numerator   = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + EPSILON;
    vec3 specular    = numerator / denominator;

    // Energy conservation: diffuse gets the energy not reflected
    vec3 kD = (1.0 - F) * (1.0 - metallic);

    // Lambertian diffuse
    vec3 diffuse = kD * albedo * INV_PI;

    return (diffuse + specular) * lightColor * NdotL;
}

#endif // BRDF_GLSL
