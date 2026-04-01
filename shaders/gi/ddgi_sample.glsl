#ifndef DDGI_SAMPLE_GLSL
#define DDGI_SAMPLE_GLSL

// ---------------------------------------------------------------------------
// DDGI Irradiance Sampling — Include File (NOT compiled standalone)
//
// Provides sampleDDGI() which, given a world position and normal, returns
// the interpolated irradiance from the surrounding 8 probes using:
//   - Trilinear interpolation based on fractional grid position
//   - Normal-based cosine weighting (probeToSurface dot normal)
//   - Chebyshev visibility test (variance shadow map style)
//
// Requires:
//   - DDGIUniforms struct to be declared (or included before this file)
//   - Irradiance and visibility atlases bound as sampler2D
// ---------------------------------------------------------------------------

// -- Constants matching the update shaders -----------------------------------

const int DDGI_IRRADIANCE_PROBE_SIZE   = 8;
const int DDGI_IRRADIANCE_WITH_BORDER  = DDGI_IRRADIANCE_PROBE_SIZE + 2;
const int DDGI_VISIBILITY_PROBE_SIZE   = 16;
const int DDGI_VISIBILITY_WITH_BORDER  = DDGI_VISIBILITY_PROBE_SIZE + 2;

// ---------------------------------------------------------------------------
// Atlas UV helpers
// ---------------------------------------------------------------------------

/// Compute the UV coordinates in the tiled atlas for a given probe index
/// and direction (encoded as octahedral UV in [0,1]^2).
vec2 ddgiProbeAtlasUV_irradiance(uint probeIndex, vec2 octUV, ivec3 gridDims, vec2 atlasSize) {
    int ix = int(probeIndex) % gridDims.x;
    int iy = (int(probeIndex) / gridDims.x) % gridDims.y;
    int iz = int(probeIndex) / (gridDims.x * gridDims.y);

    int col = ix + iz * gridDims.x;
    int row = iy;

    // Texel coordinate within the bordered probe tile
    vec2 texel = vec2(col * DDGI_IRRADIANCE_WITH_BORDER,
                      row * DDGI_IRRADIANCE_WITH_BORDER)
               + vec2(1.0)  // skip border
               + octUV * float(DDGI_IRRADIANCE_PROBE_SIZE);

    return texel / atlasSize;
}

vec2 ddgiProbeAtlasUV_visibility(uint probeIndex, vec2 octUV, ivec3 gridDims, vec2 atlasSize) {
    int ix = int(probeIndex) % gridDims.x;
    int iy = (int(probeIndex) / gridDims.x) % gridDims.y;
    int iz = int(probeIndex) / (gridDims.x * gridDims.y);

    int col = ix + iz * gridDims.x;
    int row = iy;

    vec2 texel = vec2(col * DDGI_VISIBILITY_WITH_BORDER,
                      row * DDGI_VISIBILITY_WITH_BORDER)
               + vec2(1.0)
               + octUV * float(DDGI_VISIBILITY_PROBE_SIZE);

    return texel / atlasSize;
}

// ---------------------------------------------------------------------------
// Octahedral encode: direction -> [0,1]^2
// ---------------------------------------------------------------------------

vec2 ddgiOctEncode(vec3 n) {
    vec3 a = n / (abs(n.x) + abs(n.y) + abs(n.z));
    if (a.z < 0.0) {
        a.xy = (1.0 - abs(a.yx)) * vec2(a.x >= 0.0 ? 1.0 : -1.0,
                                          a.y >= 0.0 ? 1.0 : -1.0);
    }
    return a.xy * 0.5 + 0.5; // map to [0,1]
}

// ---------------------------------------------------------------------------
// Chebyshev visibility test
//
// Given the mean distance and mean distance-squared stored in the visibility
// atlas, compute the probability that the surface at 'dist' from the probe
// is actually visible (not occluded).
// ---------------------------------------------------------------------------

float ddgiChebyshevWeight(float meanDist, float meanDist2, float dist) {
    // If the surface is closer than the mean distance, it's certainly visible
    if (dist <= meanDist) {
        return 1.0;
    }

    // Variance
    float variance = abs(meanDist2 - meanDist * meanDist);
    // Clamp to avoid division by zero; small minimum variance
    variance = max(variance, 1e-4);

    float diff = dist - meanDist;
    // One-tailed Chebyshev inequality: P(X >= dist) <= variance / (variance + diff^2)
    float weight = variance / (variance + diff * diff);

    // Clamp to [0,1] and apply a soft threshold to reduce light leaking
    return max(weight * weight * weight, 0.0);
}

// ---------------------------------------------------------------------------
// sampleDDGI — main entry point
//
// @param worldPos          Surface position in world space
// @param normal            Surface normal (unit length)
// @param ddgi              DDGI uniform parameters
// @param irradianceAtlas   Sampler for the irradiance atlas (filtered)
// @param visibilityAtlas   Sampler for the visibility atlas (filtered)
// @return                  Irradiance (pre-divided by pi) at the surface
// ---------------------------------------------------------------------------

struct DDGISampleUniforms {
    ivec3  probeGridDims;
    float  probeSpacing;
    vec3   probeGridOrigin;
    float  maxRayDistance;
    uint   raysPerProbe;
    float  hysteresis;
    float  irradianceGamma;
    float  pad;
};

vec3 sampleDDGI(vec3 worldPos, vec3 normal,
                DDGISampleUniforms params,
                sampler2D irradianceAtlas, sampler2D visibilityAtlas)
{
    ivec3 gridDims   = params.probeGridDims;
    float spacing    = params.probeSpacing;
    vec3  gridOrigin = params.probeGridOrigin;

    // Atlas dimensions (in texels)
    vec2 irradianceAtlasSize = vec2(
        (gridDims.x * gridDims.z) * DDGI_IRRADIANCE_WITH_BORDER,
        gridDims.y * DDGI_IRRADIANCE_WITH_BORDER);

    vec2 visibilityAtlasSize = vec2(
        (gridDims.x * gridDims.z) * DDGI_VISIBILITY_WITH_BORDER,
        gridDims.y * DDGI_VISIBILITY_WITH_BORDER);

    // Fractional grid position of the surface point
    vec3 gridPos = (worldPos - gridOrigin) / spacing;

    // Base probe index (floor)
    ivec3 baseIdx = ivec3(floor(gridPos));
    baseIdx = clamp(baseIdx, ivec3(0), gridDims - ivec3(2));

    // Fractional offset for trilinear interpolation
    vec3 alpha = clamp(gridPos - vec3(baseIdx), vec3(0.0), vec3(1.0));

    // Bias the surface position slightly along the normal to reduce
    // self-shadowing artefacts.
    vec3 biasedWorldPos = worldPos + normal * 0.2 * spacing;

    // Accumulate weighted irradiance from the 8 surrounding probes
    vec3  totalIrradiance = vec3(0.0);
    float totalWeight     = 0.0;

    for (int i = 0; i < 8; ++i) {
        // Compute the offset of this corner in the 2x2x2 cube
        ivec3 offset = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 probeCoord = baseIdx + offset;

        // Clamp to valid range
        probeCoord = clamp(probeCoord, ivec3(0), gridDims - ivec3(1));

        // Linear probe index
        uint probeIndex = uint(probeCoord.x)
                        + uint(probeCoord.y) * uint(gridDims.x)
                        + uint(probeCoord.z) * uint(gridDims.x) * uint(gridDims.y);

        // Probe world position
        vec3 probePos = gridOrigin + vec3(probeCoord) * spacing;

        // Direction from probe to surface
        vec3 probeToSurface = biasedWorldPos - probePos;
        float distToProbe   = length(probeToSurface);
        vec3  dirToSurface  = (distToProbe > 1e-6)
                             ? probeToSurface / distToProbe
                             : normal;

        // --- Weight 1: Normal-based (backface rejection) ---
        // Probes behind the surface contribute little.
        float normalWeight = max(0.0001, dot(dirToSurface, normal));

        // --- Weight 2: Chebyshev visibility ---
        vec2 visOctUV = ddgiOctEncode(-dirToSurface);
        vec2 visUV    = ddgiProbeAtlasUV_visibility(probeIndex, visOctUV,
                                                     gridDims, visibilityAtlasSize);
        vec2 visMoments = texture(visibilityAtlas, visUV).rg;
        float visWeight = ddgiChebyshevWeight(visMoments.x, visMoments.y, distToProbe);

        // --- Weight 3: Trilinear ---
        vec3 triWeights = mix(vec3(1.0) - alpha, alpha, vec3(offset));
        float triWeight = triWeights.x * triWeights.y * triWeights.z;

        // Combined weight
        float weight = normalWeight * visWeight * triWeight;
        weight = max(weight, 1e-7);  // avoid zero-weight probes

        // Sample irradiance
        vec2 irrOctUV = ddgiOctEncode(normal);
        vec2 irrUV    = ddgiProbeAtlasUV_irradiance(probeIndex, irrOctUV,
                                                     gridDims, irradianceAtlasSize);
        vec3 irradiance = texture(irradianceAtlas, irrUV).rgb;

        // Undo perceptual gamma encoding
        if (params.irradianceGamma > 0.0) {
            irradiance = pow(max(irradiance, vec3(0.0)), vec3(params.irradianceGamma));
        }

        totalIrradiance += irradiance * weight;
        totalWeight     += weight;
    }

    if (totalWeight > 0.0) {
        totalIrradiance /= totalWeight;
    }

    return totalIrradiance;
}

#endif // DDGI_SAMPLE_GLSL
