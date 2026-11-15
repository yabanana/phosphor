// Indexed forward reference. The visibility resolve shares material semantics.
#include "material_shading.h"
#include "renderer/emission_pack.h"
#include "temporal_motion.h"

static float4x4 loadMatrix(const device float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

static float4x4 loadMatrix(constant float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

#define PLN_VC 0
#define PLN_OUT VertexOut
#define PLN_NAME(x) x
#include "forward_plain.inc"
#undef PLN_VC
#undef PLN_OUT
#undef PLN_NAME
#define PLN_VC 1
#define PLN_OUT VertexOutVC
#define PLN_NAME(x) x##_vc
#include "forward_plain.inc"
#undef PLN_VC
#undef PLN_OUT
#undef PLN_NAME

struct ShadowReceiverOutput {
    float4 point [[color(0)]];
    half4 geometricNormal [[color(1)]];
    uint4 key [[color(2)]];
};

struct ForwardSurfaceOutput {
    half4 color [[color(0)]];
    half4 normalRoughness [[color(1)]];
    half4 diffuseAlbedo [[color(2)]];
    half4 specularAlbedo [[color(3)]];
    half2 motion [[color(4)]];
    half reactive [[color(5)]];
    float4 sss [[color(6), function_constant(kSssWrite)]]; // F15.1 (ticket 17): the diffuse-irradiance target (volumetric subsurface)
};

// ICB pipelines inherit buffers. Lighting textures use the same direct
// resource-ID encoding as the material bindless table, with read-only handles.
struct ForwardLightingTextureHandle {
    texture2d<float, access::read> tex;
};
static_assert(sizeof(ForwardLightingTextureHandle) == 8, "Forward lighting texture handle ABI");

// Lighting residual must retain physical radiance above binary16 range.
// The legacy forward_surface_fs retains ForwardSurfaceOutput and its HALF ABI.
struct ForwardLitSurfaceOutput {
    float4 color [[color(0)]];
    half4 normalRoughness [[color(1)]];
    half4 diffuseAlbedo [[color(2)]];
    half4 specularAlbedo [[color(3)]];
    half2 motion [[color(4)]];
    half reactive [[color(5)]];
    float4 sss [[color(6), function_constant(kSssWrite)]]; // F15.1 (ticket 17)
};

struct DIReceiverOutput {
    half4 shading [[color(0)]];
    half4 albedo [[color(1)]];
    float2 motion [[color(2)]];
    half occlusion [[color(3)]];
    uint emission [[color(4)]]; // F13.1: GPUDISurface::pad[1] of indexed overflow receivers
};

// ---- surface stages: the plain entry points and the vertex-colour twins (forward_surface.inc) ------------------------------------------------
#define SFV_VC 0
#define SFV_OUT SurfaceVertexOut
#define SFV_NAME(x) x
#include "forward_surface.inc"
#undef SFV_VC
#undef SFV_OUT
#undef SFV_NAME
#define SFV_VC 1
#define SFV_OUT SurfaceVertexOutVC
#define SFV_NAME(x) x##_vc
#include "forward_surface.inc"
#undef SFV_VC
#undef SFV_OUT
#undef SFV_NAME
