#pragma once
// F15 closure (--specular-ms, function constant SPECULAR_MS / runtime flags): multiple-scattering energy compensation of the GGX
// specular lobes, Kulla and Conty (SIGGRAPH 2017 course notes) with the environment form of Fdez-Aguera (JCGT 2019).  The scalar
// maths and the derivation are in renderer/specular_ms.h (shared with C++); this file only wraps them for the BSDF's float3 colours.
// Included at the end of bsdf.h: every consumer (the resolve in material_shading.h, DI in restir_common.h, the composite) adds the
// lobe as an EXTRA term next to the single-scattering terms, so a scene with the option off runs the base expressions unchanged.
//
// The tables (renderer/ms_table.h) follow the DFG table in the same GPU buffer: BSDF_MS_LOSS_AVG(frame.dfgTable) is 1 - E_avg per
// roughness, BSDF_MS_LOST(frame.dfgTable) the lost-energy table L(mu, roughness); compute kernels use phosphor::MsLogLossAvgTable and
// phosphor::MsLogLostTable (program-scope arrays: only fragment functions need the buffer, see bsdf.h).
#include "renderer/specular_ms.h"

#define BSDF_MS_LOSS_AVG(dfg) ((dfg) + phosphor::DfgTableSize * phosphor::DfgTableSize * 2u)
#define BSDF_MS_LOST(dfg) (BSDF_MS_LOSS_AVG(dfg) + phosphor::MsTableSize)
// F15.1 (ticket 11, round 3): the anisotropic albedo table (renderer/aniso_table.h) follows the lost-energy table in the same buffer.
#define BSDF_ANISO_TABLE(dfg) (BSDF_MS_LOST(dfg) + phosphor::MsTableSize * phosphor::MsTableSize)

struct BsdfMs {
    float3 fms;         // F_ms of the pixel's lobe (colour of the energy that scatters more than once)
    float lostView;     // L(mu_o) = 1 - E(mu_o)
    float roughness;    // L(mu_i) is looked up per light sample at this roughness
    float inverseNorm;  // 1 / (pi (1 - E_avg))
};

// f0 / f90: the Fresnel endpoints of the specular lobe (mix(dielectric, base colour, metallic) for f0, mix(f90, 1, metallic) for f90).
inline BsdfMs bsdfMsPrepare(float3 f0, float f90, float roughness, float NdotV, const constant float* lossAvg, const constant float* lost) {
    BsdfMs m;
    const float loss = phosphor::msLossAvgLookup(lossAvg, roughness);
    m.fms = float3(phosphor::msFms(phosphor::msFavg(f0.x, f90), loss), phosphor::msFms(phosphor::msFavg(f0.y, f90), loss),
                   phosphor::msFms(phosphor::msFavg(f0.z, f90), loss));
    m.lostView = phosphor::msLostLookup(lost, NdotV, roughness);
    m.roughness = roughness;
    m.inverseNorm = 1.0f / (M_PI_F * phosphor::msOneMinusEavg(loss));
    return m;
}
// f_ms of one light sample (without the cosine, the caller's).
inline float3 bsdfMsLobe(BsdfMs m, float NdotL, const constant float* lost) {
    return m.fms * (m.lostView * phosphor::msLostLookup(lost, NdotL, m.roughness) * m.inverseNorm);
}
// The integral of f_ms cos over the hemisphere in a constant radiance 1 (exact for the tables: 2 int L mu dmu = 1 - E_avg).
inline float3 bsdfMsEnv(BsdfMs m) { return m.fms * m.lostView; }
