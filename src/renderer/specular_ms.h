#pragma once
// F15 closure: multiple-scattering energy compensation of the GGX specular lobes (scalar maths shared by C++ and MSL; the float3 wrappers
// and the hooks are in shaders/materials/bsdf_energy.h, the independent reference in tools/f15_ms.py).
//
// Model: Kulla and Conty, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH 2017 course notes (the same energy split as
// Fdez-Aguera, "A Multiple-Scattering Microfacet Model for Real-Time Image-Based Lighting", JCGT 8(1) 2019, which gives the environment
// form below).  With L(mu) = 1 - E(mu) the energy the single-scattering lobe D Vis (Fresnel 1) loses (renderer/ms_table.h, integrated
// deterministically: the split-sum DFG table is too coarse for a quantity of 1e-3 .. 1e-1), E_avg = 2 int E mu dmu (stored as the loss
// 1 - E_avg) and the cosine-weighted average Fresnel of the Schlick curve F_avg = f0 + (f90 - f0) / 21 (exact: 2 int mu (1 - mu)^5 dmu =
// 1/21):
//   F_ms          = F_avg^2 E_avg / (1 - F_avg (1 - E_avg))          (colour of the energy that scatters more than once)
//   f_ms(o, i)    = F_ms L(mu_o) L(mu_i) / (pi (1 - E_avg))          (reciprocal, added to the lobe)
//   environment   = F_ms L(mu_o)   (constant radiance 1: the integral of f_ms cos over the hemisphere, exact because
//                                    2 int L mu dmu = 1 - E_avg holds for the tables by construction)
// A white furnace on a rough metal (f0 = f90 = 1, F_ms = 1) then returns E + L = 1; for F < 1 the lobe never gains energy beyond
// F_ms <= F_avg^2 (a dielectric F0 0.04: 0.2 % at most).
#include "renderer/gpu_types.h"
#include "renderer/ms_table.h"
#if defined(__METAL_VERSION__)
#define PHOSPHOR_MS_EXP metal::exp // the header is included before `using namespace metal` (bsdf.h)
#else
#include <cmath>
#define PHOSPHOR_MS_EXP std::exp
#endif
namespace phosphor {
#if defined(__METAL_VERSION__)
#define PHOSPHOR_MS_TABLE const constant float*
#else
#define PHOSPHOR_MS_TABLE const float*
#endif
/// Lower bound of 1 - E_avg (float rounding guard only: a near mirror has 1 - E_avg ~ 1e-5, and L(mu) is of the same order there).
PHOSPHOR_GPU_CONSTANT float MsMinOneMinusEavg = 1e-7f;

/// 1 - E_avg(roughness): the table stores ln(1 - E_avg) (a power law of the roughness); `logLoss` = MsLogLossAvgTable or a GPU copy.
inline float msLossAvgLookup(PHOSPHOR_MS_TABLE logLoss, float roughness) {
    const float y = (roughness > 0 ? (roughness < 1 ? roughness : 1.0f) : 0.0f) * float(MsTableSize - 1);
    const u32 y0 = u32(y) < MsTableSize - 1 ? u32(y) : MsTableSize - 2;
    const float fy = y - float(y0);
    return PHOSPHOR_MS_EXP(logLoss[y0] * (1.0f - fy) + logLoss[y0 + 1] * fy);
}
/// L(mu, roughness) = 1 - E: bilinear lookup of ln L (the table, `logLost` = MsLogLostTable or a GPU copy), the rule of dfgTableLookup,
/// exponentiated: L spans four decades over the table and a linear interpolation of L itself is 12 % off at roughness 0.12.
inline float msLostLookup(PHOSPHOR_MS_TABLE logLost, float mu, float roughness) {
    const float n = float(MsTableSize - 1);
    const float x = (mu > 0 ? (mu < 1 ? mu : 1.0f) : 0.0f) * n, y = (roughness > 0 ? (roughness < 1 ? roughness : 1.0f) : 0.0f) * n;
    const u32 x0 = u32(x) < MsTableSize - 1 ? u32(x) : MsTableSize - 2, y0 = u32(y) < MsTableSize - 1 ? u32(y) : MsTableSize - 2;
    const float fx = x - float(x0), fy = y - float(y0);
    const float v00 = logLost[y0 * MsTableSize + x0], v10 = logLost[y0 * MsTableSize + x0 + 1];
    const float v01 = logLost[(y0 + 1) * MsTableSize + x0], v11 = logLost[(y0 + 1) * MsTableSize + x0 + 1];
    return PHOSPHOR_MS_EXP((v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy);
}
inline float msFavg(float f0, float f90) { return f0 + (f90 - f0) * (1.0f / 21.0f); }
/// F_ms of one channel for the loss 1 - E_avg (E_avg = 1 - loss).
inline float msFms(float favg, float lossAvg) {
    const float eavg = 1.0f - lossAvg;
    const float denominator = 1.0f - favg * lossAvg;
    return favg * favg * eavg / (denominator > 1e-4f ? denominator : 1e-4f);
}
inline float msOneMinusEavg(float lossAvg) { return lossAvg > MsMinOneMinusEavg ? lossAvg : MsMinOneMinusEavg; }
/// Environment weight F_ms L(mu_o) of one colour channel (f0, f90 of that channel).
inline float msEnvWeight(float f0, float f90, float lossAvg, float lostView) { return msFms(msFavg(f0, f90), lossAvg) * lostView; }
/// f_ms of one light sample for one channel: `lostView` = L(mu_o), `lostLight` = L(mu_i); the cosine is the caller's.
inline float msLobeValue(float f0, float f90, float lossAvg, float lostView, float lostLight) {
    return msFms(msFavg(f0, f90), lossAvg) * lostView * lostLight / (3.14159265358979323846f * msOneMinusEavg(lossAvg));
}
} // namespace phosphor
