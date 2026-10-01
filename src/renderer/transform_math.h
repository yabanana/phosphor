#pragma once

// ---------------------------------------------------------------------------
// F5.2 -- transform math shared by C++ (the CPU reference, tests) and MSL
// (shaders/transforms.metal): the same operations in the same order on both
// sides, so the GPU result is bit-identical to the CPU one.
//
// Matrices are 16 floats, column-major (glm / GPUInstance::modelMatrix).
//
//  * mat4Mul: glm's operator*(mat4, mat4) summation order,
//        R[j][i] = ((A[0][i] B[j][0] + A[1][i] B[j][1]) + A[2][i] B[j][2]) + A[3][i] B[j][3],
//    with floating-point contraction OFF.  Measured in spike S6: this clang
//    does not fuse glm's product, and `#pragma METAL fp contract(off)` makes
//    the MSL product produce the same bits; without the pragma the Metal
//    compiler fuses it (an fma chain) and the world matrices differ in the
//    last bit from the CPU's.  Every function below sets the pragma itself
//    (a function-scope pragma does not leak into the includer).
//
//  * motionWorld: the procedural motion of a root (F5.6).  sin/cos are NOT
//    evaluated on the GPU: the CPU computes the (sin a, cos a) of each speed
//    class once per frame (motionSinCosTable, in double, then cast to float)
//    and uploads the table; both sides then only multiply and add.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"

#ifdef __METAL_VERSION__
#define PHOSPHOR_TM_THREAD thread
#define PHOSPHOR_FP_STRICT _Pragma("METAL fp contract(off)")
#else
#include <cmath>
#define PHOSPHOR_TM_THREAD
#define PHOSPHOR_FP_STRICT _Pragma("clang fp contract(off)")
#endif

namespace phosphor {

/// r = a * b (column-major 4x4, glm's summation order).  `r` must not alias
/// `a` or `b`.
inline void mat4Mul(PHOSPHOR_TM_THREAD const float* a, PHOSPHOR_TM_THREAD const float* b,
                    PHOSPHOR_TM_THREAD float* r) {
    PHOSPHOR_FP_STRICT
    for (u32 j = 0; j < 4; ++j) {
        const float b0 = b[4 * j + 0], b1 = b[4 * j + 1], b2 = b[4 * j + 2], b3 = b[4 * j + 3];
        for (u32 i = 0; i < 4; ++i) {
            r[4 * j + i] = a[0 + i] * b0 + a[4 + i] * b1 + a[8 + i] * b2 + a[12 + i] * b3;
        }
    }
}

/// World matrix of a moving root:
///   world = translate(orbit) * rotateY(a) * base,
///   orbit = centre + (r (cos a cosP - sin a sinP), h, r (sin a cosP + cos a sinP))
/// with (sinA, cosA) of the motion's speed class.  `base` is 3x4 (three basis
/// columns + translation, last row 0 0 0 1), so rotateY(a) * base is exactly
///   x' = c x + s z,  y' = y,  z' = c z - s x   (per column, translation included)
/// and the orbit is added to the translation.
inline void motionWorld(PHOSPHOR_TM_THREAD const GPUMotion& m, float sinA, float cosA,
                        PHOSPHOR_TM_THREAD float* out) {
    PHOSPHOR_FP_STRICT
    const float s = sinA, c = cosA;
    for (u32 j = 0; j < 4; ++j) {
        const float x = m.base[3 * j + 0], y = m.base[3 * j + 1], z = m.base[3 * j + 2];
        out[4 * j + 0] = c * x + s * z;
        out[4 * j + 1] = y;
        out[4 * j + 2] = c * z - s * x;
        out[4 * j + 3] = j == 3 ? 1.0f : 0.0f;
    }
    const float cp = m.cosPhase, sp = m.sinPhase;
    out[12] += m.centre[0] + m.radius * (c * cp - s * sp);
    out[13] += m.centre[1] + m.height;
    out[14] += m.centre[2] + m.radius * (s * cp + c * sp);
}

#ifndef __METAL_VERSION__

/// Angular speed in rad/s of speed class k: a fixed ramp 0.05 .. 1.55 rad/s
/// (0.05 + 1.5 k / (SCENE_MOTION_CLASSES - 1)), odd classes rotate backwards.
inline double motionClassSpeed(u32 k) {
    const double v = 0.05 + 1.5 * static_cast<double>(k) / static_cast<double>(SCENE_MOTION_CLASSES - 1);
    return (k & 1u) ? -v : v;
}

/// Fills out[2k], out[2k+1] = sin, cos of angle(k) = t * speed(k): evaluated in
/// double and cast to float once, so the CPU reference and the GPU (which gets
/// this table uploaded in GPUMotionFrame::sinCos) use identical values.
inline void motionSinCosTable(double timeSeconds, float out[SCENE_MOTION_CLASSES * 2]) {
    for (u32 k = 0; k < SCENE_MOTION_CLASSES; ++k) {
        const double a = timeSeconds * motionClassSpeed(k);
        out[2 * k + 0] = static_cast<float>(std::sin(a));
        out[2 * k + 1] = static_cast<float>(std::cos(a));
    }
}

#endif

} // namespace phosphor
