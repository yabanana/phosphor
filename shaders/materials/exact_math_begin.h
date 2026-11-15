// F15.2 (ticket 21): bit-exact material shading.  Active ONLY in the exact module (shaders/material_passes_exact.metal, compiled with
// PHOSPHOR_EXACT_MODULE and -fmetal-math-mode=safe); in the base module this file is empty, so featureless scenes keep today's code.
#ifdef PHOSPHOR_EXACT_MODULE
// F15.2 (ticket 21): bit-exact material shading.  A class pipeline (MATERIAL_FEATURES = one feature mask) must resolve a pixel
// bit for bit like the all-lobes pipeline (measured 2026-11-15, M5 Max, ticket 21): under Metal's default fast math the compiler
// contracts, reassociates and picks fast builtins by code shape, so two specialisations of the same source differ by up to 1e-2
// relative (ToyCar).  The three together make the result a function of the source expression only (every one of them was needed:
// safe maths alone, precise builtins alone and contract(off) with precise builtins each left 549..207229 differing pixels):
//   math_mode(safe)   no reassociation / reciprocal / finite-math rewrites;
//   contract(off)     no multiply-add fusion;
//   precise builtins  the unqualified calls in the region below resolve to metal::precise (a using-declaration in the nested
//                     namespace is found before metal's fast overloads).
// Include this file before a region of shading code and exact_math_end.h after it; it is deliberately not include-guarded.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
namespace phx_math {
using metal::precise::acos;
using metal::precise::asin;
using metal::precise::atan;
using metal::precise::atan2;
using metal::precise::cos;
using metal::precise::cosh;
using metal::precise::distance;
using metal::precise::exp;
using metal::precise::exp2;
using metal::precise::exp10;
using metal::precise::fmod;
using metal::precise::length;
using metal::precise::log;
using metal::precise::log2;
using metal::precise::log10;
using metal::precise::normalize;
using metal::precise::pow;
using metal::precise::powr;
using metal::precise::rsqrt;
using metal::precise::sin;
using metal::precise::sinh;
using metal::precise::sqrt;
using metal::precise::tan;
using metal::precise::tanh;
namespace code {
#endif
