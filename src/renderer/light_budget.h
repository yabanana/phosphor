#pragma once

// ---------------------------------------------------------------------------
// F15 (GPU safety) -- per-pixel local-light budget of the clustered direct-lighting fallback, shared by C++ (tests) and MSL
// (shaders/light_cluster.metal).
//
// The clustered shading kernels shade EVERY light of the cell list, and of ALL lights when the cell overflowed (physically
// unbounded emissive triangles have range 0 and intersect every cell).  A mesh of ~75k emissive triangles therefore costs
// pixels x 75k visibility rays and frame 0 never completes (ticket 20).  The budget bounds the per-pixel work:
//
//  * a list of n <= lightShadeBudget() lights is shaded exactly, as before (the identity: bit-identical results);
//  * a longer list is shaded by STRATIFIED sampling: it is cut into m = lightShadeBudget() contiguous strata of near-equal
//    size, ONE light is drawn uniformly inside each stratum, and its contribution is multiplied by the stratum size
//    (the inverse of its selection probability).  Every light has selection probability 1 / size(stratum), so the
//    estimator  sum_j size_j * f(light_j)  is UNBIASED for  sum_i f(light_i)  for any stratum partition and any fixed
//    uniform draw; the per-pixel cost never exceeds m lights.  The variance is higher than the exact sum (declared:
//    uniform, not power-weighted, selection); the Monte Carlo noise is not hidden by any clamp.
// ---------------------------------------------------------------------------

#ifdef __METAL_VERSION__
#define PHOSPHOR_LB_UINT uint
#define PHOSPHOR_LB_OUT thread uint&
#else
#include <cstdint>
#define PHOSPHOR_LB_UINT std::uint32_t
#define PHOSPHOR_LB_OUT std::uint32_t&
#endif

namespace phosphor {

/// Most local lights one pixel shades in the clustered fallback.
inline PHOSPHOR_LB_UINT lightShadeBudget() { return 128u; }

/// Lights shaded for a list of `listCount` entries.
inline PHOSPHOR_LB_UINT lightBudgetCount(PHOSPHOR_LB_UINT listCount) {
    return listCount < lightShadeBudget() ? listCount : lightShadeBudget();
}

/// Stratum j of m over a list of n entries (m = lightBudgetCount(n) < n): the half-open range [begin, begin + size).
/// Strata partition [0, n) exactly, sizes differ by at most one, none is empty.  `u` in [0, 1) picks the entry.
inline void lightBudgetPick(PHOSPHOR_LB_UINT n, PHOSPHOR_LB_UINT m, PHOSPHOR_LB_UINT j, float u,
                            PHOSPHOR_LB_OUT index, PHOSPHOR_LB_OUT size) {
    // 64-bit products: j * n cannot overflow for any list length.
#ifdef __METAL_VERSION__
    const PHOSPHOR_LB_UINT begin = PHOSPHOR_LB_UINT((ulong(j) * ulong(n)) / ulong(m));
    const PHOSPHOR_LB_UINT end = PHOSPHOR_LB_UINT((ulong(j + 1u) * ulong(n)) / ulong(m));
#else
    const PHOSPHOR_LB_UINT begin = PHOSPHOR_LB_UINT((std::uint64_t(j) * n) / m);
    const PHOSPHOR_LB_UINT end = PHOSPHOR_LB_UINT((std::uint64_t(j + 1u) * n) / m);
#endif
    size = end - begin;
    PHOSPHOR_LB_UINT offset = PHOSPHOR_LB_UINT(u * float(size));
    if (offset >= size) offset = size - 1u;
    index = begin + offset;
}

} // namespace phosphor

#undef PHOSPHOR_LB_UINT
#undef PHOSPHOR_LB_OUT
